import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.pipeline import Pipeline
from scipy import sparse  # for sparse -> dense

# ============================================================
# 0. HELPER: ASSIGN INTERPRETABLE REGIMES
# ============================================================
def assign_regimes(df_algo, cluster_col, allow_noise=False):
    """
    NEW: Assign regimes using a hybrid approach:
    - Forward 30d returns determine bull/bear ordering
    - Technical indicators break ties and determine steady vs high_vol
    """

    # --- Required features ---
    required = [
        "BTC_mom_7d",
        "BTC_vol_7d",
        "BTC_ADX_14",
        "BTC_MACD_hist",
        "BTC_BB_width_20",
        "BTC_fwd_7d_log_ret"
    ]

    for r in required:
        if r not in df_algo.columns:
            raise ValueError(f"Missing required feature for regime scoring: {r}")

    labels = df_algo[cluster_col].values
    unique_clusters = np.unique(labels)

    if allow_noise:
        valid_clusters = [c for c in unique_clusters if c != -1]
    else:
        valid_clusters = [c for c in unique_clusters]

    if len(valid_clusters) < 4:
        print(f"[WARN] Only {len(valid_clusters)} clusters; fallback mapping.")
        df_algo["regime"] = [
            "noise" if (allow_noise and c == -1) else f"cluster_{c}"
            for c in labels
        ]
        return df_algo, None

    # --- Compute cluster summaries ---
    summary = df_algo.groupby(cluster_col)[required].mean()

    # 1) Expansion (Growth) = cluster with highest forward returns
    expansion_cluster = summary["BTC_fwd_7d_log_ret"].idxmax()

    # 2) Contraction (Downtrend) = cluster with lowest forward returns
    contraction_cluster = summary["BTC_fwd_7d_log_ret"].idxmin()

    # 3) Dislocation (High-Volatility) - from remaining clusters
    remaining_for_vol = summary.drop([expansion_cluster, contraction_cluster])
    dislocation_cluster = (remaining_for_vol["BTC_vol_7d"] + remaining_for_vol["BTC_BB_width_20"]).idxmax()

    # 4) Compression (Low-Vol) - last remaining cluster
    remaining_for_comp = remaining_for_vol.drop(dislocation_cluster)
    compression_cluster = (remaining_for_comp["BTC_vol_7d"] + remaining_for_comp["BTC_ADX_14"]).idxmin()

    # Final regime names
    regime_map = {
        expansion_cluster:   "Expansion (Growth)",
        contraction_cluster: "Contraction (Downtrend)",
        dislocation_cluster: "Dislocation (High-Volatility)",
        compression_cluster: "Compression (Low-Volatility)"
    }

    # Verify all clusters are mapped
    all_clusters = set(summary.index)
    mapped_clusters = set(regime_map.keys())
    if all_clusters != mapped_clusters:
        unmapped = all_clusters - mapped_clusters
        print(f"[WARN] Unmapped clusters: {unmapped}")
        # Assign remaining clusters to the closest regime
        for unmapped_cluster in unmapped:
            # Find closest regime by forward returns
            unmapped_ret = summary.loc[unmapped_cluster, "BTC_fwd_7d_log_ret"]
            if unmapped_ret >= summary.loc[expansion_cluster, "BTC_fwd_7d_log_ret"]:
                regime_map[unmapped_cluster] = "Expansion (Growth)"
            elif unmapped_ret <= summary.loc[contraction_cluster, "BTC_fwd_7d_log_ret"]:
                regime_map[unmapped_cluster] = "Contraction (Downtrend)"
            else:
                # Check volatility
                unmapped_vol = summary.loc[unmapped_cluster, "BTC_vol_7d"] + summary.loc[unmapped_cluster, "BTC_BB_width_20"]
                if unmapped_vol >= (summary.loc[dislocation_cluster, "BTC_vol_7d"] + summary.loc[dislocation_cluster, "BTC_BB_width_20"]):
                    regime_map[unmapped_cluster] = "Dislocation (High-Volatility)"
                else:
                    regime_map[unmapped_cluster] = "Compression (Low-Volatility)"

    df_algo["regime"] = [
        "noise" if (allow_noise and c == -1) else regime_map.get(c, f"cluster_{c}")
        for c in labels
    ]

    return df_algo, regime_map


# ============================================================
# 1. LOAD DATA
# ============================================================
INPUT_PATH = "Data/regime_features_hourly_categorical.csv"

df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
df = df.dropna()

print("Loaded hourly features:", df.shape)

# ============================================================
# 2. FEATURE GROUPS (UPDATED TO USE NEW FEATURES)
# ============================================================
possible_numeric_features = [
    # Log returns
    "BTC_log_ret_1h",
    "BTC_log_ret_1d",
    "BTC_log_ret_7d",
    "BTC_log_ret_30d",

    # Simple hourly returns
    "BTC_ret_1h",
    "SPX_ret_1h",

    # Volatility
    "BTC_vol_7d",
    "SPX_vol_7d",

    # Momentum
    "BTC_mom_1d",
    "BTC_mom_7d",
    "SPX_mom_1d",
    "SPX_mom_7d",

    # Range
    "BTC_range_1d",

    # Technical indicators (BTC)
    "BTC_RSI_14",
    "BTC_MACD_line",
    "BTC_MACD_signal",
    "BTC_MACD_hist",
    "BTC_BB_width_20",
    "BTC_ADX_14",

    #US10Y
    'US10Y_chg_7d', 
    'US10Y_vol_7d',
    'US10Y_level_7d_mean'
]

numeric_features = [c for c in possible_numeric_features if c in df.columns]

categorical_features = [
    "hour_of_day_cat",
    "day_of_week_cat",
    "weekend_cat",
    "BTC_vol_regime_7d",
    "SPX_vol_regime_7d",
    "BTC_mom_regime_7d",
    "SPX_mom_regime_7d",
    "BTC_range_regime_1d",
]
categorical_features = [c for c in categorical_features if c in df.columns]

data = df[numeric_features + categorical_features].copy()

# ============================================================
# 3. PREPROCESSOR (ONE-HOT + SCALE)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Fit once, use everywhere
X_transformed = preprocessor.fit_transform(data)
X_dense = X_transformed.toarray() if sparse.issparse(X_transformed) else X_transformed

eval_rows = []  # store metrics for all models

# ============================================================
# 4. KMEANS (4 clusters)
# ============================================================
print("\nRunning KMeans...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
cluster_labels_km = kmeans.fit_predict(X_dense)

df_km = df.copy()
df_km["cluster_kmeans"] = cluster_labels_km

# Metrics
sil_km = silhouette_score(X_dense, cluster_labels_km)
db_km = davies_bouldin_score(X_dense, cluster_labels_km)
ch_km = calinski_harabasz_score(X_dense, cluster_labels_km)
inertia_km = kmeans.inertia_

eval_rows.append(["KMeans (4)", sil_km, db_km, ch_km, inertia_km, "Official regimes"])

# Interpretable regimes (no noise for KMeans)
df_km, regime_map_km = assign_regimes(df_km, "cluster_kmeans", allow_noise=False)
print("\nKMeans regime map:", regime_map_km)

df_km.to_csv("Data/Output/kmeans_regimes.csv")
print("Saved: Data/Output/kmeans_regimes.csv")

# ============================================================
# 5. AGGLOMERATIVE (4 clusters)
# ============================================================
print("\nRunning Agglomerative Clustering...")
agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
cluster_labels_agg = agg.fit_predict(X_dense)

df_agg = df.copy()
df_agg["cluster_agglomerative"] = cluster_labels_agg

sil_agg = silhouette_score(X_dense, cluster_labels_agg)
db_agg = davies_bouldin_score(X_dense, cluster_labels_agg)
ch_agg = calinski_harabasz_score(X_dense, cluster_labels_agg)

eval_rows.append(["Agglomerative (4)", sil_agg, db_agg, ch_agg, np.nan, "Hierarchical"])

# Interpretable regimes (no noise for Agglomerative)
df_agg, regime_map_agg = assign_regimes(df_agg, "cluster_agglomerative", allow_noise=False)
print("\nAgglomerative regime map:", regime_map_agg)

df_agg.to_csv("Data/Output/agglomerative_regimes.csv")
print("Saved: Data/Output/agglomerative_regimes.csv")

# ============================================================
# 6. DBSCAN
# ============================================================
print("\nRunning DBSCAN...")
dbscan = DBSCAN(eps=1.5, min_samples=10, n_jobs=-1)
cluster_labels_db = dbscan.fit_predict(X_dense)

df_db = df.copy()
df_db["cluster_dbscan"] = cluster_labels_db

unique_labels = np.unique(cluster_labels_db)
print("DBSCAN unique labels:", unique_labels)

# Metrics on non-noise points only
valid_mask = cluster_labels_db != -1
valid_clusters = np.unique(cluster_labels_db[valid_mask])

if len(valid_clusters) < 2:
    print("[WARN] Not enough non-noise clusters for DBSCAN metrics.")
    sil_db = db_db = ch_db = np.nan
else:
    X_core = X_dense[valid_mask]
    labels_core = cluster_labels_db[valid_mask]

    sil_db = silhouette_score(X_core, labels_core)
    db_db = davies_bouldin_score(X_core, labels_core)
    ch_db = calinski_harabasz_score(X_core, labels_core)

eval_rows.append(["DBSCAN", sil_db, db_db, ch_db, np.nan, "Density-based; noise=-1"])

# Interpretable regimes (noise -> "noise")
df_db, regime_map_db = assign_regimes(df_db, "cluster_dbscan", allow_noise=True)
print("\nDBSCAN regime map:", regime_map_db)

df_db.to_csv("Data/Output/dbscan_regimes.csv")
print("Saved: Data/Output/dbscan_regimes.csv")

# ============================================================
# 7. METRIC COMPARISON TABLE
# ============================================================
comparison_df = pd.DataFrame(
    eval_rows,
    columns=["Model", "Silhouette", "DB Index", "CH Index", "Inertia", "Notes"],
)

print("\n=== Cluster Evaluation Summary ===")
print(comparison_df)

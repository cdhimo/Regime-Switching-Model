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
best_model = None
best_score = -np.inf
best_df = None
best_regime_map = None

# ============================================================
# 4. KMEANS HYPERPARAMETER TUNING (4 clusters)
# ============================================================
print("\n" + "="*60)
print("TUNING KMEANS (n_clusters=4)")
print("="*60)

kmeans_params = {
    'n_init': [10, 20, 30],
    'max_iter': [300, 500],
    'algorithm': ['lloyd', 'elkan']
}

best_kmeans = None
best_kmeans_score = -np.inf
best_kmeans_params = None
best_kmeans_labels = None

for n_init in kmeans_params['n_init']:
    for max_iter in kmeans_params['max_iter']:
        for algorithm in kmeans_params['algorithm']:
            try:
                kmeans = KMeans(
                    n_clusters=4,
                    random_state=42,
                    n_init=n_init,
                    max_iter=max_iter,
                    algorithm=algorithm
                )
                cluster_labels = kmeans.fit_predict(X_dense)
                
                # Check cluster balance
                unique, counts = np.unique(cluster_labels, return_counts=True)
                min_cluster_size = counts.min()
                max_cluster_size = counts.max()
                balance_ratio = min_cluster_size / max_cluster_size if max_cluster_size > 0 else 0
                
                sil = silhouette_score(X_dense, cluster_labels)
                ch = calinski_harabasz_score(X_dense, cluster_labels)
                db = davies_bouldin_score(X_dense, cluster_labels)
                inertia = kmeans.inertia_
                
                # Combined score: 60% silhouette, 40% CH (normalized), with balance bonus
                ch_norm = ch / 1000
                combined_score = 0.6 * sil + 0.4 * ch_norm + 0.1 * balance_ratio
                
                params_str = f"n_init={n_init}, max_iter={max_iter}, algorithm={algorithm}"
                
                # Always record in evaluation metrics
                eval_rows.append([
                    f"KMeans_{n_init}_{max_iter}_{algorithm}",
                    sil, db, ch, inertia, params_str
                ])
                
                # Check if too imbalanced for best model selection
                if balance_ratio < 0.05:
                    print(f"  {params_str}: Sil={sil:.4f}, CH={ch:.2f}, Balance={balance_ratio:.3f} (SKIPPED for best - too imbalanced)")
                else:
                    print(f"  {params_str}: Sil={sil:.4f}, CH={ch:.2f}, Balance={balance_ratio:.3f}, Combined={combined_score:.4f}")
                    
                    # Only consider balanced clusters for best model
                    if combined_score > best_kmeans_score:
                        best_kmeans_score = combined_score
                        best_kmeans = kmeans
                        best_kmeans_params = params_str
                        best_kmeans_labels = cluster_labels
            except Exception as e:
                print(f"  Error: {e}")
                continue

print(f"\n✓ Best KMeans: {best_kmeans_params}")
print(f"  Score: {best_kmeans_score:.4f}")

df_km = df.copy()
df_km["cluster_kmeans"] = best_kmeans_labels
sil_km = silhouette_score(X_dense, best_kmeans_labels)
ch_km = calinski_harabasz_score(X_dense, best_kmeans_labels)
combined_km = 0.6 * sil_km + 0.4 * (ch_km / 1000)

# Interpretable regimes (no noise for KMeans)
df_km, regime_map_km = assign_regimes(df_km, "cluster_kmeans", allow_noise=False)
print("KMeans regime map:", regime_map_km)

# ============================================================
# 5. AGGLOMERATIVE HYPERPARAMETER TUNING (4 clusters)
# ============================================================
print("\n" + "="*60)
print("TUNING AGGLOMERATIVE CLUSTERING (n_clusters=4)")
print("="*60)

agg_params = {
    'linkage': ['ward', 'complete', 'average', 'single']
}

best_agg = None
best_agg_score = -np.inf
best_agg_params = None
best_agg_labels = None

for linkage in agg_params['linkage']:
    try:
        agg = AgglomerativeClustering(n_clusters=4, linkage=linkage)
        cluster_labels = agg.fit_predict(X_dense)
        
        # Check cluster balance
        unique, counts = np.unique(cluster_labels, return_counts=True)
        min_cluster_size = counts.min()
        max_cluster_size = counts.max()
        balance_ratio = min_cluster_size / max_cluster_size if max_cluster_size > 0 else 0
        
        sil = silhouette_score(X_dense, cluster_labels)
        ch = calinski_harabasz_score(X_dense, cluster_labels)
        db = davies_bouldin_score(X_dense, cluster_labels)
        
        # Combined score with balance bonus
        ch_norm = ch / 1000
        combined_score = 0.6 * sil + 0.4 * ch_norm + 0.1 * balance_ratio
        
        params_str = f"linkage={linkage}"
        
        # Always record in evaluation metrics
        eval_rows.append([
            f"Agglomerative_{linkage}",
            sil, db, ch, np.nan, params_str
        ])
        
        # Check if too imbalanced for best model selection
        if balance_ratio < 0.05:
            print(f"  {params_str}: Sil={sil:.4f}, CH={ch:.2f}, Balance={balance_ratio:.3f} (SKIPPED for best - too imbalanced)")
        else:
            print(f"  {params_str}: Sil={sil:.4f}, CH={ch:.2f}, Balance={balance_ratio:.3f}, Combined={combined_score:.4f}")
            
            # Only consider balanced clusters for best model
            if combined_score > best_agg_score:
                best_agg_score = combined_score
                best_agg = agg
                best_agg_params = params_str
                best_agg_labels = cluster_labels
    except Exception as e:
        print(f"  Error: {e}")
        continue

print(f"\n✓ Best Agglomerative: {best_agg_params}")
print(f"  Score: {best_agg_score:.4f}")

df_agg = df.copy()
df_agg["cluster_agglomerative"] = best_agg_labels
sil_agg = silhouette_score(X_dense, best_agg_labels)
ch_agg = calinski_harabasz_score(X_dense, best_agg_labels)
combined_agg = 0.6 * sil_agg + 0.4 * (ch_agg / 1000)

# Interpretable regimes (no noise for Agglomerative)
df_agg, regime_map_agg = assign_regimes(df_agg, "cluster_agglomerative", allow_noise=False)
print("Agglomerative regime map:", regime_map_agg)

# ============================================================
# 6. SELECT BEST MODEL (KMeans vs Agglomerative)
# ============================================================
print("\n" + "="*60)
print("SELECTING BEST MODEL")
print("="*60)

if combined_km > combined_agg:
    best_model = "KMeans"
    best_score = combined_km
    best_params = best_kmeans_params
    best_df = df_km
    best_regime_map = regime_map_km
    best_sil = sil_km
    best_ch = ch_km
    print(f"✓ Best: KMeans (Score: {combined_km:.4f}, Sil: {sil_km:.4f}, CH: {ch_km:.2f})")
else:
    best_model = "Agglomerative"
    best_score = combined_agg
    best_params = best_agg_params
    best_df = df_agg
    best_regime_map = regime_map_agg
    best_sil = sil_agg
    best_ch = ch_agg
    print(f"✓ Best: Agglomerative (Score: {combined_agg:.4f}, Sil: {sil_agg:.4f}, CH: {ch_agg:.2f})")

# Check cluster distribution for best model
best_cluster_col = "cluster_kmeans" if best_model == "KMeans" else "cluster_agglomerative"
unique_clusters, counts = np.unique(best_df[best_cluster_col], return_counts=True)
print(f"\nCluster distribution:")
for cluster, count in zip(unique_clusters, counts):
    pct = (count / len(best_df)) * 100
    print(f"  Cluster {cluster}: {count} points ({pct:.1f}%)")

# Warn if clusters are too imbalanced
min_count = counts.min()
max_count = counts.max()
balance_ratio = min_count / max_count if max_count > 0 else 0
if balance_ratio < 0.1:
    print(f"\n⚠ WARNING: Clusters are very imbalanced (balance ratio: {balance_ratio:.3f})")
    print("   Consider using different features or dimensionality reduction (PCA)")

# Save only the best model
output_path = "Data/Output/best_clustering_regimes.csv"
best_df.to_csv(output_path)
print(f"\n✓ Saved best model to: {output_path}")

# ============================================================
# 7. METRIC COMPARISON TABLE
# ============================================================
comparison_df = pd.DataFrame(
    eval_rows,
    columns=["Model", "Silhouette", "DB Index", "CH Index", "Inertia", "Parameters"],
)

# Add combined score
comparison_df["Combined_Score"] = (
    0.6 * comparison_df["Silhouette"] + 
    0.4 * (comparison_df["CH Index"] / 1000)
)

# Sort by combined score
comparison_df = comparison_df.sort_values("Combined_Score", ascending=False)

print("\n" + "="*80)
print("CLUSTER EVALUATION SUMMARY (All Models)")
print("="*80)
print(comparison_df.to_string(index=False))

# Save evaluation metrics
metrics_path = "Data/Output/clustering_evaluation_metrics.csv"
comparison_df.to_csv(metrics_path, index=False)
print(f"\n✓ Saved evaluation metrics to: {metrics_path}")

print("\n" + "="*80)
print("BEST MODEL SUMMARY")
print("="*80)
print(f"Model: {best_model}")
print(f"Parameters: {best_params}")
print(f"Silhouette Score: {best_sil:.4f}")
print(f"Calinski-Harabasz Index: {best_ch:.2f}")
print(f"Combined Score: {best_score:.4f}")
print(f"Regime Map: {best_regime_map}")
print("="*80)

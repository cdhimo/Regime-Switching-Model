import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# ============================================================
# 1. LOAD HOURLY FEATURES
# ============================================================
# Adjust path if needed
INPUT_PATH = "Data/regime_features_hourly_categorical.csv"

df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
df = df.dropna()

print("Loaded hourly features:", df.shape)

# ------------------------------------------------------------
# Identify feature groups
# ------------------------------------------------------------
numeric_features = [
    "BTC_ret_1h",
    "SPX_ret_1h",
    "BTC_vol_7d",
    "SPX_vol_7d",
    "BTC_mom_1d",
    "BTC_mom_7d",
    "SPX_mom_1d",
    "SPX_mom_7d",
    "BTC_range_1d",
]

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

TARGET = "BTC_fwd_30d_log_ret"  # label (not used in clustering here)

# Features only (we leave TARGET in df for later use, but not in "data")
data = df[numeric_features + categorical_features].copy()

# ============================================================
# 2. PREPROCESS PIPELINE (ONE-HOT + STANDARDIZE)
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ============================================================
# 3. KMEANS CLUSTERING (4 REGIMES)
# ============================================================
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("cluster", kmeans)
])

print("Fitting KMeans clustering...")
pipeline.fit(data)

cluster_labels = pipeline["cluster"].labels_
df["cluster"] = cluster_labels

print("\nCluster counts:")
print(df["cluster"].value_counts())

# ============================================================
# 4. ANALYZE CLUSTER PROFILES TO NAME REGIMES
# ============================================================
cluster_summary = df.groupby("cluster")[numeric_features].mean()

print("\nCluster Summary (mean numeric features by cluster):")
print(cluster_summary)

summary = cluster_summary.copy()

# ------------------------------------------------------------
# Ensure each regime gets a UNIQUE cluster
# ------------------------------------------------------------

# 1) Order clusters by 7-day BTC momentum (from lowest to highest)
ordered_by_mom = summary["BTC_mom_7d"].sort_values()

# Bear = lowest momentum
bear_cluster = ordered_by_mom.index[0]

# Bull = highest momentum
bull_cluster = ordered_by_mom.index[-1]

# 2) Remaining clusters (candidates for steady / high_volume)
remaining = summary.drop(index=[bear_cluster, bull_cluster])

# High-volume = among remaining, max 1-day BTC range
high_vol_cluster = remaining["BTC_range_1d"].idxmax()

# Steady = the last leftover cluster
steady_cluster_candidates = [
    c for c in summary.index
    if c not in {bear_cluster, bull_cluster, high_vol_cluster}
]
if len(steady_cluster_candidates) != 1:
    raise RuntimeError(
        f"Unexpected number of steady cluster candidates: {steady_cluster_candidates}"
    )
steady_cluster = steady_cluster_candidates[0]

regime_map = {
    bull_cluster: "bull",
    bear_cluster: "bear",
    high_vol_cluster: "high_volume",
    steady_cluster: "steady",
}

df["regime"] = df["cluster"].map(regime_map)

print("\nRegime Mapping (cluster -> regime):")
print(regime_map)

print("\nRegime counts:")
print(df["regime"].value_counts())

# Quick sanity check: all four regimes should appear
expected_regimes = {"bull", "bear", "steady", "high_volume"}
present_regimes = set(df["regime"].unique())
missing = expected_regimes - present_regimes
if missing:
    print("\n[WARN] Missing regimes:", missing)
else:
    print("\nAll four regimes present ✔")

# ============================================================
# 5. SAVE OUTPUT
# ============================================================
OUTPUT_PATH = "Data/regime_features_with_regimes.csv"
df.to_csv(OUTPUT_PATH)

print(f"\nSaved with regimes to: {OUTPUT_PATH}")

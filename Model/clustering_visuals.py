import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates

# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "Data/Output/agglomerative_regimes.csv"

# Where to save all figures and outputs
FIG_DIR = "Figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
df = df.sort_index()

print("Loaded for visuals:", df.shape)
print("Columns:", df.columns.tolist())

if "regime" not in df.columns:
    raise ValueError("Expected a 'regime' column in the CSV.")

regime_col = "regime"

# Try to locate a BTC price column
if "BTC_Close" in df.columns:
    price_col = "BTC_Close"
elif "BTC_close" in df.columns:
    price_col = "BTC_close"
elif "close" in df.columns:
    price_col = "close"
else:
    raise ValueError("Could not find a BTC price column (BTC_Close / BTC_close / close).")

# Try to locate forward 30d BTC log-return column
fwd_col = "BTC_fwd_30d_log_ret" if "BTC_fwd_30d_log_ret" in df.columns else None

# Some volatility / range features (only use those that exist)
vol_candidates = [c for c in ["BTC_vol_7d", "BTC_vol_30d", "BTC_range_1d"] if c in df.columns]
vol_col = vol_candidates[0] if vol_candidates else None

# Numeric feature list for PCA & parallel coordinates
numeric_candidates = [
    c for c in df.columns
    if df[c].dtype != "O" and any(k in c for k in ["BTC_", "SPX_", "US10Y"])
]

numeric_candidates = list(dict.fromkeys(numeric_candidates))  # dedupe

print("Numeric candidates for PCA/parallel:", numeric_candidates)

# ============================================================
# 1. REGIME COUNTS BAR CHART
# ============================================================
plt.figure(figsize=(6, 4))
df[regime_col].value_counts().plot(kind="bar")
plt.title("Regime Counts")
plt.ylabel("Number of observations")
plt.xlabel("Regime")
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "regime_counts.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print("Saved:", fig_path)

# ============================================================
# 2. BTC PRICE OVER TIME BY REGIME
# ============================================================
plt.figure(figsize=(10, 4))
for regime, group in df.groupby(regime_col):
    plt.plot(group.index, group[price_col], ".", label=regime, alpha=0.6)

plt.title("BTC Price over Time by Regime")
plt.xlabel("Time")
plt.ylabel("BTC Price")
plt.legend()
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "btc_price_by_regime.png")
plt.savefig(fig_path, dpi=300)
plt.close()
print("Saved:", fig_path)

# ============================================================
# 3. FORWARD 30D BTC RETURNS BY REGIME (KDE + BOXPLOT)
# ============================================================
if fwd_col is not None:
    # KDE
    plt.figure(figsize=(8, 4))
    for regime in df[regime_col].unique():
        subset = df[df[regime_col] == regime][fwd_col].dropna()
        if subset.empty:
            continue
        subset.plot(kind="kde", label=regime)
    plt.title("Distribution of 30-day Forward BTC Log Returns by Regime")
    plt.xlabel(fwd_col)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fwd_returns_kde_by_regime.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)

    # Boxplot
    plt.figure(figsize=(8, 4))
    data_box = [df[df[regime_col] == r][fwd_col].dropna() for r in df[regime_col].unique()]
    plt.boxplot(data_box, labels=df[regime_col].unique())
    plt.title("30-day Forward BTC Log Returns by Regime (Boxplot)")
    plt.ylabel(fwd_col)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fwd_returns_boxplot_by_regime.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)
else:
    print("[INFO] Forward 30d return column not found; skipping forward-return plots.")

# ============================================================
# 4. VOLATILITY BY REGIME (BOXPLOT)
# ============================================================
if vol_col is not None:
    plt.figure(figsize=(8, 4))
    data_box = [df[df[regime_col] == r][vol_col].dropna() for r in df[regime_col].unique()]
    plt.boxplot(data_box, labels=df[regime_col].unique())
    plt.title(f"{vol_col} by Regime (Boxplot)")
    plt.ylabel(vol_col)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "volatility_boxplot_by_regime.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)
else:
    print("[INFO] No volatility column found (BTC_vol_7d / BTC_vol_30d / BTC_range_1d); skipping volatility boxplot.")

# ============================================================
# 5. PARALLEL COORDINATES PLOT
# ============================================================
if len(numeric_candidates) >= 2:
    par_df = df[numeric_candidates + [regime_col]].dropna()

    # Optionally subsample for clarity if too many points
    max_points = 1000
    if par_df.shape[0] > max_points:
        par_df = par_df.sample(max_points, random_state=42).sort_index()

    # Normalize numeric columns for clearer parallel coordinates
    scaler = StandardScaler()
    par_df_scaled = par_df.copy()
    par_df_scaled[numeric_candidates] = scaler.fit_transform(par_df[numeric_candidates])

    plt.figure(figsize=(10, 6))
    parallel_coordinates(par_df_scaled, class_column=regime_col)
    plt.title("Parallel Coordinates of Features by Regime")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "parallel_coordinates_by_regime.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)
else:
    print("[INFO] Not enough numeric features for parallel coordinates.")

# ============================================================
# 6. PCA 2D SCATTER COLORED BY REGIME
# ============================================================
if len(numeric_candidates) >= 2:
    num_df = df[numeric_candidates].dropna()
    regimes_for_pca = df.loc[num_df.index, regime_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_df.values)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, index=num_df.index, columns=["PC1", "PC2"])
    pca_df[regime_col] = regimes_for_pca

    plt.figure(figsize=(7, 5))
    for regime, group in pca_df.groupby(regime_col):
        plt.scatter(group["PC1"], group["PC2"], label=regime, alpha=0.5, s=10)
    plt.title("PCA of Features Colored by Regime")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "pca_scatter_by_regime.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
else:
    print("[INFO] Not enough numeric features for PCA.")

# ============================================================
# 7. MEAN FORWARD RETURN BY REGIME (BAR CHART)
# ============================================================
if fwd_col is not None:
    mean_returns = df.groupby(regime_col)[fwd_col].mean()
    std_returns = df.groupby(regime_col)[fwd_col].std()

    x = np.arange(len(mean_returns))
    plt.figure(figsize=(7, 4))
    plt.bar(x, mean_returns.values)
    plt.xticks(x, mean_returns.index)
    plt.title("Mean 30-day Forward BTC Log Return by Regime")
    plt.ylabel(fwd_col)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "mean_fwd_return_by_regime.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)
else:
    print("[INFO] No forward return column; skipping mean forward-return bar chart.")



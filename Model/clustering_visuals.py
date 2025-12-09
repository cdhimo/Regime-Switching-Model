import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8')

df = pd.read_csv("Data/regime_features_with_regimes.csv", index_col=0, parse_dates=True)

# Choose relevant numeric features for plots
num_features = [
    "BTC_ret_1h", "BTC_mom_1d", "BTC_mom_7d",
    "BTC_vol_7d", "BTC_range_1d",
    "SPX_ret_1h", "SPX_vol_7d"
]

# ------------------------------------------------------
# 1. TIME SERIES REGIME OVERLAY PLOT
# ------------------------------------------------------
def plot_regime_timeseries():
    regime_colors = {
        "bull": "green",
        "bear": "red",
        "steady": "blue",
        "high_volume": "orange"
    }

    plt.figure(figsize=(16,6))
    plt.plot(df.index, df["BTC_Close"], color="black", alpha=0.7, label="BTC Price")

    # Shade regions by regime
    for regime, color in regime_colors.items():
        mask = df["regime"] == regime
        plt.scatter(df.index[mask], df["BTC_Close"][mask], s=10, color=color, label=regime)

    plt.legend()
    plt.title("BTC Regime Overlay")
    plt.xlabel("Date")
    plt.ylabel("BTC Price")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 2. PCA 2D SCATTER PLOT
# ------------------------------------------------------
def plot_pca_clusters():
    from sklearn.preprocessing import StandardScaler

    df_clean = df.dropna(subset=num_features)
    X = df_clean[num_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(X_scaled)

    df_clean["PC1"] = pc[:,0]
    df_clean["PC2"] = pc[:,1]

    plt.figure(figsize=(10,7))
    sns.scatterplot(
        data=df_clean,
        x="PC1",
        y="PC2",
        hue="regime",
        palette={"bull": "green","bear": "red","steady":"blue","high_volume":"orange"},
        alpha=0.6
    )

    plt.title("PCA Projection of Regime Clusters")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 3. CLUSTER HEATMAP OF FEATURE MEANS
# ------------------------------------------------------
def plot_feature_heatmap():
    cluster_means = df.groupby("regime")[num_features].mean()

    plt.figure(figsize=(10,6))
    sns.heatmap(cluster_means, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Average Feature Values per Regime")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 4. RADAR CHARTS FOR EACH REGIME
# ------------------------------------------------------
def plot_radar_charts():
    cluster_means = df.groupby("regime")[num_features].mean()

    categories = num_features
    N = len(categories)

    for regime in cluster_means.index:
        values = cluster_means.loc[regime].values
        values = np.concatenate([values, [values[0]]])  # wrap around

        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(f"Radar Chart – {regime.capitalize()} Regime")
        plt.tight_layout()
        plt.show()

# ------------------------------------------------------
# 5. REGIME FREQUENCY BAR CHART
# ------------------------------------------------------
def plot_regime_hist():
    plt.figure(figsize=(8,4))
    df["regime"].value_counts().plot(kind="bar", color=["green","red","blue","orange"])
    plt.title("Regime Frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# 6. TRANSITION MATRIX (Markov-style)
# ------------------------------------------------------
def plot_transition_matrix():
    regimes = df["regime"].unique()
    idx = {r:i for i,r in enumerate(regimes)}

    transition = np.zeros((len(regimes), len(regimes)))

    prev = df["regime"].iloc[0]
    for r in df["regime"].iloc[1:]:
        transition[idx[prev], idx[r]] += 1
        prev = r

    # Normalize rows
    transition = transition / transition.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        transition,
        annot=True,
        xticklabels=regimes,
        yticklabels=regimes,
        cmap="Blues",
        fmt=".2f"
    )
    plt.title("Regime Transition Matrix")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------
# RUN ALL VISUALS
# ------------------------------------------------------
if __name__ == "__main__":
    plot_regime_timeseries()
    plot_pca_clusters()
    plot_feature_heatmap()
    plot_radar_charts()
    plot_regime_hist()
    plot_transition_matrix()

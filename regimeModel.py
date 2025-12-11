import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(
    "Data/Output/kmeans_regimes.csv"
)

# ============================================================
# create lags
# ============================================================
LAGS = 24 * 3         # 72 hours of lagged 1h returns

# Lagged BTC returns
for i in range(1, LAGS + 1):
    df[f"BTC_ret_1h_lag_{i}"] = df["BTC_ret_1h"].shift(i)

# ============================================================
# Feature selection
# ============================================================
numeric_base_features = [
    # Momentum
    "BTC_mom_1d", "BTC_mom_7d",
    "SPX_mom_1d", "SPX_mom_7d",

    # Volatility / range
    "BTC_vol_7d", "SPX_vol_7d",
    "BTC_range_1d",

    # Rates
    "US10Y_chg_1d", "US10Y_chg_7d", "US10Y_vol_7d",

    # Technical indicators
    "BTC_RSI_14", "BTC_MACD_line", "BTC_MACD_signal", "BTC_ADX_14",
]

# Short-term lagged returns (6 lags)
lag_features = [f'BTC_ret_1h_lag_{i}' for i in range(1, LAGS + 1)]

# Categorical (keep minimal)
time_features = ["day_of_week_cat", "weekend_cat"]

# Optional regimes (keep only vol regime)
regime_features = ["BTC_vol_regime_7d", "regime"]

# Combine
feature_cols = numeric_base_features + lag_features + time_features + regime_features

# Drop rows that don't have full lags or future target
df = df.dropna().reset_index(drop=True)

# ============================================================
# Prepare X and y
# ============================================================

# One-hot encode categorical features (regimes + time)
X = pd.get_dummies(
    df[feature_cols],
    columns=regime_features + time_features,
    drop_first=True,
)

# Target: 7-day forward log return
y = df["BTC_fwd_7d_log_ret"].values

# Numeric columns to scale
numerical_features = numeric_base_features + lag_features

# Preprocessor: scale numeric features, passthrough dummies
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features)
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,  # simpler column names
)

# ============================================================
# Model pipeline
# ============================================================
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=8,          # can tune
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=25,
                min_samples_split=50
            ),
        ),
    ]
)

# ============================================================
# KMeans-based "cluster CV" folds
# ============================================================
n_folds = 5
X_scaled_for_kmeans = X.copy()

scaler_kmeans = StandardScaler()
X_scaled_for_kmeans[numerical_features] = scaler_kmeans.fit_transform(
    X[numerical_features]
)

kmeans = KMeans(n_clusters=n_folds, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled_for_kmeans)  # each row assigned a cluster (fold)

# Prepare arrays to store predictions
y_pred_all = np.zeros_like(y, dtype=float)
y_mask = np.zeros_like(y, dtype=bool)

# ============================================================
# Cluster-based cross-validation
# ============================================================
for fold in range(n_folds):
    test_idx = np.where(clusters == fold)[0]
    train_idx = np.where(clusters != fold)[0]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y[train_idx]

    print(
        f"--- Training Fold {fold + 1}/{n_folds} "
        f"(Train size: {len(X_train)}, Test size: {len(X_test)}) ---"
    )

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Predict on this fold
    y_pred = model_pipeline.predict(X_test)

    # Store predictions
    y_pred_all[test_idx] = y_pred
    y_mask[test_idx] = True

# ============================================================
# Evaluation
# ============================================================
y_true_final = y[y_mask]
y_pred_final = y_pred_all[y_mask]

mae = mean_absolute_error(y_true_final, y_pred_final)
mse = mean_squared_error(y_true_final, y_pred_final)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_final, y_pred_final)

n = y_true_final.shape[0]  # number of samples
p = X.shape[1]             # number of features (after encoding)

if (n - p - 1) > 0:
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
else:
    r2_adj = np.nan  # Cannot calculate Adjusted R-squared
def horizon_sharpe(pred_returns):
    pred_returns = np.array(pred_returns)
    if pred_returns.std() > 0:
        return pred_returns.mean() / pred_returns.std()
    else:
        return np.nan


print("=== Random Forest KMeans-Fold Evaluation (1h ahead) ===")
print(f"MAE   : {mae:.6f}")
print(f"MSE   : {mse:.6f}")
print(f"RMSE  : {rmse:.6f}")
print(f"R^2   : {r2:.4f}")
print(f"Adj R^2: {r2_adj:.4f}")
print(f"Crypto 7-Day Sharpe: {horizon_sharpe(y_pred_final):.4f}")

# ============================================================
# Plot predictions
# ============================================================
plt.figure(figsize=(14, 6))
plt.plot(np.where(y_mask, y, np.nan), label="Actual 7-Day Log Return", zorder=1)
plt.plot(np.where(y_mask, y_pred_all, np.nan), label="Predicted 7-Day Log Return", zorder=2)
plt.title("KMeans-Fold CV: 7-Day Ahead BTC Log Return Forecast (Full Dataset)")
plt.xlabel("Time Index")
plt.ylabel("7-Day Log Return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

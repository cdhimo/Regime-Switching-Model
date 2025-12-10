import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("C:/Users/tjgre/OneDrive/code/Regime-Switching-Model/Data/regime_features_with_regimes.csv")

# define horizon and target
HORIZON = 720
LAGS = 24

# lagged BTC returns
for i in range(1, LAGS + 1):
    df[f'BTC_ret_1h_lag_{i}'] = df['BTC_ret_1h'].shift(i)

# feature selection
regime_features = ['BTC_vol_regime_7d','BTC_mom_regime_7d','BTC_range_regime_1d','regime']
time_features = ['hour_of_day_cat','day_of_week_cat','weekend_cat']

lag_features = [f'BTC_ret_1h_lag_{i}' for i in range(1, LAGS + 1)]
feature_cols = lag_features + regime_features + time_features

df = df.dropna().reset_index(drop=True)  # drop rows that don't have full lag or future target

# One-hot encode categorical features (regimes + time)
X = pd.get_dummies(df[feature_cols], columns=regime_features + time_features, drop_first=True)
y = df['BTC_fwd_30d_log_ret'].values
numerical_features = lag_features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False # Ensures output column names are simple
)

# Define the complete model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=500,
        max_depth=20, # or tune this, e.g., 10 or 20
        random_state=42,
        n_jobs=-1, min_samples_leaf = 15
    ))
])

# Apply KMeans to generate clusters
n_folds = 5
X_scaled_for_kmeans = X.copy()
scaler_kmeans = StandardScaler()
X_scaled_for_kmeans[numerical_features] = scaler_kmeans.fit_transform(X[numerical_features])

kmeans = KMeans(n_clusters=n_folds, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled_for_kmeans)  # each row assigned a cluster (fold)

# Prepare arrays to store predictions
y_pred_all = np.zeros_like(y)
y_mask = np.zeros_like(y, dtype=bool)

# Cluster-based cross-validation
for fold in range(n_folds):
    test_idx = np.where(clusters == fold)[0]
    train_idx = np.where(clusters != fold)[0]

    # Select data for the current fold
    # NOTE: We use the ONE-HOT encoded but UNSCALED X here,
    # because the scaling is handled by the 'preprocessor' inside the pipeline.
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y[train_idx]

    print(f"--- Training Fold {fold + 1}/{n_folds} (Train size: {len(X_train)}, Test size: {len(X_test)}) ---")

    # Train model using the pipeline (scaling happens automatically inside fit/predict)
    model_pipeline.fit(X_train, y_train)

    # Predict
    y_pred = model_pipeline.predict(X_test)

    # Store predictions
    y_pred_all[test_idx] = y_pred
    y_mask[test_idx] = True

y_true_final = y[y_mask]
y_pred_final = y_pred_all[y_mask]
# ----------------------------
# Evaluation metrics
mae = mean_absolute_error(y_true_final, y_pred_final)
mse = mean_squared_error(y_true_final, y_pred_final)
rmse = np.sqrt(mse)
r2 = r2_score(y_true_final, y_pred_final)

n = y_true_final[y_mask].shape[0]  # number of samples
p = X.shape[1]                   # number of features (after encoding)

if (n - p - 1) > 0:
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
else:
    r2_adj = np.nan # Cannot calculate Adjusted R-squared

print("=== Random Forest KMeans-Fold Evaluation ===")
print(f"MAE  : {mae:.6f}")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R^2  : {r2:.4f}")
print(f"Adj R^2: {r2_adj:.4f}")

# ----------------------------
# Plot predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
# Only plot the predicted indices
plt.plot(np.where(y_mask, y, np.nan), label="Actual 30-day Log Return", zorder=1)
plt.plot(np.where(y_mask, y_pred_all, np.nan), label="Predicted 30-day Log Return", zorder=2)
plt.title("KMeans-Fold CV: 30-Day Ahead BTC Log Return Forecast (Full Dataset)")
plt.xlabel("Time Index")
plt.ylabel("Log Return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

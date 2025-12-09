import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD HOURLY STACKED DATA
# ============================================================
df = pd.read_csv("regime_data.csv", parse_dates=["datetime"])
df = df.sort_values("datetime")

prices = df.pivot(index="datetime", columns="symbol", values="close")
prices = prices[["BTC", "SPX"]].dropna()
prices = prices.rename(columns={"BTC": "BTC_Close", "SPX": "SPX_Close"})

btc = prices["BTC_Close"]
spx = prices["SPX_Close"]

# Hourly returns
btc_ret_1h = np.log(btc / btc.shift(1))
spx_ret_1h = np.log(spx / spx.shift(1))

# ============================================================
# 2. DEFINING WINDOWS
# ============================================================
H_1D = 24
H_7D = 24 * 7
H_30D = 24 * 30  # Label horizon

# ============================================================
# 3. NUMERIC FEATURES
# ============================================================
btc_fwd_30d = np.log(btc.shift(-H_30D) / btc)

btc_vol_7d = btc_ret_1h.rolling(H_7D).std()
spx_vol_7d = spx_ret_1h.rolling(H_7D).std()

btc_mom_1d = btc / btc.shift(H_1D) - 1
btc_mom_7d = btc / btc.shift(H_7D) - 1
spx_mom_1d = spx / spx.shift(H_1D) - 1
spx_mom_7d = spx / spx.shift(H_7D) - 1

btc_high_1d = btc.rolling(H_1D).max()
btc_low_1d = btc.rolling(H_1D).min()
btc_range_1d = btc_high_1d - btc_low_1d

# ============================================================
# 4. CATEGORICAL TIME FEATURES (REAL STRINGS)
# ============================================================
idx = prices.index

hour_of_day_cat = idx.hour.map(lambda h: f"hour_{h}")
dow_map = {0:"mon",1:"tue",2:"wed",3:"thu",4:"fri",5:"sat",6:"sun"}
day_of_week_cat = idx.dayofweek.map(dow_map)
weekend_cat = idx.dayofweek.map(lambda d: "weekend" if d>=5 else "weekday")

# ============================================================
# 5. CATEGORICAL BUCKETS AS STRINGS
# ============================================================
def bucket_to_label(series, low_label, mid_label, high_label):
    q1 = series.quantile(0.33)
    q2 = series.quantile(0.66)
    def f(x):
        if pd.isna(x): 
            return np.nan
        if x < q1: return low_label
        if x < q2: return mid_label
        return high_label
    return series.apply(f)

BTC_vol_regime_7d = bucket_to_label(btc_vol_7d, "low_vol", "mid_vol", "high_vol")
SPX_vol_regime_7d = bucket_to_label(spx_vol_7d, "low_vol", "mid_vol", "high_vol")

BTC_mom_regime_7d = bucket_to_label(btc_mom_7d, "neg_mom", "neutral_mom", "pos_mom")
SPX_mom_regime_7d = bucket_to_label(spx_mom_7d, "neg_mom", "neutral_mom", "pos_mom")

BTC_range_regime_1d = bucket_to_label(btc_range_1d, "small_range", "mid_range", "large_range")

# ============================================================
# 6. BUILD FINAL FEATURE TABLE
# ============================================================
features = pd.DataFrame({
    # Numeric
    "BTC_Close": btc,
    "SPX_Close": spx,
    "BTC_ret_1h": btc_ret_1h,
    "SPX_ret_1h": spx_ret_1h,
    "BTC_vol_7d": btc_vol_7d,
    "SPX_vol_7d": spx_vol_7d,
    "BTC_mom_1d": btc_mom_1d,
    "BTC_mom_7d": btc_mom_7d,
    "SPX_mom_1d": spx_mom_1d,
    "SPX_mom_7d": spx_mom_7d,
    "BTC_range_1d": btc_range_1d,

    # Time categorical
    "hour_of_day_cat": hour_of_day_cat,
    "day_of_week_cat": day_of_week_cat,
    "weekend_cat": weekend_cat,

    # Regime categorical
    "BTC_vol_regime_7d": BTC_vol_regime_7d,
    "SPX_vol_regime_7d": SPX_vol_regime_7d,
    "BTC_mom_regime_7d": BTC_mom_regime_7d,
    "SPX_mom_regime_7d": SPX_mom_regime_7d,
    "BTC_range_regime_1d": BTC_range_regime_1d,

    # Label
    "BTC_fwd_30d_log_ret": btc_fwd_30d,
})

# Drop rows missing key features
features_clean = features.dropna(subset=[
    "BTC_vol_7d", "BTC_mom_7d", "BTC_range_1d",
    "SPX_vol_7d", "SPX_mom_7d",
    "BTC_fwd_30d_log_ret"
])

print("Final rows:", len(features_clean))

# Save
features_clean.to_csv("regime_features_hourly_categorical.csv")
print("Saved: regime_features_hourly_categorical.csv")

import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD HOURLY STACKED DATA
# ============================================================
# Expect regime_data.csv from your Pyth fetcher:
# columns: datetime, open, high, low, close, log_ret_1h, vol_realized, symbol
df = pd.read_csv("Data/regime_data.csv", parse_dates=["datetime"])
df = df.sort_values("datetime")

# Pivot close, high, low so we can compute indicators per asset
prices_close = df.pivot(index="datetime", columns="symbol", values="close")
prices_high  = df.pivot(index="datetime", columns="symbol", values="high")
prices_low   = df.pivot(index="datetime", columns="symbol", values="low")

# ------------------------------------------------------------
# Keep BTC, SPX, and US10Y (rate) and align indices
# Adjust "US10Y" to your rate symbol if needed.
# ------------------------------------------------------------
prices_close = prices_close[["BTC", "SPX", "US10Y"]].dropna()
prices_high  = prices_high.reindex(prices_close.index)[["BTC", "SPX", "US10Y"]]
prices_low   = prices_low.reindex(prices_close.index)[["BTC", "SPX", "US10Y"]]

prices_close = prices_close.rename(columns={
    "BTC": "BTC_Close",
    "SPX": "SPX_Close",
    "US10Y": "US10Y"
})

btc   = prices_close["BTC_Close"]
spx   = prices_close["SPX_Close"]
us10y = prices_close["US10Y"]

btc_high = prices_high["BTC"]
btc_low  = prices_low["BTC"]

# Hourly log returns (for BTC and SPX)
btc_ret_1h = np.log(btc / btc.shift(1))
spx_ret_1h = np.log(spx / spx.shift(1))

# ============================================================
# 2. DEFINING WINDOWS
# ============================================================
H_1D  = 24
H_7D  = 24 * 7
H_30D = 24 * 30  # ~30 days of hours, used for features + labels

# ============================================================
# 3. BTC BACKWARD-LOOKING LOG RETURNS (FEATURES)
# ============================================================
# 1h (already above, but keep explicit)
BTC_log_ret_1h = btc_ret_1h

# 1-day (24h) backward log return
BTC_log_ret_1d = np.log(btc / btc.shift(H_1D))

# 7-day (168h) backward log return
BTC_log_ret_7d = np.log(btc / btc.shift(H_7D))

# 30-day (720h) backward log return
BTC_log_ret_30d = np.log(btc / btc.shift(H_30D))

# ============================================================
# 4. BTC FORWARD-LOOKING LOG RETURNS (LABELS / TARGETS)
# ============================================================
# 1-hour forward log return
BTC_fwd_1h_log_ret = np.log(btc.shift(-1) / btc)

# 1-day (24h) forward log return
BTC_fwd_1d_log_ret = np.log(btc.shift(-H_1D) / btc)

# 7-day (168h) forward log return
BTC_fwd_7d_log_ret = np.log(btc.shift(-H_7D) / btc)

# 30-day (720h) forward log return
BTC_fwd_30d_log_ret = np.log(btc.shift(-H_30D) / btc)

# ============================================================
# 5. BASE NUMERIC FEATURES (VOL, MOMENTUM, RANGE)
# ============================================================
btc_vol_7d = btc_ret_1h.rolling(H_7D).std()
spx_vol_7d = spx_ret_1h.rolling(H_7D).std()

btc_mom_1d = btc / btc.shift(H_1D) - 1
btc_mom_7d = btc / btc.shift(H_7D) - 1
spx_mom_1d = spx / spx.shift(H_1D) - 1
spx_mom_7d = spx / spx.shift(H_7D) - 1

btc_high_1d  = btc.rolling(H_1D).max()
btc_low_1d   = btc.rolling(H_1D).min()
btc_range_1d = btc_high_1d - btc_low_1d

# ============================================================
# 6. TECHNICAL INDICATORS: RSI, MACD, BOLLINGER, ADX (BTC)
# ============================================================

# ---------- RSI (14-period) ----------
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

BTC_RSI_14 = compute_rsi(btc, window=14)

# ---------- MACD (12-26 EMA, signal 9 EMA) ----------
fast_span   = 12
slow_span   = 26
signal_span = 9

ema_fast        = btc.ewm(span=fast_span, adjust=False).mean()
ema_slow        = btc.ewm(span=slow_span, adjust=False).mean()
BTC_MACD_line   = ema_fast - ema_slow
BTC_MACD_signal = BTC_MACD_line.ewm(span=signal_span, adjust=False).mean()
BTC_MACD_hist   = BTC_MACD_line - BTC_MACD_signal

# ---------- Bollinger Bands (20-period, 2 std) ----------
BB_window = 20
BB_mid    = btc.rolling(BB_window).mean()
BB_std    = btc.rolling(BB_window).std()

BTC_BB_mid_20   = BB_mid
BTC_BB_upper_20 = BB_mid + 2 * BB_std
BTC_BB_lower_20 = BB_mid - 2 * BB_std
BTC_BB_width_20 = BTC_BB_upper_20 - BTC_BB_lower_20

# ---------- ADX (14-period) ----------
def compute_adx(high, low, close, window=14):
    """
    Basic ADX implementation (Welles Wilder style using rolling sums).
    high, low, close: pd.Series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    TR  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm  = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    TR_n       = TR.rolling(window).sum()
    plus_dm_n  = plus_dm.rolling(window).sum()
    minus_dm_n = minus_dm.rolling(window).sum()

    plus_di  = 100 * (plus_dm_n / TR_n)
    minus_di = 100 * (minus_dm_n / TR_n)

    dx  = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    adx = dx.rolling(window).mean()
    return adx

BTC_ADX_14 = compute_adx(btc_high, btc_low, btc, window=14)

# ============================================================
# 7. RATE FEATURES (US10Y) - LEVEL, CHANGE, VOL
# ============================================================
US10Y_chg_1h = us10y - us10y.shift(1)
US10Y_chg_1d = us10y - us10y.shift(H_1D)
US10Y_chg_7d = us10y - us10y.shift(H_7D)

US10Y_vol_7d        = US10Y_chg_1h.rolling(H_7D).std()
US10Y_level_7d_mean = us10y.rolling(H_7D).mean()

# ============================================================
# 8. CATEGORICAL TIME FEATURES (STRING LABELS)
# ============================================================
idx = prices_close.index

hour_of_day_cat = idx.hour.map(lambda h: f"hour_{h}")
dow_map = {0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"}
day_of_week_cat = idx.dayofweek.map(dow_map)
weekend_cat = idx.dayofweek.map(lambda d: "weekend" if d >= 5 else "weekday")

# ============================================================
# 9. CATEGORICAL BUCKETS (STRING REGIMES)
# ============================================================
def bucket_to_label(series, low_label, mid_label, high_label):
    q1 = series.quantile(0.33)
    q2 = series.quantile(0.66)
    def f(x):
        if pd.isna(x):
            return np.nan
        if x < q1:
            return low_label
        if x < q2:
            return mid_label
        return high_label
    return series.apply(f)

BTC_vol_regime_7d = bucket_to_label(btc_vol_7d, "low_vol", "mid_vol", "high_vol")
SPX_vol_regime_7d = bucket_to_label(spx_vol_7d, "low_vol", "mid_vol", "high_vol")

BTC_mom_regime_7d = bucket_to_label(btc_mom_7d, "neg_mom", "neutral_mom", "pos_mom")
SPX_mom_regime_7d = bucket_to_label(spx_mom_7d, "neg_mom", "neutral_mom", "pos_mom")

BTC_range_regime_1d = bucket_to_label(btc_range_1d, "small_range", "mid_range", "large_range")

# ============================================================
# 10. BUILD FINAL FEATURE TABLE
# ============================================================
features = pd.DataFrame({
    # Core prices
    "BTC_Close": btc,
    "SPX_Close": spx,
    "US10Y": us10y,

    # Hourly log returns
    "BTC_ret_1h": btc_ret_1h,
    "SPX_ret_1h": spx_ret_1h,

    # BTC multi-horizon log returns (backward-looking)
    "BTC_log_ret_1h": BTC_log_ret_1h,
    "BTC_log_ret_1d": BTC_log_ret_1d,
    "BTC_log_ret_7d": BTC_log_ret_7d,
    "BTC_log_ret_30d": BTC_log_ret_30d,

    # Volatility & momentum
    "BTC_vol_7d": btc_vol_7d,
    "SPX_vol_7d": spx_vol_7d,
    "BTC_mom_1d": btc_mom_1d,
    "BTC_mom_7d": btc_mom_7d,
    "SPX_mom_1d": spx_mom_1d,
    "SPX_mom_7d": spx_mom_7d,
    "BTC_range_1d": btc_range_1d,

    # Rate features (US10Y)
    "US10Y_chg_1h": US10Y_chg_1h,
    "US10Y_chg_1d": US10Y_chg_1d,
    "US10Y_chg_7d": US10Y_chg_7d,
    "US10Y_vol_7d": US10Y_vol_7d,
    "US10Y_level_7d_mean": US10Y_level_7d_mean,

    # Technical indicators (BTC)
    "BTC_RSI_14": BTC_RSI_14,
    "BTC_MACD_line": BTC_MACD_line,
    "BTC_MACD_signal": BTC_MACD_signal,
    "BTC_MACD_hist": BTC_MACD_hist,
    "BTC_BB_mid_20": BTC_BB_mid_20,
    "BTC_BB_upper_20": BTC_BB_upper_20,
    "BTC_BB_lower_20": BTC_BB_lower_20,
    "BTC_BB_width_20": BTC_BB_width_20,
    "BTC_ADX_14": BTC_ADX_14,

    # Time categorical (as strings)
    "hour_of_day_cat": hour_of_day_cat,
    "day_of_week_cat": day_of_week_cat,
    "weekend_cat": weekend_cat,

    # Regime-style categorical buckets
    "BTC_vol_regime_7d": BTC_vol_regime_7d,
    "SPX_vol_regime_7d": SPX_vol_regime_7d,
    "BTC_mom_regime_7d": BTC_mom_regime_7d,
    "SPX_mom_regime_7d": SPX_mom_regime_7d,
    "BTC_range_regime_1d": BTC_range_regime_1d,

    # Forward returns (labels / extra targets)
    "BTC_fwd_1h_log_ret": BTC_fwd_1h_log_ret,
    "BTC_fwd_1d_log_ret": BTC_fwd_1d_log_ret,
    "BTC_fwd_7d_log_ret": BTC_fwd_7d_log_ret,
    "BTC_fwd_30d_log_ret": BTC_fwd_30d_log_ret,
})

# ============================================================
# 11. CLEAN & SAVE
# ============================================================
# Require core features + key label + some rate info
features_clean = features.dropna(subset=[
    "BTC_vol_7d",
    "BTC_mom_7d",
    "BTC_range_1d",
    "SPX_vol_7d",
    "SPX_mom_7d",
    "US10Y_chg_7d",
    "US10Y_vol_7d",
    "BTC_fwd_30d_log_ret",  # main long-horizon label
])

print("Final rows:", len(features_clean))

OUTPUT_PATH = "Data/regime_features_hourly_categorical.csv"
features_clean.to_csv(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")

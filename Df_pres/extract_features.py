# -----------------------------------------
# Extract 7-day features and RSI for all assets
# -----------------------------------------
import pandas as pd

# Load the full features dataset
df = pd.read_csv("Data/regime_features_hourly_categorical.csv", parse_dates=["datetime"])

# Select 7-day features and RSI for all assets (BTC, SPX, US10Y)
selected_features = ["datetime"]
rename_map = {}

# 7-day log returns
if "BTC_log_ret_7d" in df.columns:
    selected_features.append("BTC_log_ret_7d")

# 7-day volatility (numeric) for all assets
# Check for numeric version first (with _num suffix), then fallback to base name
for asset in ["BTC", "SPX", "US10Y"]:
    vol_num_col = f"{asset}_vol_7d_num"
    vol_base_col = f"{asset}_vol_7d"
    
    # Check if numeric version exists (new CSV format)
    if vol_num_col in df.columns:
        selected_features.append(vol_num_col)
        rename_map[vol_num_col] = f"{asset}_vol_7d_num"
    # Otherwise check if base name exists and is numeric (old CSV format)
    elif vol_base_col in df.columns:
        # Check if it's numeric (not categorical) by sampling a value
        sample_val = df[vol_base_col].dropna().iloc[0] if len(df[vol_base_col].dropna()) > 0 else None
        if sample_val is not None and isinstance(sample_val, (int, float)) and not isinstance(sample_val, str):
            selected_features.append(vol_base_col)
            rename_map[vol_base_col] = f"{asset}_vol_7d_num"

# 7-day momentum (numeric) for BTC and SPX
for asset in ["BTC", "SPX"]:
    mom_num_col = f"{asset}_mom_7d_num"
    mom_base_col = f"{asset}_mom_7d"
    
    if mom_num_col in df.columns:
        selected_features.append(mom_num_col)
        rename_map[mom_num_col] = f"{asset}_mom_7d_num"
    elif mom_base_col in df.columns:
        sample_val = df[mom_base_col].dropna().iloc[0] if len(df[mom_base_col].dropna()) > 0 else None
        if sample_val is not None and isinstance(sample_val, (int, float)) and not isinstance(sample_val, str):
            selected_features.append(mom_base_col)
            rename_map[mom_base_col] = f"{asset}_mom_7d_num"

# 7-day categorical features (vol and mom) - these are string values
for asset in ["BTC", "SPX"]:
    # Volatility categorical - check for old regime name first
    for old_name in [f"{asset}_vol_regime_7d", f"{asset}_vol_7d"]:
        if old_name in df.columns:
            # Check if it's categorical (string)
            sample_val = df[old_name].dropna().iloc[0] if len(df[old_name].dropna()) > 0 else None
            if sample_val is not None and isinstance(sample_val, str):
                if old_name not in selected_features:
                    selected_features.append(old_name)
                    rename_map[old_name] = f"{asset}_vol_7d"
                break
    
    # Momentum categorical
    for old_name in [f"{asset}_mom_regime_7d", f"{asset}_mom_7d"]:
        if old_name in df.columns:
            sample_val = df[old_name].dropna().iloc[0] if len(df[old_name].dropna()) > 0 else None
            if sample_val is not None and isinstance(sample_val, str):
                if old_name not in selected_features:
                    selected_features.append(old_name)
                    rename_map[old_name] = f"{asset}_mom_7d"
                break

# US10Y 7-day features
for col in ["US10Y_chg_7d", "US10Y_level_7d_mean"]:
    if col in df.columns:
        selected_features.append(col)

# RSI for all assets
for asset in ["BTC", "SPX", "US10Y"]:
    rsi_col = f"{asset}_RSI_14"
    if rsi_col in df.columns:
        selected_features.append(rsi_col)

# Extract features
features_subset = df[selected_features].copy()

# Apply renaming
if rename_map:
    features_subset = features_subset.rename(columns=rename_map)

# Remove rows with any missing values
features_subset = features_subset.dropna()

# Save to new CSV
output_path = "Df_pres/selected_features.csv"
features_subset.to_csv(output_path, index=False)

print(f"✓ Extracted {len(features_subset)} rows with {len(features_subset.columns)} features")
print(f"✓ Saved to: {output_path}")
print(f"\nColumns: {list(features_subset.columns)}")
print(f"\nFirst few rows:")
print(features_subset.head().to_string(index=False))

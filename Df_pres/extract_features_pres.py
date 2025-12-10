# -----------------------------------------
# Extract 12-hour features and RSI for presentation slides
# (Predicting 7-day forward BTC log returns using 12h lookback features)
# -----------------------------------------
import pandas as pd

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Load the full features dataset
df = pd.read_csv("Data/regime_features_hourly_categorical.csv", parse_dates=["datetime"])

# Select 12-hour features and RSI for all assets (BTC, SPX, US10Y)
selected_features = ["datetime"]
rename_map = {}

# 12-hour log returns (BTC only - most important)
for col_name in ["BTC_log_ret_12h"]:
    if col_name in df.columns:
        selected_features.append(col_name)

# 12-hour volatility (categorical) - cleaner for presentation
for asset in ["BTC", "SPX", "US10Y"]:
    for old_name in [f"{asset}_vol_12h"]:
        if old_name in df.columns:
            sample_val = df[old_name].dropna().iloc[0] if len(df[old_name].dropna()) > 0 else None
            if sample_val is not None and isinstance(sample_val, str):
                if old_name not in selected_features:
                    selected_features.append(old_name)
                    rename_map[old_name] = f"{asset}_vol_12h"
                break

# 12-hour momentum (categorical) - BTC and SPX only
for asset in ["BTC", "SPX"]:
    for old_name in [f"{asset}_mom_12h"]:
        if old_name in df.columns:
            sample_val = df[old_name].dropna().iloc[0] if len(df[old_name].dropna()) > 0 else None
            if sample_val is not None and isinstance(sample_val, str):
                if old_name not in selected_features:
                    selected_features.append(old_name)
                    rename_map[old_name] = f"{asset}_mom_12h"
                break

# RSI for all assets
for asset in ["BTC", "SPX", "US10Y"]:
    rsi_col = f"{asset}_RSI_14"
    if rsi_col in df.columns:
        selected_features.append(rsi_col)
        if asset == "BTC":
            rename_map[rsi_col] = "BTC_RSI"
        else:
            rename_map[rsi_col] = f"{asset}_RSI"

# Extract features
features_subset = df[selected_features].copy()

# Apply renaming
if rename_map:
    features_subset = features_subset.rename(columns=rename_map)

# Format datetime for readability
features_subset["datetime"] = pd.to_datetime(features_subset["datetime"]).dt.strftime("%Y-%m-%d %H:%M")

# Round numeric columns
numeric_cols = features_subset.select_dtypes(include=[float, int]).columns
for col in numeric_cols:
    if col != "datetime":
        features_subset[col] = features_subset[col].round(4)

# Get last 10 rows for presentation (most recent data)
features_pres = features_subset.tail(10).copy()

# Remove rows with missing values in essential columns only (not all columns)
# Keep rows that have at least the main features
essential_cols = ["datetime", "BTC_log_ret_12h", "BTC_vol_12h", "BTC_mom_12h"]
essential_cols = [c for c in essential_cols if c in features_pres.columns]
features_pres = features_pres.dropna(subset=essential_cols)

# Save full dataset
output_path = "Df_pres/selected_features.csv"
try:
    features_subset.dropna().to_csv(output_path, index=False)
    saved_full = True
except PermissionError:
    print("⚠ Could not save full dataset (file may be open)")
    saved_full = False

# Save presentation sample
pres_path = "Df_pres/selected_features_pres.csv"
try:
    features_pres.to_csv(pres_path, index=False)
    saved_pres = True
except PermissionError:
    print("⚠ Could not save presentation sample (file may be open)")
    saved_pres = False

print("\n" + "="*80)
print("PRESENTATION-READY TABLE (Last 10 Hours)")
print("Features: 12-hour lookback for predicting 7-day forward BTC returns")
print("="*80 + "\n")

# Format for presentation
if HAS_TABULATE:
    print("--- Grid Format (for slides) ---")
    print(tabulate(features_pres, headers='keys', tablefmt='grid', showindex=False))
    print("\n")
    
    print("--- Simple Format (easy copy-paste) ---")
    print(tabulate(features_pres, headers='keys', tablefmt='simple', showindex=False))
    print("\n")
else:
    print(features_pres.to_string(index=False))
    print("\n")

# HTML table for PowerPoint/Google Slides
print("--- HTML Table (copy into PowerPoint/Google Slides) ---")
html_table = features_pres.to_html(index=False, classes='table', table_id='features_pres')
print(html_table)
print("\n")

if saved_full:
    print(f"✓ Full dataset ({len(features_subset.dropna())} rows) saved to: {output_path}")
if saved_pres:
    print(f"✓ Presentation sample ({len(features_pres)} rows) saved to: {pres_path}")
print("\n" + "="*80)

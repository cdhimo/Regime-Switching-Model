# -----------------------------------------
# Present target variables: Forward log returns
# (1h, 1d, 7d forward BTC log returns)
# -----------------------------------------
import pandas as pd

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Note: Install 'tabulate' for prettier tables: pip install tabulate\n")

# Load the features dataset which contains forward returns
df = pd.read_csv("Data/regime_features_hourly_categorical.csv", parse_dates=["datetime"])

# Select target variables (forward log returns)
target_cols = [
    "datetime",
    "BTC_fwd_1h_log_ret",
    "BTC_fwd_1d_log_ret",
    "BTC_fwd_7d_log_ret",
]

# Check which columns exist
available_cols = [col for col in target_cols if col in df.columns]
targets = df[available_cols].copy()

# Get last 10 rows for presentation (most recent data)
targets_pres = targets.tail(10).copy()

# Remove rows with all NaN target values
targets_pres = targets_pres.dropna(subset=[col for col in targets_pres.columns if col != "datetime"])

# Format datetime for readability
targets_pres["datetime"] = pd.to_datetime(targets_pres["datetime"]).dt.strftime("%Y-%m-%d %H:%M")

# Round numeric columns
numeric_cols = targets_pres.select_dtypes(include=[float, int]).columns
for col in numeric_cols:
    if col != "datetime":
        targets_pres[col] = targets_pres[col].round(6)

# Rename columns for presentation
targets_pres = targets_pres.rename(columns={
    "datetime": "Time (UTC)",
    "BTC_fwd_1h_log_ret": "1h Forward Return",
    "BTC_fwd_1d_log_ret": "1d Forward Return",
    "BTC_fwd_7d_log_ret": "7d Forward Return",
})

# Save full dataset
output_path = "Df_pres/target_variables.csv"
try:
    targets.dropna().to_csv(output_path, index=False)
    saved_full = True
except PermissionError:
    print("⚠ Could not save full dataset (file may be open)")
    saved_full = False

# Save presentation sample
pres_path = "Df_pres/target_variables_pres.csv"
try:
    targets_pres.to_csv(pres_path, index=False)
    saved_pres = True
except PermissionError:
    print("⚠ Could not save presentation sample (file may be open)")
    saved_pres = False

print("\n" + "="*80)
print("TARGET VARIABLES: FORWARD LOG RETURNS (Last 10 Hours)")
print("BTC Forward Returns: 1h, 1d, 7d")
print("="*80 + "\n")

# Format for presentation
if HAS_TABULATE:
    print("--- Grid Format (for slides) ---")
    print(tabulate(targets_pres, headers='keys', tablefmt='grid', showindex=False))
    print("\n")
    
    print("--- Simple Format (easy copy-paste) ---")
    print(tabulate(targets_pres, headers='keys', tablefmt='simple', showindex=False))
    print("\n")
else:
    print(targets_pres.to_string(index=False))
    print("\n")

# HTML table for PowerPoint/Google Slides
print("--- HTML Table (copy into PowerPoint/Google Slides) ---")
html_table = targets_pres.to_html(index=False, classes='table', table_id='target_vars_pres')
print(html_table)
print("\n")

if saved_full:
    print(f"✓ Full dataset ({len(targets.dropna())} rows) saved to: {output_path}")
if saved_pres:
    print(f"✓ Presentation sample ({len(targets_pres)} rows) saved to: {pres_path}")
print("\n" + "="*80)


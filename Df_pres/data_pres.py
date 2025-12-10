# -----------------------------------------
# Pretty sample table for presentation
# -----------------------------------------
import pandas as pd

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Note: Install 'tabulate' for prettier tables: pip install tabulate\n")

assets_to_show = ["BTC", "SPX", "US10Y"]

combined_df = pd.read_csv("Data/regime_data.csv")
combined_df["datetime"] = pd.to_datetime(combined_df["datetime"])

# Calculate 12-hour log returns directly from price data
import numpy as np

# First, calculate 12h log returns for all assets
combined_df = combined_df.sort_values(["symbol", "datetime"])
combined_df["log_ret_12h"] = combined_df.groupby("symbol")["close"].apply(
    lambda x: np.log(x / x.shift(12))
).reset_index(level=0, drop=True)

# Then get the sample
sample = (
    combined_df
    .query("symbol in @assets_to_show")
    .sort_values(["symbol", "datetime"])
    .groupby("symbol")
    .tail(4)   # last 4 hours for each asset
    .loc[:, ["datetime", "symbol", "close", "vol_realized", "log_ret_12h"]]
)

# Format for readability
sample["datetime"] = pd.to_datetime(sample["datetime"]).dt.strftime("%Y-%m-%d %H:%M")
sample = sample.round({
    "close": 2,
    "vol_realized": 6,
    "log_ret_12h": 6,
})

# Rename columns for the slide
sample = sample.rename(columns={
    "datetime": "Time (UTC)",
    "symbol": "Asset",
    "close": "Price",
    "vol_realized": "24h Realized Vol",
    "log_ret_12h": "12h Log Return"
})

# Reorder columns for better presentation
column_order = ["Time (UTC)", "Asset", "Price", "12h Log Return", "24h Realized Vol"]
column_order = [col for col in column_order if col in sample.columns]
sample = sample[column_order]

# ============================================
# Multiple output formats for presentations
# ============================================

print("\n" + "="*70)
print("SAMPLE HOURLY DATA (BTC, SPX, US10Y)")
print("="*70 + "\n")

# Option 1: Pretty table with tabulate (if available)
if HAS_TABULATE:
    print("--- Formatted Table (for terminal/slides) ---")
    print(tabulate(sample, headers='keys', tablefmt='grid', showindex=False))
    print("\n")
    
    print("--- Simple Table (for copy-paste) ---")
    print(tabulate(sample, headers='keys', tablefmt='simple', showindex=False))
    print("\n")
else:
    # Fallback to pandas formatting
    print("--- Formatted Table ---")
    print(sample.to_string(index=False))
    print("\n")

# Option 2: HTML table (great for PowerPoint/Google Slides)
print("--- HTML Table (copy into PowerPoint/Google Slides) ---")
html_table = sample.to_html(index=False, classes='table', table_id='sample_data')
print(html_table)
print("\n")

# Option 3: Markdown table (for markdown presentations)
print("--- Markdown Table (for markdown presentations) ---")
print(sample.to_markdown(index=False))
print("\n")

# Option 4: Save to files for easy import
try:
    sample.to_csv("Df_pres/sample_presentation_data.csv", index=False)
    print("✓ Saved to: Df_pres/sample_presentation_data.csv")
except PermissionError:
    print("⚠ Could not save CSV (file may be open)")

try:
    sample.to_excel("Df_pres/sample_presentation_data.xlsx", index=False, sheet_name="Sample Data")
    print("✓ Saved to: Df_pres/sample_presentation_data.xlsx")
except ImportError:
    print("(Excel export requires 'openpyxl': pip install openpyxl)")
except PermissionError:
    print("⚠ Could not save Excel (file may be open)")

print("\n" + "="*70)

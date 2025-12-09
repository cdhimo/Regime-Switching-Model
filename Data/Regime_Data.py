import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict

# -----------------------------------
# CONFIG
# -----------------------------------
BENCHMARKS_BASE = "https://benchmarks.pyth.network"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pyth-regime-model/1.0"})

BTC_FEED_ID   = "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
SPX_FEED_ID   = "0x2817b78438c769357182c04346fddaad1178c82f4048828fe0997c3c64624e14"
US10Y_FEED_ID = "0x9c196541230ba421baa2a499214564312a46bb47fb6b61ef63db2f70d3ce34c1"


def normalize_id(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    return s[2:] if s.startswith("0x") else s


# 1. Build hourly timestamps for last N days
def generate_recent_hourly_timestamps(days_back: int = 5,
                                      end_dt: datetime | None = None) -> List[int]:
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)

    end_dt = end_dt.replace(minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days_back)
    start_dt = start_dt.replace(minute=0, second=0, microsecond=0)

    ts_list = []
    cur = start_dt
    while cur <= end_dt:
        ts_list.append(int(cur.timestamp()))
        cur += timedelta(hours=1)
    return ts_list


# 2. Parse Benchmarks response
def parse_benchmarks_response(raw: Dict, feed_id: str):
    target = normalize_id(feed_id)
    records = []

    for entry in raw.get("parsed", []):
        if normalize_id(entry.get("id")) != target:
            continue

        p = entry["price"]
        records.append(
            {
                "publish_time": int(p["publish_time"]),
                "price": int(p["price"]),
                "expo": int(p["expo"]),
                "conf": int(p["conf"]),
            }
        )
    return records


# 3. Fetch 1 price per hour for a single feed, with 429 handling
def fetch_hourly_prices(feed_id: str,
                        timestamps: List[int],
                        per_request_delay: float = 0.1,
                        max_retries_429: int = 3) -> pd.DataFrame:
    all_rows = []

    for ts in timestamps:
        url = f"{BENCHMARKS_BASE}/v1/updates/price/{ts}"
        params = {"ids": [feed_id]}

        retries = 0
        while True:
            resp = SESSION.get(url, params=params, timeout=10)

            if resp.status_code == 404:
                # no data for this hour → skip
                # print(f"[404] {feed_id} ts={ts}")
                break

            if resp.status_code == 429:
                retries += 1
                if retries > max_retries_429:
                    raise RuntimeError(
                        f"Hit rate limit too many times for feed {feed_id}. "
                        f"Try smaller days_back or increase delay."
                    )
                # backoff
                wait_sec = 1.0 * retries
                print(f"[429] Rate limited on {feed_id} ts={ts}, retry {retries} in {wait_sec}s...")
                time.sleep(wait_sec)
                continue  # retry same ts

            resp.raise_for_status()
            raw = resp.json()
            rows = parse_benchmarks_response(raw, feed_id)

            for r in rows:
                all_rows.append(r)
                dt = datetime.fromtimestamp(r["publish_time"], tz=timezone.utc)
                px = r["price"] * (10 ** r["expo"])
                print(f"ADDED → time={dt}, price={px:.2f}, feed={feed_id[:10]}...")

            break  # success → move to next timestamp

        # tiny delay between calls to be nice
        time.sleep(per_request_delay)

    if not all_rows:
        raise ValueError(
            f"No data returned for feed {feed_id}. "
            f"Check ID or make sure there is history in this window."
        )

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["publish_time"])
    df["datetime"] = pd.to_datetime(df["publish_time"], unit="s", utc=True)
    df = df.set_index("datetime").sort_index()

    # safe conversion with negative exponents
    df["price"] = df["price"].astype(float)
    df["expo"] = df["expo"].astype(int)
    df["px"] = df["price"] * np.power(10.0, df["expo"])

    return df[["px", "conf"]]


# 4. OHLC + volatility
def build_ohlc_with_vol(df_prices: pd.DataFrame,
                        vol_window_hours: int = 24) -> pd.DataFrame:
    df_resampled = df_prices.resample("1H").last().ffill()

    ohlc = df_resampled["px"].resample("1H").agg(
        open="first",
        high="max",
        low="min",
        close="last",
    )

    ohlc["log_ret_1h"] = np.log(ohlc["close"]).diff()
    ohlc["vol_realized"] = ohlc["log_ret_1h"].rolling(vol_window_hours).std()

    return ohlc


# 5. Wrapper: BTC / SPX / US10Y
def load_three_assets_1h_ohlc(days_back: int = 5,
                              vol_window_hours: int = 24):
    print(f"Building hourly timestamps for last {days_back} days...")
    ts = generate_recent_hourly_timestamps(days_back=days_back)

    assets = {
        "BTC": BTC_FEED_ID,
        "SPX": SPX_FEED_ID,
        "US10Y": US10Y_FEED_ID,
    }

    result = {}

    for name, fid in assets.items():
        if "REPLACE_WITH" in fid:
            print(f"Skipping {name}: feed id not set.")
            continue

        print(f"\n--- {name} ---")
        df_prices = fetch_hourly_prices(fid, ts)
        print(f"{name}: collected {len(df_prices)} raw points")

        df_ohlc = build_ohlc_with_vol(df_prices, vol_window_hours=vol_window_hours)
        result[name] = df_ohlc

    return result


if __name__ == "__main__":
    # While debugging, keep days_back small
    data = load_three_assets_1h_ohlc(days_back=90, vol_window_hours=24)
    print(data)

    btc_ohlcv = data.get("BTC")
    if btc_ohlcv is not None:
        print("\nBTC OHLC+vol sample:")
        print(btc_ohlcv.head())
    
    # Combine all assets, keeping datetime as index
    combined_rows = []
    for asset_name, df in data.items():
        if df is not None and not df.empty:
            df_copy = df.copy()
            df_copy['symbol'] = asset_name
            combined_rows.append(df_copy)
    
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=False)
        combined_df = combined_df.sort_index()  # Sort by datetime
        combined_df.to_csv("regime_data.csv")
        print(f"\nSaved combined data to regime_data.csv ({len(combined_df)} rows)")
        print(f"Columns: {list(combined_df.columns)}")


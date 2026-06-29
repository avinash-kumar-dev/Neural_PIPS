import os
import sys
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ID = "CarlosSilva1/xauusd-ticks"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xauusd", "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "xauusd_m1_hf.parquet")

YEARS = ["2021", "2022", "2023", "2024", "2025", "2026"]
MONTHS_PER_YEAR = {
    "2021": list(range(5, 13)),
    "2022": list(range(1, 13)),
    "2023": list(range(1, 13)),
    "2024": list(range(1, 13)),
    "2025": list(range(1, 13)),
    "2026": list(range(1, 6)),
}


def ticks_to_m1(df_ticks: pd.DataFrame) -> pd.DataFrame:
    if df_ticks.empty:
        return pd.DataFrame()

    if "timestamp_ms" in df_ticks.columns:
        df_ticks["datetime"] = pd.to_datetime(df_ticks["timestamp_ms"], unit="ms", utc=True)
    elif "Time" in df_ticks.columns:
        df_ticks["datetime"] = pd.to_datetime(df_ticks["Time"], utc=True)
    else:
        for col in df_ticks.columns:
            if "time" in col.lower() or "date" in col.lower():
                df_ticks["datetime"] = pd.to_datetime(df_ticks[col], utc=True)
                break

    df_ticks["datetime"] = df_ticks["datetime"].dt.tz_localize(None)
    df_ticks["minute"] = df_ticks["datetime"].dt.floor("min")

    bid_col = None
    ask_col = None
    vol_col = None
    for col in df_ticks.columns:
        cl = col.lower()
        if cl == "bid":
            bid_col = col
        elif cl == "ask":
            ask_col = col
        elif cl == "volume" or cl == "vol" or cl == "tick_volume":
            vol_col = col

    if bid_col is None:
        for col in df_ticks.columns:
            if df_ticks[col].dtype in [np.float64, np.float32]:
                bid_col = col
                break

    if bid_col is None:
        return pd.DataFrame()

    price_col = bid_col

    grouped = df_ticks.groupby("minute")

    m1_data = []
    for minute, group in grouped:
        o = group[price_col].iloc[0]
        h = group[price_col].max()
        l = group[price_col].min()
        c = group[price_col].iloc[-1]
        v = len(group)
        m1_data.append({
            "datetime": minute,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        })

    return pd.DataFrame(m1_data)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_m1_chunks = []
    total_ticks = 0
    total_m1 = 0

    for year in YEARS:
        months = MONTHS_PER_YEAR[year]
        for month in months:
            filename = f"XAUUSD-{year}-{month:02d}-part0000.parquet"
            path_in_repo = f"year={year}/month={month:02d}/{filename}"

            print(f"Downloading {path_in_repo}...", end=" ", flush=True)
            try:
                local_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=path_in_repo,
                    repo_type="dataset",
                )
                df_ticks = pd.read_parquet(local_path)
                print(f"{len(df_ticks)} ticks", end=" -> ", flush=True)
                total_ticks += len(df_ticks)

                m1 = ticks_to_m1(df_ticks)
                if not m1.empty:
                    all_m1_chunks.append(m1)
                    total_m1 += len(m1)
                    print(f"{len(m1)} M1 bars")
                else:
                    print("empty")

                del df_ticks

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    if not all_m1_chunks:
        print("No data collected!")
        return

    print(f"\nCombining {len(all_m1_chunks)} chunks...")
    all_m1 = pd.concat(all_m1_chunks, ignore_index=True)
    all_m1 = all_m1.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    all_m1.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n=== RESULT ===")
    print(f"Total ticks processed: {total_ticks:,}")
    print(f"Total M1 bars: {len(all_m1):,}")
    print(f"Date range: {all_m1['datetime'].min()} to {all_m1['datetime'].max()}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

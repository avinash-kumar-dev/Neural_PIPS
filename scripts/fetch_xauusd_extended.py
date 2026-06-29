import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt5linux import MetaTrader5

TIMEFRAME_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408,
}

CHUNK_SIZE = 50000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xauusd", "data", "raw")


def fetch_and_save(mt5, symbol, timeframe, tf_code, num_bars, output_path):
    print(f"\nFetching {symbol} {timeframe} ({num_bars} bars max)...")
    chunks = []
    pos = 0
    while pos < num_bars:
        remaining = num_bars - pos
        batch = min(CHUNK_SIZE, remaining)
        rates = mt5.copy_rates_from_pos(symbol, tf_code, pos, batch)
        if rates is None or len(rates) == 0:
            break
        chunks.append(rates)
        pos += len(rates)
        print(f"  Fetched {pos}/{num_bars} bars...", end="\r")
        if len(rates) < batch:
            break

    if not chunks:
        print(f"  FAILED: No data returned")
        return None

    all_rates = np.concatenate(chunks)
    df = pd.DataFrame.from_records(all_rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"time": "datetime", "tick_volume": "volume"})
    df = df[["datetime", "open", "high", "low", "close", "volume", "spread"]]
    df = df.sort_values("datetime").reset_index(drop=True)

    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df)} bars to {output_path}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def main():
    mt5 = MetaTrader5(host="localhost", port=8001)
    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error())
        sys.exit(1)
    mt5.symbol_select("XAUUSD", True)
    print("MT5 connected. MaxBars:", mt5.terminal_info().maxbars)

    timeframes = [
        ("M1", 1, 300000),
        ("M5", 5, 300000),
        ("M15", 15, 300000),
        ("M30", 30, 200000),
        ("H1", 16385, 100000),
        ("H4", 16388, 50000),
    ]

    for tf_name, tf_code, num_bars in timeframes:
        output_path = os.path.join(OUTPUT_DIR, f"xauusd_{tf_name.lower()}.parquet")
        fetch_and_save(mt5, "XAUUSD", tf_name, tf_code, num_bars, output_path)

    mt5.shutdown()
    print("\nDone! All timeframes fetched.")


if __name__ == "__main__":
    main()

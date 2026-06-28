#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

from xauusd.data.fetcher import XAUUSDDataFetcher
from xauusd.data.storage import ParquetStorage

SYMBOL = "XAUUSD"
TIMEFRAMES = ["M5", "M15", "H1", "H4"]
NUM_BARS = 100000


def main():
    storage = ParquetStorage("xauusd/data")

    with XAUUSDDataFetcher(host="localhost", port=8001) as fetcher:
        for tf in TIMEFRAMES:
            print(f"Fetching {SYMBOL} {tf}...", end=" ", flush=True)
            try:
                df = fetcher.fetch_ohlcv(SYMBOL, tf, NUM_BARS)
                name = f"{SYMBOL.lower()}_{tf.lower()}"
                path = storage.save(df, name)
                print(f"{len(df)} bars, {df['datetime'].min()} to {df['datetime'].max()}")
            except Exception as e:
                print(f"ERROR: {e}")

    print("\nDatasets:")
    for ds in storage.list_datasets():
        df = storage.load(ds)
        print(f"  {ds}: {len(df)} bars, {df['datetime'].min()} to {df['datetime'].max()}")


if __name__ == "__main__":
    main()

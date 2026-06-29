import sys
sys.path.insert(0, ".")
from xauusd.data.fetcher import XAUUSDDataFetcher

fetcher = XAUUSDDataFetcher()
fetcher.connect()

print("Fetching 100K M1 bars for XAUUSD...")
df = fetcher.fetch_ohlcv("XAUUSD", "M1", 100000)

print(f"Total: {len(df)} bars")
print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

df.to_parquet("xauusd/data/raw/xauusd_m1.parquet", index=False)
print(f"Saved to xauusd/data/raw/xauusd_m1.parquet")

fetcher.disconnect()

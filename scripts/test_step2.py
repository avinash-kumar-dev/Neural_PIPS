import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_pipeline import DataFetcher, ParquetStorage, MT5Connection


def test_connection():
    print("=== MT5 Docker Connection Test ===")
    conn = MT5Connection()
    try:
        info = conn.connect()
        print(f"Connected to MT5: OK")
        print(f"Account: {info.login} | Server: {info.server} | Balance: {info.balance}")
        conn.disconnect()
        print("Disconnected: OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_data_fetch():
    print("\n=== Data Fetch Test ===")
    fetcher = DataFetcher()
    try:
        fetcher.connect()

        m5_data = fetcher.fetch_ohlcv('M5', 100)
        print(f"M5 candles fetched: {len(m5_data)}")
        if not m5_data.empty:
            print(f"Latest: {m5_data['time'].iloc[-1]} | Close: {m5_data['close'].iloc[-1]}")

        spread = fetcher.fetch_spread()
        print(f"Current spread: {spread:.1f} pips")

        fetcher.disconnect()
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_parquet_storage():
    print("\n=== Parquet Storage Test ===")
    storage = ParquetStorage()
    fetcher = DataFetcher()
    try:
        fetcher.connect()
        df = fetcher.fetch_ohlcv('M5', 100)
        storage.save(df, 'eurusd', 'M5')
        print("Save to Parquet: OK")

        loaded = storage.load('eurusd', 'M5')
        print(f"Load from Parquet: {len(loaded)} rows")

        fetcher.disconnect()
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_multi_timeframe():
    print("\n=== Multi-Timeframe Fetch Test ===")
    fetcher = DataFetcher()
    try:
        fetcher.connect()
        for tf in ['M5', 'M15', 'H1', 'H4']:
            df = fetcher.fetch_ohlcv(tf, 50)
            print(f"  {tf}: {len(df)} candles")
        fetcher.disconnect()
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


if __name__ == "__main__":
    results = []
    results.append(("MT5 Connection", test_connection()))
    results.append(("Data Fetch", test_data_fetch()))
    results.append(("Parquet Storage", test_parquet_storage()))
    results.append(("Multi-Timeframe", test_multi_timeframe()))

    print("\n=== Summary ===")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

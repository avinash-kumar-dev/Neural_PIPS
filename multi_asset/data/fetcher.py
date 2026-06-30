import pandas as pd
import numpy as np
from pathlib import Path
from mt5linux import MetaTrader5

TIMEFRAME_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408,
}

CHUNK_SIZE = 50000


class MultiAssetFetcher:
    def __init__(self, host="localhost", port=8001):
        self.mt5 = MetaTrader5(host=host, port=port)
        self.connected = False

    def connect(self):
        if not self.mt5.initialize():
            raise ConnectionError(f"MT5 init failed: {self.mt5.last_error()}")
        self.connected = True
        return self

    def disconnect(self):
        if self.mt5:
            self.mt5.shutdown()
            self.connected = False

    def __enter__(self):
        return self.connect()

    def __exit__(self, *args):
        self.disconnect()

    def fetch_ohlcv(self, symbol: str, timeframe: str, num_bars: int = 100000) -> pd.DataFrame:
        tf_code = TIMEFRAME_MAP.get(timeframe)
        if tf_code is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        self.mt5.symbol_select(symbol, True)

        chunks = []
        pos = 0
        while pos < num_bars:
            batch = min(CHUNK_SIZE, num_bars - pos)
            rates = self.mt5.copy_rates_from_pos(symbol, tf_code, pos, batch)
            if rates is None or len(rates) == 0:
                break
            chunks.append(rates)
            pos += len(rates)
            if len(rates) < batch:
                break

        if not chunks:
            raise RuntimeError(f"No data for {symbol} {timeframe}")

        all_rates = np.concatenate(chunks)
        df = pd.DataFrame.from_records(all_rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"time": "datetime", "tick_volume": "volume"})
        df = df[["datetime", "open", "high", "low", "close", "volume", "spread"]]
        df = df.sort_values("datetime").reset_index(drop=True)
        return df

    def fetch_all_instruments(self, instruments: dict, timeframe: str = "M15", num_bars: int = 100000) -> dict:
        data = {}
        for name, cfg in instruments.items():
            print(f"  Fetching {name} ({cfg.mt5_symbol}) {timeframe}...")
            try:
                df = self.fetch_ohlcv(cfg.mt5_symbol, timeframe, num_bars)
                data[name] = df
                print(f"    -> {len(df)} bars, {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
            except Exception as e:
                print(f"    -> ERROR: {e}")
        return data

    def fetch_multi_timeframe(self, symbol: str, timeframes: list, num_bars: int = 100000) -> dict:
        data = {}
        for tf in timeframes:
            print(f"  Fetching {symbol} {tf}...")
            try:
                df = self.fetch_ohlcv(symbol, tf, num_bars)
                data[tf] = df
                print(f"    -> {len(df)} bars")
            except Exception as e:
                print(f"    -> ERROR: {e}")
        return data


def save_data(data: dict, directory: str):
    Path(directory).mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        path = Path(directory) / f"{name.lower()}_m15.parquet"
        df.to_parquet(path, index=False)
        print(f"  Saved {path} ({len(df)} bars)")


def load_data(directory: str, symbols: list = None) -> dict:
    data = {}
    directory = Path(directory)
    for f in directory.glob("*_m15.parquet"):
        name = f.stem.replace("_m15", "").upper()
        if symbols and name not in symbols:
            continue
        data[name] = pd.read_parquet(f)
    return data

from mt5linux import MetaTrader5
import pandas as pd
import numpy as np
from datetime import datetime

TIMEFRAME_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408,
}

CHUNK_SIZE = 50000


class XAUUSDDataFetcher:
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.mt5 = None

    def connect(self):
        self.mt5 = MetaTrader5(host=self.host, port=self.port)
        if not self.mt5.initialize():
            raise ConnectionError(f"MT5 init failed: {self.mt5.last_error()}")
        self.mt5.symbol_select("XAUUSD", True)
        return self.mt5

    def disconnect(self):
        if self.mt5:
            self.mt5.shutdown()
            self.mt5 = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        num_bars: int = 100000,
    ) -> pd.DataFrame:
        tf_code = TIMEFRAME_MAP.get(timeframe)
        if tf_code is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        if self.mt5 is None:
            raise ConnectionError("Not connected")

        chunks = []
        pos = 0
        while pos < num_bars:
            remaining = num_bars - pos
            batch = min(CHUNK_SIZE, remaining)
            rates = self.mt5.copy_rates_from_pos(symbol, tf_code, pos, batch)
            if rates is None or len(rates) == 0:
                break
            chunks.append(rates)
            pos += len(rates)
            if len(rates) < batch:
                break

        if not chunks:
            raise RuntimeError(f"No data returned for {symbol} {timeframe}")

        all_rates = np.concatenate(chunks)
        df = pd.DataFrame.from_records(all_rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"time": "datetime", "tick_volume": "volume"})
        df = df[["datetime", "open", "high", "low", "close", "volume", "spread"]]
        df = df.sort_values("datetime").reset_index(drop=True)
        return df

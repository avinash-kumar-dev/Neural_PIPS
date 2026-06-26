import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TIMEFRAME_MAP = {
    'M1': 1,
    'M5': 5,
    'M15': 15,
    'M30': 30,
    'H1': 16385,
    'H4': 16388,
    'D1': 16408,
    'W1': 32769,
    'MN1': 49153,
}


class MT5Connection:
    def __init__(self, host='localhost', port=8001):
        self.host = host
        self.port = port
        self.mt5 = None

    def connect(self):
        from mt5linux import MetaTrader5
        self.mt5 = MetaTrader5(host=self.host, port=self.port)
        if not self.mt5.initialize():
            raise ConnectionError(f"MT5 initialize failed: {self.mt5.last_error()}")
        info = self.mt5.account_info()
        if info is None:
            raise ConnectionError("Failed to get account info — is MT5 connected to a broker?")
        return info

    def disconnect(self):
        if self.mt5:
            self.mt5.shutdown()

    def get_account(self):
        if self.mt5 is None:
            raise ConnectionError("Not connected")
        return self.mt5.account_info()

    def get_symbol_info(self, symbol):
        if self.mt5 is None:
            raise ConnectionError("Not connected")
        return self.mt5.symbol_info(symbol)

    def get_rates(self, symbol, timeframe, count=500):
        if self.mt5 is None:
            raise ConnectionError("Not connected")
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {symbol} TF={timeframe}")
        return rates

    def get_tick(self, symbol):
        if self.mt5 is None:
            raise ConnectionError("Not connected")
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"No tick data for {symbol}")
        return tick


class DataFetcher:
    SYMBOL = "EURUSD"

    def __init__(self, host='localhost', port=8001):
        self.conn = MT5Connection(host=host, port=port)

    def connect(self):
        return self.conn.connect()

    def fetch_ohlcv(self, timeframe, count=500):
        tf_code = TIMEFRAME_MAP[timeframe]
        rates = self.conn.get_rates(self.SYMBOL, tf_code, count)

        df = pd.DataFrame(rates)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def fetch_all(self):
        return {
            'M5': self.fetch_ohlcv('M5', 500),
            'M15': self.fetch_ohlcv('M15', 200),
            'H1': self.fetch_ohlcv('H1', 200),
            'H4': self.fetch_ohlcv('H4', 100),
        }

    def fetch_max(self):
        return {
            'M5': self.fetch_ohlcv('M5', 95000),
            'M15': self.fetch_ohlcv('M15', 40000),
            'H1': self.fetch_ohlcv('H1', 50000),
            'H4': self.fetch_ohlcv('H4', 20000),
        }

    def fetch_spread(self):
        tick = self.conn.get_tick(self.SYMBOL)
        spread_pips = (tick.ask - tick.bid) * 10000
        return spread_pips

    def disconnect(self):
        self.conn.disconnect()


class ParquetStorage:
    BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    def save(self, df, asset, timeframe, date=None):
        if date is None:
            date = datetime.utcnow()
        dir_path = os.path.join(self.BASE_DIR, asset, timeframe)
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{date.strftime('%Y-%m')}.parquet"
        filepath = os.path.join(dir_path, filename)

        if os.path.exists(filepath):
            existing = pd.read_parquet(filepath)
            df = pd.concat([existing, df]).drop_duplicates(
                subset=['time'], keep='last'
            ).sort_values('time').reset_index(drop=True)

        df.to_parquet(filepath, index=False)

    def load(self, asset, timeframe, start_date=None, end_date=None):
        all_data = []
        dir_path = os.path.join(self.BASE_DIR, asset, timeframe)
        if not os.path.exists(dir_path):
            return pd.DataFrame()

        for f in sorted(os.listdir(dir_path)):
            if f.endswith('.parquet'):
                month_str = f.replace('.parquet', '')
                month_date = datetime.strptime(month_str, '%Y-%m')
                if start_date and month_date < start_date:
                    continue
                if end_date and month_date > end_date:
                    continue
                df = pd.read_parquet(os.path.join(dir_path, f))
                all_data.append(df)

        if all_data:
            return pd.concat(all_data).drop_duplicates(
                subset=['time'], keep='last'
            ).sort_values('time').reset_index(drop=True)
        return pd.DataFrame()

    def load_all(self, asset, timeframe):
        dir_path = os.path.join(self.BASE_DIR, asset, timeframe)
        if not os.path.exists(dir_path):
            return pd.DataFrame()
        all_data = []
        for f in sorted(os.listdir(dir_path)):
            if f.endswith('.parquet'):
                df = pd.read_parquet(os.path.join(dir_path, f))
                all_data.append(df)
        if all_data:
            return pd.concat(all_data).drop_duplicates(
                subset=['time'], keep='last'
            ).sort_values('time').reset_index(drop=True)
        return pd.DataFrame()


class DataPipeline:
    def __init__(self, host='localhost', port=8001):
        self.fetcher = DataFetcher(host=host, port=port)
        self.storage = ParquetStorage()

    def connect(self):
        return self.fetcher.connect()

    def pull_historical(self):
        all_data = {}
        counts = {'M5': 95000, 'M15': 40000, 'H1': 50000, 'H4': 20000}

        for tf in ['M5', 'M15', 'H1', 'H4']:
            total_count = counts[tf]
            print(f"Fetching {tf} ({total_count} bars) from MT5...")
            df = self.fetcher.fetch_ohlcv(tf, total_count)
            if df.empty:
                print(f"  No data for {tf}")
                continue
            all_data[tf] = df
            self.storage.save(df, 'eurusd', tf)
            print(f"  Saved {len(df)} bars to data/eurusd/{tf}/")

        return all_data

    def fetch_current(self):
        return self.fetcher.fetch_all()

    def fetch_spread(self):
        return self.fetcher.fetch_spread()

    def load_historical(self, timeframe, start_date=None, end_date=None):
        return self.storage.load('eurusd', timeframe, start_date, end_date)

    def load_all(self, timeframe):
        return self.storage.load_all('eurusd', timeframe)

    def disconnect(self):
        self.fetcher.disconnect()

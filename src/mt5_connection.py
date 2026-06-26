import os
from dotenv import load_dotenv

load_dotenv()


class MT5Connection:
    def __init__(self, host=None, port=None):
        self.host = host or os.getenv('MT5_HOST', 'localhost')
        self.port = port or int(os.getenv('MT5_PORT', '8001'))
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
            self.mt5 = None

    def heartbeat(self):
        try:
            from mt5linux import MetaTrader5
            test = MetaTrader5(host=self.host, port=self.port)
            info = test.account_info()
            return info is not None
        except Exception:
            return False

    def get_account_info(self):
        if self.mt5 is None:
            raise ConnectionError("Not connected")
        return self.mt5.account_info()

    def get_symbol_info(self, symbol="EURUSD"):
        if self.mt5 is None:
            raise ConnectionError("Not connected")
        return self.mt5.symbol_info(symbol)

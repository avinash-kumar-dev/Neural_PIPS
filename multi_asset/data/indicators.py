import pandas as pd
import numpy as np
import pandas_ta as ta


def compute_basic_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    df = df.copy()

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=atr_period)

    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)

    df["rsi"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"])
    if macd is not None:
        df["macd"] = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"] = macd.iloc[:, 2]

    df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14).iloc[:, 0]

    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None:
        df["bb_upper"] = bb.iloc[:, 2]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_lower"] = bb.iloc[:, 0]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=1.5)
    if kc is not None:
        df["kc_upper"] = kc.iloc[:, 2]
        df["kc_mid"] = kc.iloc[:, 1]
        df["kc_lower"] = kc.iloc[:, 0]

    tmp = df.set_index("datetime")
    tmp.index = pd.to_datetime(tmp.index)
    tmp = tmp.sort_index()
    df["vwap"] = ta.vwap(tmp["high"], tmp["low"], tmp["close"], tmp["volume"], anchor="D")

    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    if st is not None:
        df["supertrend"] = st.iloc[:, 0]
        df["supertrend_dir"] = st.iloc[:, 1]

    ichi = ta.ichimoku(df["high"], df["low"], df["close"])
    if ichi is not None and isinstance(ichi, tuple):
        df["ichi_a"] = ichi[0].iloc[:, 0]
        df["ichi_b"] = ichi[0].iloc[:, 1]
        df["ichi_base"] = ichi[0].iloc[:, 2]
        df["ichi_conv"] = ichi[0].iloc[:, 3]

    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]

    df["body"] = abs(df["close"] - df["open"])
    df["wick_upper"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["wick_lower"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]
    df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)

    df["bullish_engulfing"] = (df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & (df["close"] > df["open"].shift(1)) & (df["open"] < df["close"].shift(1))
    df["bearish_engulfing"] = (df["close"] < df["open"]) & (df["close"].shift(1) > df["open"].shift(1)) & (df["close"] < df["open"].shift(1)) & (df["open"] > df["close"].shift(1))

    df["pin_bar_bull"] = (df["wick_lower"] > 2 * df["body"]) & (df["wick_lower"] > df["wick_upper"]) & (df["body"] > 0)
    df["pin_bar_bear"] = (df["wick_upper"] > 2 * df["body"]) & (df["wick_upper"] > df["wick_lower"]) & (df["body"] > 0)

    return df


def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    df = df.copy()
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        left_h = highs[i - lookback:i]
        right_h = highs[i + 1:i + lookback + 1]
        if highs[i] >= np.max(left_h) and highs[i] >= np.max(right_h):
            swing_high[i] = True

        left_l = lows[i - lookback:i]
        right_l = lows[i + 1:i + lookback + 1]
        if lows[i] <= np.min(left_l) and lows[i] <= np.min(right_l):
            swing_low[i] = True

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    df["swing_high_price"] = np.where(swing_high, highs, np.nan)
    df["swing_low_price"] = np.where(swing_low, lows, np.nan)

    return df


def detect_bos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].values
    sh = df["swing_high_price"].values
    sl = df["swing_low_price"].values
    n = len(df)

    bos_bull = np.zeros(n, dtype=bool)
    bos_bear = np.zeros(n, dtype=bool)

    last_sh = np.nan
    last_sl = np.nan

    for i in range(n):
        if not np.isnan(sh[i]):
            last_sh = sh[i]
        if not np.isnan(sl[i]):
            last_sl = sl[i]

        if not np.isnan(last_sh) and close[i] > last_sh:
            bos_bull[i] = True
        if not np.isnan(last_sl) and close[i] < last_sl:
            bos_bear[i] = True

    df["bos_bull"] = bos_bull
    df["bos_bear"] = bos_bear
    return df

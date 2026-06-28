import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_indicator_triggers(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
) -> pd.DataFrame:
    result = df.copy()

    result["rsi"] = compute_rsi(result["close"], rsi_period)
    macd_line, signal_line, histogram = compute_macd(result["close"])
    result["macd_line"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_hist"] = histogram

    result["macd_cross_bullish"] = (result["macd_hist"] > 0) & (result["macd_hist"].shift(1) <= 0)
    result["macd_cross_bearish"] = (result["macd_hist"] < 0) & (result["macd_hist"].shift(1) >= 0)

    result["rsi_oversold"] = result["rsi"] < rsi_oversold
    result["rsi_overbought"] = result["rsi"] > rsi_overbought

    return result

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class M5Setup:
    index: int
    direction: int
    entry_price: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    trigger_type: str
    m15_type: str
    confluence_score: float
    lots: float


def detect_m5_mss(
    df: pd.DataFrame,
    swing_lookback: int = 5,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values

    left_max = pd.Series(highs).rolling(swing_lookback, min_periods=swing_lookback).max().shift(1).values
    left_min = pd.Series(lows).rolling(swing_lookback, min_periods=swing_lookback).min().shift(1).values

    mss_bullish = np.zeros(n, dtype=bool)
    mss_bearish = np.zeros(n, dtype=bool)

    last_swing_high = np.nan
    last_swing_low = np.nan

    for i in range(swing_lookback, n):
        if not np.isnan(left_max[i]) and highs[i] > left_max[i]:
            last_swing_high = highs[i]
        if not np.isnan(left_min[i]) and lows[i] < left_min[i]:
            last_swing_low = lows[i]

        if not np.isnan(last_swing_low) and closes[i] > last_swing_low:
            mss_bullish[i] = True
            last_swing_low = np.nan

        if not np.isnan(last_swing_high) and closes[i] < last_swing_high:
            mss_bearish[i] = True
            last_swing_high = np.nan

    result["m5_mss_bullish"] = mss_bullish
    result["m5_mss_bearish"] = mss_bearish
    return result


def detect_rejection_candle(
    df: pd.DataFrame,
    min_wick_ratio: float = 1.5,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    opens = result["open"].values
    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values

    body = np.abs(closes - opens)
    upper_wick = highs - np.maximum(opens, closes)
    lower_wick = np.minimum(opens, closes) - lows

    bullish_rejection = np.zeros(n, dtype=bool)
    bearish_rejection = np.zeros(n, dtype=bool)

    for i in range(n):
        if body[i] == 0:
            continue
        if lower_wick[i] > min_wick_ratio * body[i] and closes[i] > opens[i]:
            bullish_rejection[i] = True
        if upper_wick[i] > min_wick_ratio * body[i] and closes[i] < opens[i]:
            bearish_rejection[i] = True

    result["m5_bull_rejection"] = bullish_rejection
    result["m5_bear_rejection"] = bearish_rejection
    return result


def detect_engulfing(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    opens = result["open"].values
    closes = result["close"].values

    bull_engulfing = np.zeros(n, dtype=bool)
    bear_engulfing = np.zeros(n, dtype=bool)

    for i in range(1, n):
        prev_body_top = max(opens[i - 1], closes[i - 1])
        prev_body_bot = min(opens[i - 1], closes[i - 1])
        curr_body_top = max(opens[i], closes[i])
        curr_body_bot = min(opens[i], closes[i])

        if curr_body_bot <= prev_body_bot and curr_body_top >= prev_body_top:
            if closes[i] > opens[i] and closes[i - 1] < opens[i - 1]:
                bull_engulfing[i] = True
            elif closes[i] < opens[i] and closes[i - 1] > opens[i - 1]:
                bear_engulfing[i] = True

    result["m5_bull_engulfing"] = bull_engulfing
    result["m5_bear_engulfing"] = bear_engulfing
    return result


def compute_m5_confirmation(
    df: pd.DataFrame,
    volume_ma_period: int = 20,
    volume_threshold: float = 1.5,
    body_ratio_min: float = 0.6,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    result = detect_m5_mss(result)
    result = detect_rejection_candle(result)
    result = detect_engulfing(result)

    candle_range = result["high"] - result["low"]
    body = np.abs(result["close"] - result["open"])
    result["m5_body_ratio"] = np.where(candle_range > 0, body / candle_range, 0)
    result["m5_strong_body"] = result["m5_body_ratio"] >= body_ratio_min

    result["m5_volume_ma"] = result["volume"].rolling(volume_ma_period).mean()
    result["m5_volume_spike"] = result["volume"] > volume_threshold * result["m5_volume_ma"]

    result["m5_bull_confirm"] = (
        (result["m5_mss_bullish"] | result["m5_bull_rejection"] | result["m5_bull_engulfing"])
        & result["m5_strong_body"]
        & result["m5_volume_spike"]
    )
    result["m5_bear_confirm"] = (
        (result["m5_mss_bearish"] | result["m5_bear_rejection"] | result["m5_bear_engulfing"])
        & result["m5_strong_body"]
        & result["m5_volume_spike"]
    )

    return result

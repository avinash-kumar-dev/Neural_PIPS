import pandas as pd
import numpy as np


def detect_m1_mss(
    df: pd.DataFrame,
    swing_lookback: int = 3,
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

    result["m1_mss_bullish"] = mss_bullish
    result["m1_mss_bearish"] = mss_bearish
    return result


def detect_m1_rejection(
    df: pd.DataFrame,
    min_wick_ratio: float = 2.0,
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

    bull_rej = np.zeros(n, dtype=bool)
    bear_rej = np.zeros(n, dtype=bool)

    for i in range(n):
        if body[i] == 0:
            continue
        if lower_wick[i] > min_wick_ratio * body[i] and closes[i] > opens[i]:
            bull_rej[i] = True
        if upper_wick[i] > min_wick_ratio * body[i] and closes[i] < opens[i]:
            bear_rej[i] = True

    result["m1_bull_rejection"] = bull_rej
    result["m1_bear_rejection"] = bear_rej
    return result


def detect_m1_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    opens = result["open"].values
    closes = result["close"].values

    bull_eng = np.zeros(n, dtype=bool)
    bear_eng = np.zeros(n, dtype=bool)

    for i in range(1, n):
        prev_top = max(opens[i - 1], closes[i - 1])
        prev_bot = min(opens[i - 1], closes[i - 1])
        curr_top = max(opens[i], closes[i])
        curr_bot = min(opens[i], closes[i])

        if curr_bot <= prev_bot and curr_top >= prev_top:
            if closes[i] > opens[i] and closes[i - 1] < opens[i - 1]:
                bull_eng[i] = True
            elif closes[i] < opens[i] and closes[i - 1] > opens[i - 1]:
                bear_eng[i] = True

    result["m1_bull_engulfing"] = bull_eng
    result["m1_bear_engulfing"] = bear_eng
    return result


def detect_m1_displacement(
    df: pd.DataFrame,
    body_ratio_min: float = 0.7,
    volume_threshold: float = 2.0,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    opens = result["open"].values
    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values

    candle_range = highs - lows
    body = np.abs(closes - opens)
    body_ratio = np.where(candle_range > 0, body / candle_range, 0)

    vol = result.get("volume", result.get("tick_volume", pd.Series(1, index=result.index))).values
    vol_ma = pd.Series(vol).rolling(20).mean().values
    vol_spike = vol > volume_threshold * vol_ma

    bull_displacement = (closes > opens) & (body_ratio >= body_ratio_min) & vol_spike
    bear_displacement = (closes < opens) & (body_ratio >= body_ratio_min) & vol_spike

    result["m1_bull_displacement"] = bull_displacement.values
    result["m1_bear_displacement"] = bear_displacement.values
    return result


def compute_m1_entry(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    result = detect_m1_mss(result, swing_lookback=3)
    result = detect_m1_rejection(result, min_wick_ratio=2.0)
    result = detect_m1_engulfing(result)
    result = detect_m1_displacement(result)

    result["m1_bull_entry"] = (
        result["m1_mss_bullish"] |
        result["m1_bull_rejection"] |
        result["m1_bull_engulfing"] |
        result["m1_bull_displacement"]
    )
    result["m1_bear_entry"] = (
        result["m1_mss_bearish"] |
        result["m1_bear_rejection"] |
        result["m1_bear_engulfing"] |
        result["m1_bear_displacement"]
    )

    return result

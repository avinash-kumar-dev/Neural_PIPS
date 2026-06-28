import pandas as pd
import numpy as np


def detect_choch(
    df: pd.DataFrame,
    swing_col_high: str = "swing_high",
    swing_col_low: str = "swing_low",
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)
    choch_bullish = np.zeros(n, dtype=bool)
    choch_bearish = np.zeros(n, dtype=bool)

    last_swing_high = np.nan
    last_swing_low = np.nan
    prev_swing_high = np.nan
    prev_swing_low = np.nan
    trend = 0

    for i in range(n):
        if not np.isnan(result[swing_col_high].iloc[i]):
            prev_swing_high = last_swing_high
            last_swing_high = result[swing_col_high].iloc[i]

        if not np.isnan(result[swing_col_low].iloc[i]):
            prev_swing_low = last_swing_low
            last_swing_low = result[swing_col_low].iloc[i]

        if not np.isnan(last_swing_high) and not np.isnan(last_swing_low):
            if last_swing_high > prev_swing_high and not np.isnan(prev_swing_high):
                trend = 1
            elif last_swing_low < prev_swing_low and not np.isnan(prev_swing_low):
                trend = -1

        if trend == -1 and not np.isnan(last_swing_low):
            if result["close"].iloc[i] > last_swing_high and not np.isnan(last_swing_high):
                choch_bullish[i] = True
                trend = 1

        if trend == 1 and not np.isnan(last_swing_high):
            if result["close"].iloc[i] < last_swing_low and not np.isnan(last_swing_low):
                choch_bearish[i] = True
                trend = -1

    result["choch_bullish"] = choch_bullish
    result["choch_bearish"] = choch_bearish
    return result

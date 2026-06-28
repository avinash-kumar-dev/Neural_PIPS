import pandas as pd
import numpy as np
from typing import Optional


def detect_bos(
    df: pd.DataFrame,
    swing_col_high: str = "swing_high",
    swing_col_low: str = "swing_low",
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)
    bos_bullish = np.zeros(n, dtype=bool)
    bos_bearish = np.zeros(n, dtype=bool)

    last_swing_high = np.nan
    last_swing_low = np.nan

    for i in range(n):
        if not np.isnan(result[swing_col_high].iloc[i]):
            last_swing_high = result[swing_col_high].iloc[i]
        if not np.isnan(result[swing_col_low].iloc[i]):
            last_swing_low = result[swing_col_low].iloc[i]

        if not np.isnan(last_swing_high):
            if result["close"].iloc[i] > last_swing_high:
                bos_bullish[i] = True
                last_swing_high = np.nan

        if not np.isnan(last_swing_low):
            if result["close"].iloc[i] < last_swing_low:
                bos_bearish[i] = True
                last_swing_low = np.nan

    result["bos_bullish"] = bos_bullish
    result["bos_bearish"] = bos_bearish
    return result

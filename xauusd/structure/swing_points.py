import pandas as pd
import numpy as np
from typing import Optional


def detect_swing_points(
    df: pd.DataFrame,
    lookback: int = 5,
    lookforward: Optional[int] = None,
) -> pd.DataFrame:
    if lookforward is None:
        lookforward = lookback

    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)

    left_max = pd.Series(highs).rolling(lookback, min_periods=lookback).max().shift(1).values
    right_max = pd.Series(highs).rolling(lookforward, min_periods=lookforward).max().shift(-lookforward).values

    left_min = pd.Series(lows).rolling(lookback, min_periods=lookback).min().shift(1).values
    right_min = pd.Series(lows).rolling(lookforward, min_periods=lookforward).min().shift(-lookforward).values

    for i in range(lookback, n - lookforward):
        if not np.isnan(left_max[i]) and not np.isnan(right_max[i]):
            if highs[i] > left_max[i] and highs[i] > right_max[i]:
                swing_high[i] = highs[i]
        if not np.isnan(left_min[i]) and not np.isnan(right_min[i]):
            if lows[i] < left_min[i] and lows[i] < right_min[i]:
                swing_low[i] = lows[i]

    result = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    result["swing_high"] = swing_high
    result["swing_low"] = swing_low
    return result

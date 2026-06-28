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

    for i in range(lookback, n - lookforward):
        left_highs = highs[i - lookback : i]
        right_highs = highs[i + 1 : i + 1 + lookforward]
        if highs[i] > np.max(left_highs) and highs[i] > np.max(right_highs):
            swing_high[i] = highs[i]

        left_lows = lows[i - lookback : i]
        right_lows = lows[i + 1 : i + 1 + lookforward]
        if lows[i] < np.min(left_lows) and lows[i] < np.min(right_lows):
            swing_low[i] = lows[i]

    result = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    result["swing_high"] = swing_high
    result["swing_low"] = swing_low
    return result

import pandas as pd
import numpy as np


def compute_premium_discount(
    df: pd.DataFrame,
    lookback: int = 100,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    swing_high = result["high"].rolling(lookback, center=True).max()
    swing_low = result["low"].rolling(lookback, center=True).min()

    range_size = swing_high - swing_low
    range_size = range_size.replace(0, np.nan)

    equilibrium = (swing_high + swing_low) / 2
    position = (result["close"] - swing_low) / range_size

    zone = np.zeros(n, dtype=int)
    zone[position > 0.5] = 1
    zone[position < 0.5] = -1

    result["pd_swing_high"] = swing_high
    result["pd_swing_low"] = swing_low
    result["pd_equilibrium"] = equilibrium
    result["pd_position"] = position
    result["pd_zone"] = zone
    result["pd_premium"] = zone == 1
    result["pd_discount"] = zone == -1
    result["pd_range"] = range_size

    return result

import pandas as pd
import numpy as np


def compute_ote_zones(
    df: pd.DataFrame,
    swing_lookback: int = 50,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    swing_high = result["high"].rolling(swing_lookback, center=True).max()
    swing_low = result["low"].rolling(swing_lookback, center=True).min()

    impulse = swing_high - swing_low

    ote_618_high = swing_high - 0.382 * impulse
    ote_618_low = swing_high - 0.500 * impulse
    ote_705 = swing_high - 0.450 * impulse
    ote_786 = swing_high - 0.382 * impulse

    ote_bull_upper = swing_high - 0.382 * impulse
    ote_bull_lower = swing_high - 0.500 * impulse
    ote_bull_sweet = swing_high - 0.450 * impulse

    ote_bear_upper = swing_low + 0.500 * impulse
    ote_bear_lower = swing_low + 0.382 * impulse
    ote_bear_sweet = swing_low + 0.450 * impulse

    in_ote_bull = (result["close"] >= ote_bull_lower) & (result["close"] <= ote_bull_upper)
    in_ote_bear = (result["close"] >= ote_bear_lower) & (result["close"] <= ote_bear_upper)

    result["ote_swing_high"] = swing_high
    result["ote_swing_low"] = swing_low
    result["ote_impulse"] = impulse
    result["ote_bull_upper"] = ote_bull_upper
    result["ote_bull_lower"] = ote_bull_lower
    result["ote_bull_sweet"] = ote_bull_sweet
    result["ote_bear_upper"] = ote_bear_upper
    result["ote_bear_lower"] = ote_bear_lower
    result["ote_bear_sweet"] = ote_bear_sweet
    result["in_ote_bull"] = in_ote_bull
    result["in_ote_bear"] = in_ote_bear

    return result

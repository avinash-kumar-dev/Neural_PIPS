import pandas as pd
import numpy as np

from xauusd.structure.swing_points import detect_swing_points
from xauusd.structure.bos import detect_bos
from xauusd.structure.choch import detect_choch
from xauusd.structure.ema_filter import compute_ema_filter


LONG_ONLY = 1
SHORT_ONLY = -1
NO_BIAS = 0


def compute_layer1(
    df: pd.DataFrame,
    swing_lookback: int = 5,
    ema_fast: int = 50,
    ema_slow: int = 200,
) -> pd.DataFrame:
    result = detect_swing_points(df, lookback=swing_lookback)
    result = detect_bos(result)
    result = detect_choch(result)
    result = compute_ema_filter(result, fast_period=ema_fast, slow_period=ema_slow)

    n = len(result)
    bias = np.full(n, NO_BIAS, dtype=int)

    for i in range(n):
        smc_direction = 0
        if result["bos_bullish"].iloc[i]:
            smc_direction = 1
        elif result["bos_bearish"].iloc[i]:
            smc_direction = -1
        elif result["choch_bullish"].iloc[i]:
            smc_direction = 1
        elif result["choch_bearish"].iloc[i]:
            smc_direction = -1

        ema_dir = result["ema_bias"].iloc[i]

        if smc_direction == 0:
            bias[i] = NO_BIAS
        elif ema_dir == 0 or smc_direction == ema_dir:
            bias[i] = smc_direction
        else:
            bias[i] = NO_BIAS

    result["layer1_bias"] = bias
    return result

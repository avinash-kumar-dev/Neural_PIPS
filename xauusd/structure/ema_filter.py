import pandas as pd
import numpy as np


def compute_ema_filter(
    df: pd.DataFrame,
    fast_period: int = 50,
    slow_period: int = 200,
) -> pd.DataFrame:
    result = df.copy()
    result["ema_fast"] = result["close"].ewm(span=fast_period, adjust=False).mean()
    result["ema_slow"] = result["close"].ewm(span=slow_period, adjust=False).mean()

    ema_bias = np.zeros(len(result), dtype=int)
    ema_bias[result["ema_fast"] > result["ema_slow"]] = 1
    ema_bias[result["ema_fast"] < result["ema_slow"]] = -1
    result["ema_bias"] = ema_bias
    return result

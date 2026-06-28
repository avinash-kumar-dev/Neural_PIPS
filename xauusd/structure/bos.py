import pandas as pd
import numpy as np


def detect_bos(
    df: pd.DataFrame,
    swing_col_high: str = "swing_high",
    swing_col_low: str = "swing_low",
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)
    bos_bullish = np.zeros(n, dtype=bool)
    bos_bearish = np.zeros(n, dtype=bool)

    swing_high_vals = result[swing_col_high].values
    swing_low_vals = result[swing_col_low].values
    close_vals = result["close"].values

    last_sh = np.nan
    last_sl = np.nan

    for i in range(n):
        if not np.isnan(swing_high_vals[i]):
            last_sh = swing_high_vals[i]
        if not np.isnan(swing_low_vals[i]):
            last_sl = swing_low_vals[i]

        if not np.isnan(last_sh) and close_vals[i] > last_sh:
            bos_bullish[i] = True
            last_sh = np.nan

        if not np.isnan(last_sl) and close_vals[i] < last_sl:
            bos_bearish[i] = True
            last_sl = np.nan

    result["bos_bullish"] = bos_bullish
    result["bos_bearish"] = bos_bearish
    return result

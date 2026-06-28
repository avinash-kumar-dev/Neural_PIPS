import pandas as pd
import numpy as np


def compute_asian_range(
    df: pd.DataFrame,
    asian_start_hour: int = 0,
    asian_end_hour: int = 8,
) -> pd.DataFrame:
    result = df.copy()

    if "datetime" in result.columns:
        dt = pd.to_datetime(result["datetime"])
    else:
        dt = result.index
        if not isinstance(dt, pd.DatetimeIndex):
            return result

    hour = dt.hour
    in_asian = (hour >= asian_start_hour) & (hour < asian_end_hour)

    n = len(result)
    asian_high = np.full(n, np.nan)
    asian_low = np.full(n, np.nan)

    current_high = np.nan
    current_low = np.nan
    in_session = False

    for i in range(n):
        if in_asian.iloc[i]:
            if not in_session:
                in_session = True
                current_high = result["high"].iloc[i]
                current_low = result["low"].iloc[i]
            else:
                current_high = max(current_high, result["high"].iloc[i])
                current_low = min(current_low, result["low"].iloc[i])
        else:
            if in_session:
                in_session = False

        asian_high[i] = current_high
        asian_low[i] = current_low

    result["asian_high"] = asian_high
    result["asian_low"] = asian_low
    result["asian_range"] = asian_high - asian_low
    result["in_asian"] = in_asian.values

    return result

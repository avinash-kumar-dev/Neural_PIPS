import pandas as pd
import numpy as np


def compute_vwap(
    df: pd.DataFrame,
    session_col: str = None,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    typical = (result["high"] + result["low"] + result["close"]) / 3
    volume = result["volume"].replace(0, 1)

    if session_col and session_col in result.columns:
        sessions = result[session_col].values
        cum_tp_vol = np.zeros(n)
        cum_vol = np.zeros(n)
        current_session = None
        cum_tpv = 0.0
        cum_v = 0.0

        for i in range(n):
            if sessions[i] != current_session:
                current_session = sessions[i]
                cum_tpv = 0.0
                cum_v = 0.0
            cum_tpv += typical.iloc[i] * volume.iloc[i]
            cum_v += volume.iloc[i]
            cum_tp_vol[i] = cum_tpv
            cum_vol[i] = cum_v
    else:
        cum_tp_vol = (typical * volume).cumsum().values
        cum_vol = volume.cumsum().values

    cum_vol_safe = np.where(cum_vol == 0, 1, cum_vol)
    vwap = cum_tp_vol / cum_vol_safe

    deviations = np.abs(result["close"].values - vwap)
    rolling_std = pd.Series(deviations).rolling(20).mean().values

    result["vwap"] = vwap
    result["vwap_upper1"] = vwap + 1.0 * rolling_std
    result["vwap_lower1"] = vwap - 1.0 * rolling_std
    result["vwap_upper2"] = vwap + 2.0 * rolling_std
    result["vwap_lower2"] = vwap - 2.0 * rolling_std
    result["vwap_dist"] = (result["close"].values - vwap) / np.where(rolling_std == 0, 1, rolling_std)

    return result

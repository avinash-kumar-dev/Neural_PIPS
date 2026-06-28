import pandas as pd
import numpy as np

from xauusd.entry.m5_confirmation import compute_m5_confirmation


def align_m5_to_m15(
    df_m15: pd.DataFrame,
    df_m5: pd.DataFrame,
    lookahead_bars: int = 3,
) -> pd.DataFrame:
    result = df_m15.copy()

    if "datetime" not in df_m5.columns:
        if "time" in df_m5.columns:
            df_m5 = df_m5.copy()
            df_m5["datetime"] = pd.to_datetime(df_m5["time"], unit="s")

    df_m5 = compute_m5_confirmation(df_m5)

    m5_dt = pd.to_datetime(df_m5["datetime"])
    m5_bull = df_m5["m5_bull_confirm"].values
    m5_bear = df_m5["m5_bear_confirm"].values
    m5_rej_bull = df_m5["m5_bull_rejection"].values
    m5_rej_bear = df_m5["m5_bear_rejection"].values
    m5_eng_bull = df_m5["m5_bull_engulfing"].values
    m5_eng_bear = df_m5["m5_bear_engulfing"].values
    m5_mss_bull = df_m5["m5_mss_bullish"].values
    m5_mss_bear = df_m5["m5_mss_bearish"].values
    m5_vol_spike = df_m5["m5_volume_spike"].values

    m15_dt = pd.to_datetime(result["datetime"]).values.astype("datetime64[ns]")
    m15_bull = np.zeros(len(result), dtype=bool)
    m15_bear = np.zeros(len(result), dtype=bool)
    m15_has_mss = np.zeros(len(result), dtype=bool)
    m15_has_rejection = np.zeros(len(result), dtype=bool)
    m15_has_engulfing = np.zeros(len(result), dtype=bool)
    m15_has_volume = np.zeros(len(result), dtype=bool)

    m5_dt_values = m5_dt.values.astype("datetime64[ns]")

    for i in range(len(result)):
        t = m15_dt[i]
        t_end = t + np.timedelta64(15, "m")

        mask = (m5_dt_values >= t) & (m5_dt_values < t_end)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            continue

        start = int(idx[0])
        end = min(int(idx[-1]) + lookahead_bars + 1, len(df_m5))

        window_bull = m5_bull[start:end]
        window_bear = m5_bear[start:end]
        window_rej_bull = m5_rej_bull[start:end]
        window_rej_bear = m5_rej_bear[start:end]
        window_eng_bull = m5_eng_bull[start:end]
        window_eng_bear = m5_eng_bear[start:end]
        window_mss_bull = m5_mss_bull[start:end]
        window_mss_bear = m5_mss_bear[start:end]
        window_vol = m5_vol_spike[start:end]

        m15_bull[i] = np.any(window_bull)
        m15_bear[i] = np.any(window_bear)
        m15_has_mss[i] = np.any(window_mss_bull) or np.any(window_mss_bear)
        m15_has_rejection[i] = np.any(window_rej_bull) or np.any(window_rej_bear)
        m15_has_engulfing[i] = np.any(window_eng_bull) or np.any(window_eng_bear)
        m15_has_volume[i] = np.any(window_vol)

    result["m5_bull_confirm"] = m15_bull
    result["m5_bear_confirm"] = m15_bear
    result["m5_has_mss"] = m15_has_mss
    result["m5_has_rejection"] = m15_has_rejection
    result["m5_has_engulfing"] = m15_has_engulfing
    result["m5_has_volume"] = m15_has_volume

    return result

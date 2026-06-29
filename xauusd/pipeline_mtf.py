import pandas as pd
import numpy as np

from xauusd.cascade import compute_cascade_bias
from xauusd.pipeline import run_full_pipeline
from xauusd.entry.m5_alignment import align_m5_to_m15
from xauusd.entry.m1_entry import compute_m1_entry


def run_mtf_pipeline(
    df_m15: pd.DataFrame,
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame = None,
    df_m1: pd.DataFrame = None,
    min_confluence: float = 35.0,
) -> pd.DataFrame:
    print("  Running M15 base pipeline...")
    result = run_full_pipeline(df_m15, min_confluence=min_confluence)

    print("  Computing H4 + H1 cascade bias...")
    result = compute_cascade_bias(result, df_h4, df_h1)

    cascade = result["cascade_bias"].values
    layer1 = result["layer1_bias"].values

    final_long = np.zeros(len(result), dtype=bool)
    final_short = np.zeros(len(result), dtype=bool)

    long_mask = (result["long_entry"].values) & ((cascade == 1) | (layer1 == 1))
    short_mask = (result["short_entry"].values) & ((cascade == -1) | (layer1 == -1))

    final_long = long_mask
    final_short = short_mask

    result["long_entry"] = final_long
    result["short_entry"] = final_short

    if df_m5 is not None:
        print("  Aligning M5 confirmation...")
        result = align_m5_to_m15(result, df_m5)

    if df_m1 is not None:
        print("  Computing M1 entry signals...")
        m1_signals = compute_m1_entry(df_m1)
        m1_bull = m1_signals["m1_bull_entry"].values
        m1_bear = m1_signals["m1_bear_entry"].values

        m1_dt = pd.to_datetime(m1_signals["datetime"]).values.astype("datetime64[ns]")
        m15_dt = pd.to_datetime(result["datetime"]).values.astype("datetime64[ns]")

        m1_has_bull = np.zeros(len(result), dtype=bool)
        m1_has_bear = np.zeros(len(result), dtype=bool)

        m1_idx = 0
        for i in range(len(result)):
            t = m15_dt[i]
            t_end = t + np.timedelta64(15, "m")

            while m1_idx < len(m1_dt) - 1 and m1_dt[m1_idx + 1] <= t:
                m1_idx += 1

            start = m1_idx
            end = start
            while end < len(m1_dt) and m1_dt[end] < t_end:
                end += 1

            if start < end:
                m1_has_bull[i] = np.any(m1_bull[start:end])
                m1_has_bear[i] = np.any(m1_bear[start:end])

        result["m1_bull_entry"] = m1_has_bull
        result["m1_bear_entry"] = m1_has_bear

    result["long_entry"] = result["long_entry"].values
    result["short_entry"] = result["short_entry"].values

    return result

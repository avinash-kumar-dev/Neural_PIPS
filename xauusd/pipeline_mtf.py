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

    long_mask = (result["long_entry"].values) & (cascade == 1) & (layer1 == 1)
    short_mask = (result["short_entry"].values) & (cascade == -1) & (layer1 == -1)

    result["long_entry"] = long_mask
    result["short_entry"] = short_mask

    if df_m5 is not None:
        print("  Aligning M5 confirmation...")
        result = align_m5_to_m15(result, df_m5)

        m5_bull = result.get("m5_bull_confirm", pd.Series(False, index=result.index)).values
        m5_bear = result.get("m5_bear_confirm", pd.Series(False, index=result.index)).values

        result["long_entry"] = result["long_entry"].values & m5_bull
        result["short_entry"] = result["short_entry"].values & m5_bear

    if df_m1 is not None:
        print("  Computing M1 entry signals...")
        m1_signals = compute_m1_entry(df_m1)
        m1_bull = m1_signals["m1_bull_entry"].values
        m1_bear = m1_signals["m1_bear_entry"].values
        m1_bull_sl = m1_signals["m1_bull_sl"].values
        m1_bear_sl = m1_signals["m1_bear_sl"].values

        m1_dt = pd.to_datetime(m1_signals["datetime"]).values.astype("datetime64[ns]")
        m15_dt = pd.to_datetime(result["datetime"]).values.astype("datetime64[ns]")

        m1_has_bull = np.zeros(len(result), dtype=bool)
        m1_has_bear = np.zeros(len(result), dtype=bool)
        m1_best_bull_sl = np.full(len(result), np.nan)
        m1_best_bear_sl = np.full(len(result), np.nan)

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
                bull_window = m1_bull[start:end]
                bear_window = m1_bear[start:end]
                bull_sl_window = m1_bull_sl[start:end]
                bear_sl_window = m1_bear_sl[start:end]

                m1_has_bull[i] = np.any(bull_window)
                m1_has_bear[i] = np.any(bear_window)

                if m1_has_bull[i]:
                    valid_bull_sl = bull_sl_window[~np.isnan(bull_sl_window)]
                    if len(valid_bull_sl) > 0:
                        m1_best_bull_sl[i] = np.max(valid_bull_sl)

                if m1_has_bear[i]:
                    valid_bear_sl = bear_sl_window[~np.isnan(bear_sl_window)]
                    if len(valid_bear_sl) > 0:
                        m1_best_bear_sl[i] = np.min(valid_bear_sl)

        result["m1_bull_entry"] = m1_has_bull
        result["m1_bear_entry"] = m1_has_bear

        long_entry = result["long_entry"].values
        short_entry = result["short_entry"].values
        long_sl = result["long_sl"].values.copy()
        short_sl = result["short_sl"].values.copy()

        for i in range(len(result)):
            if long_entry[i] and not np.isnan(m1_best_bull_sl[i]):
                long_sl[i] = m1_best_bull_sl[i]
            if short_entry[i] and not np.isnan(m1_best_bear_sl[i]):
                short_sl[i] = m1_best_bear_sl[i]

        result["long_sl"] = long_sl
        result["short_sl"] = short_sl

    return result

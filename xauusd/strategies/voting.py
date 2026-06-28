import pandas as pd
import numpy as np
from xauusd.strategies.trend_structure import run_trend_strategy
from xauusd.strategies.breakout_retest import compute_breakout_retest
from xauusd.strategies.liquidity_sweep import compute_liquidity_sweep


def compute_voting_signals(
    df: pd.DataFrame,
    min_agreement: int = 2,
) -> pd.DataFrame:
    trend = run_trend_strategy(df)
    breakout = compute_breakout_retest(df)
    sweep = compute_liquidity_sweep(df)

    result = df.copy()
    n = len(result)

    long_votes = np.zeros(n, dtype=int)
    short_votes = np.zeros(n, dtype=int)
    long_sl = np.full(n, np.nan)
    short_sl = np.full(n, np.nan)
    long_triggers = np.array([""] * n, dtype=object)
    short_triggers = np.array([""] * n, dtype=object)

    for i in range(n):
        if trend["long_entry"].iloc[i] and trend["layer3_confirmed"].iloc[i]:
            long_votes[i] += 1
            long_sl[i] = trend["long_sl"].iloc[i]
            long_triggers[i] = f"T:{trend['long_trigger'].iloc[i]}"

        if trend["short_entry"].iloc[i] and trend["layer3_confirmed"].iloc[i]:
            short_votes[i] += 1
            short_sl[i] = trend["short_sl"].iloc[i]
            short_triggers[i] = f"T:{trend['short_trigger'].iloc[i]}"

        if breakout["retest_bull"].iloc[i]:
            long_votes[i] += 1
            if pd.isna(long_sl[i]) or (not pd.isna(breakout["retest_bull_sl"].iloc[i]) and breakout["retest_bull_sl"].iloc[i] > long_sl[i]):
                long_sl[i] = breakout["retest_bull_sl"].iloc[i]
            long_triggers[i] = f"{long_triggers[i]}+BR" if long_triggers[i] else "BR"

        if breakout["retest_bear"].iloc[i]:
            short_votes[i] += 1
            if pd.isna(short_sl[i]) or (not pd.isna(breakout["retest_bear_sl"].iloc[i]) and breakout["retest_bear_sl"].iloc[i] < short_sl[i]):
                short_sl[i] = breakout["retest_bear_sl"].iloc[i]
            short_triggers[i] = f"{short_triggers[i]}+BR" if short_triggers[i] else "BR"

        if sweep["sweep_high_signal"].iloc[i]:
            long_votes[i] += 1
            if pd.isna(long_sl[i]) or (not pd.isna(sweep["sweep_high_sl"].iloc[i]) and sweep["sweep_high_sl"].iloc[i] > long_sl[i]):
                long_sl[i] = sweep["sweep_high_sl"].iloc[i]
            long_triggers[i] = f"{long_triggers[i]}+LS" if long_triggers[i] else "LS"

        if sweep["sweep_low_signal"].iloc[i]:
            short_votes[i] += 1
            if pd.isna(short_sl[i]) or (not pd.isna(sweep["sweep_low_sl"].iloc[i]) and sweep["sweep_low_sl"].iloc[i] < short_sl[i]):
                short_sl[i] = sweep["sweep_low_sl"].iloc[i]
            short_triggers[i] = f"{short_triggers[i]}+LS" if short_triggers[i] else "LS"

    long_confirmed = long_votes >= min_agreement
    short_confirmed = short_votes >= min_agreement

    long_confirmed = long_confirmed & ~short_confirmed
    short_confirmed = short_confirmed & ~long_confirmed

    result["long_entry"] = long_confirmed
    result["short_entry"] = short_confirmed
    result["long_sl"] = long_sl
    result["short_sl"] = short_sl
    result["long_trigger"] = long_triggers
    result["short_trigger"] = short_triggers
    result["long_votes"] = long_votes
    result["short_votes"] = short_votes

    trend_layer3 = trend["layer3_confirmed"].values if "layer3_confirmed" in trend.columns else np.ones(n, dtype=bool)
    result["layer3_confirmed"] = trend_layer3

    return result

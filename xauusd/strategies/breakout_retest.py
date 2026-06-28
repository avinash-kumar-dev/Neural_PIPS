import pandas as pd
import numpy as np
from xauusd.risk.layer4 import compute_atr, PIP_VALUE


def compute_breakout_retest(
    df: pd.DataFrame,
    session_lookback: int = 12,
    retest_window: int = 5,
    atr_period: int = 14,
    min_body_ratio: float = 0.6,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    atr = compute_atr(result, period=atr_period)
    result["atr"] = atr

    session_high = np.full(n, np.nan)
    session_low = np.full(n, np.nan)

    for i in range(session_lookback, n):
        session_high[i] = result["high"].iloc[i - session_lookback : i].max()
        session_low[i] = result["low"].iloc[i - session_lookback : i].min()

    result["session_high"] = session_high
    result["session_low"] = session_low

    breakout_bull = np.zeros(n, dtype=bool)
    breakout_bear = np.zeros(n, dtype=bool)
    retest_bull = np.zeros(n, dtype=bool)
    retest_bear = np.zeros(n, dtype=bool)
    bull_sl = np.full(n, np.nan)
    bear_sl = np.full(n, np.nan)

    broken_high = np.full(n, np.nan)
    broken_low = np.full(n, np.nan)

    for i in range(session_lookback, n):
        if pd.isna(session_high[i]) or pd.isna(session_low[i]):
            continue

        candle_range = result["high"].iloc[i] - result["low"].iloc[i]
        body = abs(result["close"].iloc[i] - result["open"].iloc[i])
        body_ratio = body / candle_range if candle_range > 0 else 0

        if result["close"].iloc[i] > session_high[i] and body_ratio >= min_body_ratio:
            breakout_bull[i] = True
            broken_high[i] = session_high[i]

        if result["close"].iloc[i] < session_low[i] and body_ratio >= min_body_ratio:
            breakout_bear[i] = True
            broken_low[i] = session_low[i]

        for j in range(max(session_lookback, i - retest_window), i):
            if breakout_bull[j] and not pd.isna(broken_high[j]):
                if result["low"].iloc[i] <= broken_high[j] * 1.001:
                    if result["close"].iloc[i] > result["open"].iloc[i]:
                        retest_bull[i] = True
                        bull_sl[i] = broken_high[j] - atr.iloc[i] * 0.5 if not pd.isna(atr.iloc[i]) else broken_high[j] - 5 * PIP_VALUE
                        break

            if breakout_bear[j] and not pd.isna(broken_low[j]):
                if result["high"].iloc[i] >= broken_low[j] * 0.999:
                    if result["close"].iloc[i] < result["open"].iloc[i]:
                        retest_bear[i] = True
                        bear_sl[i] = broken_low[j] + atr.iloc[i] * 0.5 if not pd.isna(atr.iloc[i]) else broken_low[j] + 5 * PIP_VALUE
                        break

    result["breakout_bull"] = breakout_bull
    result["breakout_bear"] = breakout_bear
    result["retest_bull"] = retest_bull
    result["retest_bear"] = retest_bear
    result["retest_bull_sl"] = bull_sl
    result["retest_bear_sl"] = bear_sl
    return result

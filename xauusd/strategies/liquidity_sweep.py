import pandas as pd
import numpy as np
from xauusd.risk.layer4 import compute_atr, PIP_VALUE


def compute_liquidity_sweep(
    df: pd.DataFrame,
    swing_lookback: int = 5,
    sweep_buffer: float = 0.002,
    atr_period: int = 14,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values
    opens = result["open"].values

    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)

    for i in range(swing_lookback, n):
        left_highs = highs[i - swing_lookback : i]
        right_highs = highs[i + 1 : min(i + 1 + swing_lookback, n)]
        if len(right_highs) > 0 and highs[i] > np.max(left_highs) and highs[i] > np.max(right_highs):
            swing_high[i] = highs[i]

        left_lows = lows[i - swing_lookback : i]
        right_lows = lows[i + 1 : min(i + 1 + swing_lookback, n)]
        if len(right_lows) > 0 and lows[i] < np.min(left_lows) and lows[i] < np.min(right_lows):
            swing_low[i] = lows[i]

    result["sweep_swing_high"] = swing_high
    result["sweep_swing_low"] = swing_low

    sweep_high_bull = np.zeros(n, dtype=bool)
    sweep_low_bear = np.zeros(n, dtype=bool)
    sweep_high_sl = np.full(n, np.nan)
    sweep_low_sl = np.full(n, np.nan)

    active_highs = []
    active_lows = []

    for i in range(n):
        if not np.isnan(swing_high[i]):
            active_highs.append((i, swing_high[i]))
        if not np.isnan(swing_low[i]):
            active_lows.append((i, swing_low[i]))

        active_highs = [(idx, lvl) for idx, lvl in active_highs if i - idx <= 30]
        active_lows = [(idx, lvl) for idx, lvl in active_lows if i - idx <= 30]

        for idx, lvl in active_highs:
            if highs[i] > lvl * (1 + sweep_buffer) and closes[i] < lvl:
                if closes[i] < opens[i]:
                    sweep_high_bull[i] = True
                    sweep_high_sl[i] = lvl
                    break

        for idx, lvl in active_lows:
            if lows[i] < lvl * (1 - sweep_buffer) and closes[i] > lvl:
                if closes[i] > opens[i]:
                    sweep_low_bear[i] = True
                    sweep_low_sl[i] = lvl
                    break

    result["sweep_high_signal"] = sweep_high_bull
    result["sweep_low_signal"] = sweep_low_bear
    result["sweep_high_sl"] = sweep_high_sl
    result["sweep_low_sl"] = sweep_low_sl
    return result

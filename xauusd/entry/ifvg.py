import pandas as pd
import numpy as np


def detect_ifvg(
    df: pd.DataFrame,
    min_size_atr: float = 0.5,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    tr = np.maximum(
        result["high"] - result["low"],
        np.maximum(
            abs(result["high"] - result["close"].shift(1)),
            abs(result["low"] - result["close"].shift(1)),
        ),
    )
    atr = pd.Series(tr).rolling(14).mean().values
    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values

    fvg_bull_h = np.full(n, np.nan)
    fvg_bull_l = np.full(n, np.nan)
    fvg_bear_h = np.full(n, np.nan)
    fvg_bear_l = np.full(n, np.nan)

    for i in range(2, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        if lows[i] > highs[i - 2]:
            gap = lows[i] - highs[i - 2]
            if gap >= min_size_atr * atr[i]:
                fvg_bull_h[i] = lows[i]
                fvg_bull_l[i] = highs[i - 2]
        if highs[i] < lows[i - 2]:
            gap = lows[i - 2] - highs[i]
            if gap >= min_size_atr * atr[i]:
                fvg_bear_h[i] = lows[i - 2]
                fvg_bear_l[i] = highs[i]

    ifvg_bull_high = np.full(n, np.nan)
    ifvg_bull_low = np.full(n, np.nan)
    ifvg_bear_high = np.full(n, np.nan)
    ifvg_bear_low = np.full(n, np.nan)

    bull_q = []
    bear_q = []
    max_age = 30

    for i in range(n):
        while bull_q and bull_q[0][0] < i - max_age:
            bull_q.pop(0)
        while bear_q and bear_q[0][0] < i - max_age:
            bear_q.pop(0)

        if not np.isnan(fvg_bull_h[i]):
            bull_q.append((i, fvg_bull_h[i], fvg_bull_l[i]))
        if not np.isnan(fvg_bear_h[i]):
            bear_q.append((i, fvg_bear_h[i], fvg_bear_l[i]))

        for idx, h, l in bull_q:
            if closes[i] <= l:
                ifvg_bull_high[i] = h
                ifvg_bull_low[i] = l
                break

        for idx, h, l in bear_q:
            if closes[i] >= h:
                ifvg_bear_high[i] = h
                ifvg_bear_low[i] = l
                break

    result["ifvg_bull_high"] = ifvg_bull_high
    result["ifvg_bull_low"] = ifvg_bull_low
    result["ifvg_bear_high"] = ifvg_bear_high
    result["ifvg_bear_low"] = ifvg_bear_low
    result["has_ifvg_bull"] = ~np.isnan(ifvg_bull_high)
    result["has_ifvg_bear"] = ~np.isnan(ifvg_bear_high)
    return result

import pandas as pd
import numpy as np


def detect_fvg(
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

    result["fvg_bull_high"] = fvg_bull_h
    result["fvg_bull_low"] = fvg_bull_l
    result["fvg_bear_high"] = fvg_bear_h
    result["fvg_bear_low"] = fvg_bear_l
    return result


def check_fvg_retest(df: pd.DataFrame, max_age: int = 30) -> pd.DataFrame:
    result = df.copy()
    n = len(result)
    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values
    opens = result["open"].values

    fvg_bull_h_vals = result["fvg_bull_high"].values if "fvg_bull_high" in result.columns else np.full(n, np.nan)
    fvg_bull_l_vals = result["fvg_bull_low"].values if "fvg_bull_low" in result.columns else np.full(n, np.nan)
    fvg_bear_h_vals = result["fvg_bear_high"].values if "fvg_bear_high" in result.columns else np.full(n, np.nan)
    fvg_bear_l_vals = result["fvg_bear_low"].values if "fvg_bear_low" in result.columns else np.full(n, np.nan)

    fvg_bull_hold = np.zeros(n, dtype=bool)
    fvg_bear_hold = np.zeros(n, dtype=bool)
    fvg_bull_fill = np.zeros(n, dtype=bool)
    fvg_bear_fill = np.zeros(n, dtype=bool)
    fvg_bull_sl = np.full(n, np.nan)
    fvg_bear_sl = np.full(n, np.nan)

    bull_q = []
    bear_q = []

    for i in range(n):
        while bull_q and bull_q[0][0] < i - max_age:
            bull_q.pop(0)
        while bear_q and bear_q[0][0] < i - max_age:
            bear_q.pop(0)

        if not np.isnan(fvg_bull_h_vals[i]):
            bull_q.append((i, fvg_bull_h_vals[i], fvg_bull_l_vals[i]))
        if not np.isnan(fvg_bear_h_vals[i]):
            bear_q.append((i, fvg_bear_h_vals[i], fvg_bear_l_vals[i]))

        for idx, h, l in bull_q:
            if lows[i] <= h and lows[i] >= l and closes[i] > opens[i]:
                fvg_bull_hold[i] = True
                fvg_bull_sl[i] = l
                break
            if lows[i] <= l:
                fvg_bull_fill[i] = True
                fvg_bull_sl[i] = l
                break

        for idx, h, l in bear_q:
            if highs[i] >= l and highs[i] <= h and closes[i] < opens[i]:
                fvg_bear_hold[i] = True
                fvg_bear_sl[i] = h
                break
            if highs[i] >= h:
                fvg_bear_fill[i] = True
                fvg_bear_sl[i] = h
                break

    result["fvg_bull_hold"] = fvg_bull_hold
    result["fvg_bear_hold"] = fvg_bear_hold
    result["fvg_bull_fill"] = fvg_bull_fill
    result["fvg_bear_fill"] = fvg_bear_fill
    result["fvg_bull_sl"] = fvg_bull_sl
    result["fvg_bear_sl"] = fvg_bear_sl
    return result

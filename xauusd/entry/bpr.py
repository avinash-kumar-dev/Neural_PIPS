import pandas as pd
import numpy as np


def detect_bpr(
    df: pd.DataFrame,
    min_overlap_atr: float = 0.3,
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

    bull_fvg_h = np.full(n, np.nan)
    bull_fvg_l = np.full(n, np.nan)
    bear_fvg_h = np.full(n, np.nan)
    bear_fvg_l = np.full(n, np.nan)

    for i in range(2, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        if lows[i] > highs[i - 2]:
            gap = lows[i] - highs[i - 2]
            if gap >= 0.3 * atr[i]:
                bull_fvg_h[i] = lows[i]
                bull_fvg_l[i] = highs[i - 2]
        if highs[i] < lows[i - 2]:
            gap = lows[i - 2] - highs[i]
            if gap >= 0.3 * atr[i]:
                bear_fvg_h[i] = lows[i - 2]
                bear_fvg_l[i] = highs[i]

    bpr_high = np.full(n, np.nan)
    bpr_low = np.full(n, np.nan)
    bpr_mid = np.full(n, np.nan)

    bull_q = []
    bear_q = []
    max_age = 30

    for i in range(n):
        while bull_q and bull_q[0][0] < i - max_age:
            bull_q.pop(0)
        while bear_q and bear_q[0][0] < i - max_age:
            bear_q.pop(0)

        if not np.isnan(bull_fvg_h[i]):
            bull_q.append((i, bull_fvg_h[i], bull_fvg_l[i]))
        if not np.isnan(bear_fvg_h[i]):
            bear_q.append((i, bear_fvg_h[i], bear_fvg_l[i]))

        for b_idx, b_h, b_l in bull_q:
            for r_idx, r_h, r_l in bear_q:
                overlap_high = min(b_h, r_h)
                overlap_low = max(b_l, r_l)
                if overlap_high > overlap_low:
                    overlap_size = overlap_high - overlap_low
                    if not np.isnan(atr[i]) and atr[i] > 0 and overlap_size >= min_overlap_atr * atr[i]:
                        bpr_high[i] = overlap_high
                        bpr_low[i] = overlap_low
                        bpr_mid[i] = (overlap_high + overlap_low) / 2
                        break
            if not np.isnan(bpr_high[i]):
                break

    result["bpr_high"] = bpr_high
    result["bpr_low"] = bpr_low
    result["bpr_mid"] = bpr_mid
    result["has_bpr"] = ~np.isnan(bpr_high)
    return result

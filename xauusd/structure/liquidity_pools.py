import pandas as pd
import numpy as np


def detect_liquidity_pools(
    df: pd.DataFrame,
    lookback: int = 20,
    tolerance_atr: float = 0.5,
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

    eqh_level = np.full(n, np.nan)
    eql_level = np.full(n, np.nan)
    eqh_count = np.zeros(n, dtype=int)
    eql_count = np.zeros(n, dtype=int)

    for i in range(lookback, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        tol = tolerance_atr * atr[i]
        window_h = highs[max(0, i - lookback):i + 1]
        window_l = lows[max(0, i - lookback):i + 1]

        for j in range(len(window_h)):
            count = np.sum(np.abs(window_h - window_h[j]) < tol)
            if count >= 3:
                eqh_level[i] = window_h[j]
                eqh_count[i] = count
                break

        for j in range(len(window_l)):
            count = np.sum(np.abs(window_l - window_l[j]) < tol)
            if count >= 3:
                eql_level[i] = window_l[j]
                eql_count[i] = count
                break

    result["eqh_level"] = eqh_level
    result["eql_level"] = eql_level
    result["eqh_count"] = eqh_count
    result["eql_count"] = eql_count
    result["has_eqh"] = ~np.isnan(eqh_level)
    result["has_eql"] = ~np.isnan(eql_level)
    return result

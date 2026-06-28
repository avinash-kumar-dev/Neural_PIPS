import pandas as pd
import numpy as np


def detect_breaker_blocks(
    df: pd.DataFrame,
    lookback: int = 30,
    displacement_mult: float = 1.3,
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
    opens = result["open"].values
    body = np.abs(closes - opens)

    breaker_bull_high = np.full(n, np.nan)
    breaker_bull_low = np.full(n, np.nan)
    breaker_bear_high = np.full(n, np.nan)
    breaker_bear_low = np.full(n, np.nan)

    pending_bull = []
    pending_bear = []

    for i in range(2, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        avg_body = np.mean(body[max(0, i - 10):i]) if i >= 2 else body[i]

        if avg_body > 0 and closes[i] > opens[i] and body[i] > displacement_mult * avg_body:
            if i >= 2 and closes[i - 1] < opens[i - 1]:
                pending_bull.append({
                    "high": highs[i - 1],
                    "low": lows[i - 1],
                    "broken": False,
                    "bar": i,
                })

        if avg_body > 0 and closes[i] < opens[i] and body[i] > displacement_mult * avg_body:
            if i >= 2 and closes[i - 1] > opens[i - 1]:
                pending_bear.append({
                    "high": highs[i - 1],
                    "low": lows[i - 1],
                    "broken": False,
                    "bar": i,
                })

        for ob in pending_bull:
            if not ob["broken"]:
                if closes[i] < ob["low"]:
                    ob["broken"] = True
            elif i - ob["bar"] <= lookback:
                if lows[i] <= ob["high"] and closes[i] >= ob["low"]:
                    breaker_bull_high[i] = ob["high"]
                    breaker_bull_low[i] = ob["low"]
                    break

        for ob in pending_bear:
            if not ob["broken"]:
                if closes[i] > ob["high"]:
                    ob["broken"] = True
            elif i - ob["bar"] <= lookback:
                if highs[i] >= ob["low"] and closes[i] <= ob["high"]:
                    breaker_bear_high[i] = ob["high"]
                    breaker_bear_low[i] = ob["low"]
                    break

        pending_bull = [ob for ob in pending_bull if i - ob["bar"] <= lookback]
        pending_bear = [ob for ob in pending_bear if i - ob["bar"] <= lookback]

    result["breaker_bull_high"] = breaker_bull_high
    result["breaker_bull_low"] = breaker_bull_low
    result["breaker_bear_high"] = breaker_bear_high
    result["breaker_bear_low"] = breaker_bear_low
    result["has_breaker_bull"] = ~np.isnan(breaker_bull_high)
    result["has_breaker_bear"] = ~np.isnan(breaker_bear_high)
    return result

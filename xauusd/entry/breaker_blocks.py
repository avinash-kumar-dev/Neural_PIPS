import pandas as pd
import numpy as np


def detect_breaker_blocks(
    df: pd.DataFrame,
    lookback: int = 20,
    displacement_mult: float = 1.5,
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

    ob_queue = []

    for i in range(lookback + 2, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        if closes[i] > opens[i] and body[i] > displacement_mult * np.mean(body[max(0, i - 10):i]):
            if i >= 2:
                prev_close = closes[i - 1]
                prev_open = opens[i - 1]
                if prev_close < prev_open:
                    ob_queue.append({
                        "idx": i - 1,
                        "high": highs[i - 1],
                        "low": lows[i - 1],
                        "direction": "bull",
                        "broken": False,
                    })

        if closes[i] < opens[i] and body[i] > displacement_mult * np.mean(body[max(0, i - 10):i]):
            if i >= 2:
                prev_close = closes[i - 1]
                prev_open = opens[i - 1]
                if prev_close > prev_open:
                    ob_queue.append({
                        "idx": i - 1,
                        "high": highs[i - 1],
                        "low": lows[i - 1],
                        "direction": "bear",
                        "broken": False,
                    })

        for ob in ob_queue:
            if ob["broken"]:
                continue
            if ob["idx"] < i - 50:
                ob["broken"] = True
                continue

            if ob["direction"] == "bull" and closes[i] < ob["low"]:
                ob["broken"] = True
                if lows[i] <= ob["high"] and lows[i] >= ob["low"]:
                    breaker_bull_high[i] = ob["high"]
                    breaker_bull_low[i] = ob["low"]

            elif ob["direction"] == "bear" and closes[i] > ob["high"]:
                ob["broken"] = True
                if highs[i] >= ob["low"] and highs[i] <= ob["high"]:
                    breaker_bear_high[i] = ob["high"]
                    breaker_bear_low[i] = ob["low"]

    result["breaker_bull_high"] = breaker_bull_high
    result["breaker_bull_low"] = breaker_bull_low
    result["breaker_bear_high"] = breaker_bear_high
    result["breaker_bear_low"] = breaker_bear_low
    result["has_breaker_bull"] = ~np.isnan(breaker_bull_high)
    result["has_breaker_bear"] = ~np.isnan(breaker_bear_high)
    return result

import pandas as pd
import numpy as np


def detect_rejection_blocks(
    df: pd.DataFrame,
    min_wick_ratio: float = 2.0,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    highs = result["high"].values
    lows = result["low"].values
    opens = result["open"].values
    closes = result["close"].values

    body = np.abs(closes - opens)
    upper_wick = highs - np.maximum(opens, closes)
    lower_wick = np.minimum(opens, closes) - lows
    total_range = highs - lows

    rej_bull_high = np.full(n, np.nan)
    rej_bull_low = np.full(n, np.nan)
    rej_bear_high = np.full(n, np.nan)
    rej_bear_low = np.full(n, np.nan)

    for i in range(1, n):
        if total_range[i] == 0:
            continue

        if lower_wick[i] > min_wick_ratio * body[i] and lower_wick[i] > upper_wick[i]:
            rej_bull_high[i] = closes[i] if closes[i] > opens[i] else opens[i]
            rej_bull_low[i] = lows[i]

        if upper_wick[i] > min_wick_ratio * body[i] and upper_wick[i] > lower_wick[i]:
            rej_bear_low[i] = closes[i] if closes[i] < opens[i] else opens[i]
            rej_bear_high[i] = highs[i]

    result["rej_bull_high"] = rej_bull_high
    result["rej_bull_low"] = rej_bull_low
    result["rej_bear_high"] = rej_bear_high
    result["rej_bear_low"] = rej_bear_low
    result["has_rej_bull"] = ~np.isnan(rej_bull_high)
    result["has_rej_bear"] = ~np.isnan(rej_bear_high)
    return result

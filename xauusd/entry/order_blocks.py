import pandas as pd
import numpy as np


def detect_order_blocks(
    df: pd.DataFrame,
    atr_period: int = 14,
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
    atr = pd.Series(tr).rolling(atr_period).mean().values
    body = (result["close"] - result["open"]).values
    candle_range = (result["high"] - result["low"]).values
    opens = result["open"].values
    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values

    ob_bull_high = np.full(n, np.nan)
    ob_bull_low = np.full(n, np.nan)
    ob_bear_high = np.full(n, np.nan)
    ob_bear_low = np.full(n, np.nan)

    for i in range(atr_period + 2, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        disp = candle_range[i] / atr[i]
        if disp >= displacement_mult:
            if body[i] > 0 and i > 0:
                ob_bull_high[i] = highs[i - 1]
                ob_bull_low[i] = lows[i - 1]
            elif body[i] < 0 and i > 0:
                ob_bear_high[i] = highs[i - 1]
                ob_bear_low[i] = lows[i - 1]

    result["ob_bull_high"] = ob_bull_high
    result["ob_bull_low"] = ob_bull_low
    result["ob_bear_high"] = ob_bear_high
    result["ob_bear_low"] = ob_bear_low
    return result


def check_ob_retest(df: pd.DataFrame, max_age: int = 20) -> pd.DataFrame:
    result = df.copy()
    n = len(result)
    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values
    opens = result["open"].values
    ob_bull_h = result["ob_bull_high"].values if "ob_bull_high" in result.columns else np.full(n, np.nan)
    ob_bull_l = result["ob_bull_low"].values if "ob_bull_low" in result.columns else np.full(n, np.nan)
    ob_bear_h = result["ob_bear_high"].values if "ob_bear_high" in result.columns else np.full(n, np.nan)
    ob_bear_l = result["ob_bear_low"].values if "ob_bear_low" in result.columns else np.full(n, np.nan)

    ob_bull_signal = np.zeros(n, dtype=bool)
    ob_bear_signal = np.zeros(n, dtype=bool)
    ob_bull_sl = np.full(n, np.nan)
    ob_bear_sl = np.full(n, np.nan)

    bull_queue = []
    bear_queue = []

    for i in range(n):
        while bull_queue and bull_queue[0][0] < i - max_age:
            bull_queue.pop(0)
        while bear_queue and bear_queue[0][0] < i - max_age:
            bear_queue.pop(0)

        if not np.isnan(ob_bull_h[i]):
            bull_queue.append((i, ob_bull_h[i], ob_bull_l[i]))
        if not np.isnan(ob_bear_h[i]):
            bear_queue.append((i, ob_bear_h[i], ob_bear_l[i]))

        for idx, h, l in bull_queue:
            if l <= lows[i] <= h and closes[i] > opens[i]:
                ob_bull_signal[i] = True
                ob_bull_sl[i] = l
                break

        for idx, h, l in bear_queue:
            if l <= highs[i] <= h and closes[i] < opens[i]:
                ob_bear_signal[i] = True
                ob_bear_sl[i] = h
                break

    result["ob_bull_signal"] = ob_bull_signal
    result["ob_bear_signal"] = ob_bear_signal
    result["ob_bull_sl"] = ob_bull_sl
    result["ob_bear_sl"] = ob_bear_sl
    return result

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrderBlock:
    index: int
    direction: int  # 1 = bullish OB, -1 = bearish OB
    high: float
    low: float
    displacement_index: int
    displacement_size: float


def detect_order_blocks(
    df: pd.DataFrame,
    atr_period: int = 14,
    displacement_mult: float = 1.5,
    max_age: int = 20,
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
    atr = pd.Series(tr).rolling(atr_period).mean()

    ob_bullish = np.full(n, np.nan)
    ob_bearish = np.full(n, np.nan)

    for i in range(atr_period + 2, n):
        candle_range = result["high"].iloc[i] - result["low"].iloc[i]
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0:
            continue

        displacement = candle_range / current_atr

        if displacement >= displacement_mult:
            body = result["close"].iloc[i] - result["open"].iloc[i]
            if body > 0:
                ob_bullish[i] = result["low"].iloc[i]
                if i - 1 >= 0:
                    result.loc[result.index[i], "ob_bullish_high"] = result["high"].iloc[i - 1]
                    result.loc[result.index[i], "ob_bullish_low"] = result["low"].iloc[i - 1]
            elif body < 0:
                ob_bearish[i] = result["high"].iloc[i]
                if i - 1 >= 0:
                    result.loc[result.index[i], "ob_bearish_high"] = result["high"].iloc[i - 1]
                    result.loc[result.index[i], "ob_bearish_low"] = result["low"].iloc[i - 1]

    result["ob_bullish_price"] = ob_bullish
    result["ob_bearish_price"] = ob_bearish
    return result


def check_ob_retest(
    df: pd.DataFrame,
    max_age: int = 20,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    ob_bull_signal = np.zeros(n, dtype=bool)
    ob_bear_signal = np.zeros(n, dtype=bool)
    ob_bull_sl = np.full(n, np.nan)
    ob_bear_sl = np.full(n, np.nan)

    active_bull_obs = []
    active_bear_obs = []

    for i in range(n):
        active_bull_obs = [(idx, h, l) for idx, h, l in active_bull_obs if i - idx <= max_age]
        active_bear_obs = [(idx, h, l) for idx, h, l in active_bear_obs if i - idx <= max_age]

        if not pd.isna(result.get("ob_bullish_price", pd.Series(dtype=float)).iloc[i] if "ob_bullish_price" in result.columns else np.nan):
            active_bull_obs.append((i, result["high"].iloc[i], result["low"].iloc[i]))

        if not pd.isna(result.get("ob_bearish_price", pd.Series(dtype=float)).iloc[i] if "ob_bearish_price" in result.columns else np.nan):
            active_bear_obs.append((i, result["high"].iloc[i], result["low"].iloc[i]))

        for idx, h, l in active_bull_obs:
            if l <= result["low"].iloc[i] <= h and result["close"].iloc[i] > result["open"].iloc[i]:
                ob_bull_signal[i] = True
                ob_bull_sl[i] = l
                break

        for idx, h, l in active_bear_obs:
            if l <= result["high"].iloc[i] <= h and result["close"].iloc[i] < result["open"].iloc[i]:
                ob_bear_signal[i] = True
                ob_bear_sl[i] = h
                break

    result["ob_bull_signal"] = ob_bull_signal
    result["ob_bear_signal"] = ob_bear_signal
    result["ob_bull_sl"] = ob_bull_sl
    result["ob_bear_sl"] = ob_bear_sl
    return result

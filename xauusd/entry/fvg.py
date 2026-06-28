import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class FVGSignal:
    index: int
    direction: int  # 1 = bullish, -1 = bearish
    zone_high: float
    zone_low: float
    signal_type: str  # "hold" or "fill"


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
    atr = pd.Series(tr).rolling(14).mean()

    fvg_bull_zone_high = np.full(n, np.nan)
    fvg_bull_zone_low = np.full(n, np.nan)
    fvg_bear_zone_high = np.full(n, np.nan)
    fvg_bear_zone_low = np.full(n, np.nan)

    for i in range(2, n):
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0:
            continue

        if result["low"].iloc[i] > result["high"].iloc[i - 2]:
            gap_size = result["low"].iloc[i] - result["high"].iloc[i - 2]
            if gap_size >= min_size_atr * current_atr:
                fvg_bull_zone_high[i] = result["low"].iloc[i]
                fvg_bull_zone_low[i] = result["high"].iloc[i - 2]

        if result["high"].iloc[i] < result["low"].iloc[i - 2]:
            gap_size = result["low"].iloc[i - 2] - result["high"].iloc[i]
            if gap_size >= min_size_atr * current_atr:
                fvg_bear_zone_high[i] = result["low"].iloc[i - 2]
                fvg_bear_zone_low[i] = result["high"].iloc[i]

    result["fvg_bull_high"] = fvg_bull_zone_high
    result["fvg_bull_low"] = fvg_bull_zone_low
    result["fvg_bear_high"] = fvg_bear_zone_high
    result["fvg_bear_low"] = fvg_bear_zone_low
    return result


def check_fvg_retest(
    df: pd.DataFrame,
    max_age: int = 30,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    fvg_bull_hold = np.zeros(n, dtype=bool)
    fvg_bear_hold = np.zeros(n, dtype=bool)
    fvg_bull_fill = np.zeros(n, dtype=bool)
    fvg_bear_fill = np.zeros(n, dtype=bool)
    fvg_bull_sl = np.full(n, np.nan)
    fvg_bear_sl = np.full(n, np.nan)

    active_bull_fvgs = []
    active_bear_fvgs = []

    for i in range(n):
        active_bull_fvgs = [(idx, h, l) for idx, h, l in active_bull_fvgs if i - idx <= max_age]
        active_bear_fvgs = [(idx, h, l) for idx, h, l in active_bear_fvgs if i - idx <= max_age]

        if "fvg_bull_high" in result.columns and not pd.isna(result["fvg_bull_high"].iloc[i]):
            active_bull_fvgs.append((i, result["fvg_bull_high"].iloc[i], result["fvg_bull_low"].iloc[i]))

        if "fvg_bear_high" in result.columns and not pd.isna(result["fvg_bear_high"].iloc[i]):
            active_bear_fvgs.append((i, result["fvg_bear_high"].iloc[i], result["fvg_bear_low"].iloc[i]))

        for idx, h, l in active_bull_fvgs:
            if result["low"].iloc[i] <= h and result["low"].iloc[i] >= l:
                if result["close"].iloc[i] > result["open"].iloc[i]:
                    fvg_bull_hold[i] = True
                    fvg_bull_sl[i] = l
                    break

        for idx, h, l in active_bear_fvgs:
            if result["high"].iloc[i] >= l and result["high"].iloc[i] <= h:
                if result["close"].iloc[i] < result["open"].iloc[i]:
                    fvg_bear_hold[i] = True
                    fvg_bear_sl[i] = h
                    break

        for idx, h, l in active_bull_fvgs:
            if result["close"].iloc[i] < l:
                if result["close"].iloc[i] > result["open"].iloc[i]:
                    fvg_bull_fill[i] = True
                    fvg_bull_sl[i] = l
                    break

        for idx, h, l in active_bear_fvgs:
            if result["close"].iloc[i] > h:
                if result["close"].iloc[i] < result["open"].iloc[i]:
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

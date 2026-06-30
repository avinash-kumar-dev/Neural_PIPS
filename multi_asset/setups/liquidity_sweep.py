import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class LiquiditySweepConfig:
    swing_lookback: int = 5
    fvg_min_size_atr: float = 0.3
    max_bars_after_sweep: int = 10


def detect_fvg(df: pd.DataFrame, min_size_atr: float = 0.3) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    fvg_bull = np.zeros(n, dtype=bool)
    fvg_bear = np.zeros(n, dtype=bool)

    low_vals = df["low"].values
    high_vals = df["high"].values
    atr_vals = df["atr"].values

    for i in range(2, n):
        atr_val = atr_vals[i] if not np.isnan(atr_vals[i]) else 1.0

        if low_vals[i] > high_vals[i - 2]:
            gap = low_vals[i] - high_vals[i - 2]
            if gap >= min_size_atr * atr_val:
                fvg_bull[i] = True

        if high_vals[i] < low_vals[i - 2]:
            gap = low_vals[i - 2] - high_vals[i]
            if gap >= min_size_atr * atr_val:
                fvg_bear[i] = True

    df["fvg_bull"] = fvg_bull
    df["fvg_bear"] = fvg_bear
    return df


def detect_ob(df: pd.DataFrame, displacement_mult: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    ob_bull = np.zeros(n, dtype=bool)
    ob_bear = np.zeros(n, dtype=bool)

    open_vals = df["open"].values
    close_vals = df["close"].values
    high_vals = df["high"].values
    low_vals = df["low"].values
    atr_vals = df["atr"].values

    for i in range(1, n):
        atr_val = atr_vals[i] if not np.isnan(atr_vals[i]) else 1.0
        candle_range = high_vals[i] - low_vals[i]
        if candle_range < displacement_mult * atr_val:
            continue

        if close_vals[i] > open_vals[i]:
            for j in range(i + 1, min(i + 20, n)):
                if close_vals[j] < open_vals[j] and close_vals[j] <= open_vals[i]:
                    ob_bull[i] = True
                    break
                if close_vals[j] > high_vals[i]:
                    break

        if close_vals[i] < open_vals[i]:
            for j in range(i + 1, min(i + 20, n)):
                if close_vals[j] > open_vals[j] and close_vals[j] >= open_vals[i]:
                    ob_bear[i] = True
                    break
                if close_vals[j] < low_vals[i]:
                    break

    df["ob_bull"] = ob_bull
    df["ob_bear"] = ob_bear
    return df


def detect_liquidity_sweep_signals(
    df: pd.DataFrame,
    cfg: LiquiditySweepConfig = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = LiquiditySweepConfig()

    df = df.copy()
    df = detect_fvg(df, cfg.fvg_min_size_atr)
    df = detect_ob(df)

    n = len(df)
    lb = cfg.swing_lookback
    high_vals = df["high"].values
    low_vals = df["low"].values
    close_vals = df["close"].values
    atr_vals = df["atr"].values

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(lb, n - lb):
        is_sh = True
        is_sl = True
        for j in range(1, lb + 1):
            if high_vals[i] < high_vals[i - j] or high_vals[i] < high_vals[i + j]:
                is_sh = False
            if low_vals[i] > low_vals[i - j] or low_vals[i] > low_vals[i + j]:
                is_sl = False
        swing_high[i] = is_sh
        swing_low[i] = is_sl

    ls_long = np.zeros(n, dtype=bool)
    ls_short = np.zeros(n, dtype=bool)
    ls_sl_long = np.full(n, np.nan)
    ls_sl_short = np.full(n, np.nan)

    fvg_bull = df["fvg_bull"].values
    fvg_bear = df["fvg_bear"].values
    ob_bull = df["ob_bull"].values
    ob_bear = df["ob_bear"].values

    last_sh_price = np.nan
    last_sh_idx = -1
    last_sl_price = np.nan
    last_sl_idx = -1

    for i in range(n):
        if swing_high[i]:
            last_sh_price = high_vals[i]
            last_sh_idx = i
        if swing_low[i]:
            last_sl_price = low_vals[i]
            last_sl_idx = i

        if not np.isnan(last_sh_price) and (i - last_sh_idx) > lb:
            if high_vals[i] > last_sh_price:
                atr_val = atr_vals[i] if not np.isnan(atr_vals[i]) else 1.0
                for j in range(i + 1, min(i + cfg.max_bars_after_sweep, n)):
                    if close_vals[j] < low_vals[i]:
                        if fvg_bear[j] or ob_bear[j]:
                            ls_short[j] = True
                            ls_sl_short[j] = high_vals[i] + atr_val * 0.2
                            break

        if not np.isnan(last_sl_price) and (i - last_sl_idx) > lb:
            if low_vals[i] < last_sl_price:
                atr_val = atr_vals[i] if not np.isnan(atr_vals[i]) else 1.0
                for j in range(i + 1, min(i + cfg.max_bars_after_sweep, n)):
                    if close_vals[j] > high_vals[i]:
                        if fvg_bull[j] or ob_bull[j]:
                            ls_long[j] = True
                            ls_sl_long[j] = low_vals[i] - atr_val * 0.2
                            break

    df["ls_long"] = ls_long
    df["ls_short"] = ls_short
    df["ls_sl_long"] = ls_sl_long
    df["ls_sl_short"] = ls_sl_short

    return df

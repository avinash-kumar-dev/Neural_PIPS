import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class LiquiditySweepConfig:
    swing_lookback: int = 5
    sweep_tolerance_atr: float = 0.3
    mss_lookback: int = 3
    fvg_min_size_atr: float = 0.3
    max_bars_after_sweep: int = 10


def detect_liquidity_levels(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    eqh = np.zeros(n, dtype=bool)
    eql = np.zeros(n, dtype=bool)

    for i in range(lookback, n):
        window_high = df["high"].iloc[i - lookback:i].values
        window_low = df["low"].iloc[i - lookback:i].values

        current_high = df["high"].iloc[i]
        current_low = df["low"].iloc[i]

        atr_val = df["atr"].iloc[i] if "atr" in df.columns and not np.isnan(df["atr"].iloc[i]) else 1.0
        tolerance = atr_val * 0.05

        high_matches = np.sum(np.abs(window_high - current_high) < tolerance)
        if high_matches >= 2:
            eqh[i] = True

        low_matches = np.sum(np.abs(window_low - current_low) < tolerance)
        if low_matches >= 2:
            eql[i] = True

    df["eqh"] = eqh
    df["eql"] = eql
    return df


def detect_fvg(df: pd.DataFrame, min_size_atr: float = 0.3) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    fvg_bull = np.zeros(n, dtype=bool)
    fvg_bear = np.zeros(n, dtype=bool)
    fvg_top = np.full(n, np.nan)
    fvg_bottom = np.full(n, np.nan)

    for i in range(2, n):
        atr_val = df["atr"].iloc[i] if "atr" in df.columns and not np.isnan(df["atr"].iloc[i]) else 1.0

        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            gap = df["low"].iloc[i] - df["high"].iloc[i - 2]
            if gap >= min_size_atr * atr_val:
                fvg_bull[i] = True
                fvg_top[i] = df["low"].iloc[i]
                fvg_bottom[i] = df["high"].iloc[i - 2]

        if df["high"].iloc[i] < df["low"].iloc[i - 2]:
            gap = df["low"].iloc[i - 2] - df["high"].iloc[i]
            if gap >= min_size_atr * atr_val:
                fvg_bear[i] = True
                fvg_top[i] = df["low"].iloc[i - 2]
                fvg_bottom[i] = df["high"].iloc[i]

    df["fvg_bull"] = fvg_bull
    df["fvg_bear"] = fvg_bear
    df["fvg_top"] = fvg_top
    df["fvg_bottom"] = fvg_bottom
    return df


def detect_order_blocks(df: pd.DataFrame, displacement_mult: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    ob_bull = np.zeros(n, dtype=bool)
    ob_bear = np.zeros(n, dtype=bool)

    for i in range(1, n):
        atr_val = df["atr"].iloc[i] if "atr" in df.columns and not np.isnan(df["atr"].iloc[i]) else 1.0
        body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        candle_range = df["high"].iloc[i] - df["low"].iloc[i]

        if candle_range < displacement_mult * atr_val:
            continue

        if df["close"].iloc[i] > df["open"].iloc[i]:
            for j in range(i + 1, min(i + 20, n)):
                if df["close"].iloc[j] < df["open"].iloc[j] and df["close"].iloc[j] <= df["open"].iloc[i]:
                    ob_bull[i] = True
                    break
                if df["close"].iloc[j] > df["high"].iloc[i]:
                    break

        if df["close"].iloc[i] < df["open"].iloc[i]:
            for j in range(i + 1, min(i + 20, n)):
                if df["close"].iloc[j] > df["open"].iloc[j] and df["close"].iloc[j] >= df["open"].iloc[i]:
                    ob_bear[i] = True
                    break
                if df["close"].iloc[j] < df["low"].iloc[i]:
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
    df = detect_liquidity_levels(df)
    df = detect_fvg(df, cfg.fvg_min_size_atr)
    df = detect_order_blocks(df)

    n = len(df)
    ls_long = np.zeros(n, dtype=bool)
    ls_short = np.zeros(n, dtype=bool)
    ls_sl_long = np.full(n, np.nan)
    ls_sl_short = np.full(n, np.nan)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    lb = cfg.swing_lookback

    for i in range(lb, n - lb):
        if all(df["high"].iloc[i] >= df["high"].iloc[i - j] for j in range(1, lb + 1)) and \
           all(df["high"].iloc[i] >= df["high"].iloc[i + j] for j in range(1, lb + 1)):
            swing_high[i] = True
        if all(df["low"].iloc[i] <= df["low"].iloc[i - j] for j in range(1, lb + 1)) and \
           all(df["low"].iloc[i] <= df["low"].iloc[i + j] for j in range(1, lb + 1)):
            swing_low[i] = True

    df["ls_swing_high"] = swing_high
    df["ls_swing_low"] = swing_low

    last_sh_price = np.nan
    last_sh_idx = -1
    last_sl_price = np.nan
    last_sl_idx = -1

    for i in range(n):
        if swing_high[i]:
            last_sh_price = df["high"].iloc[i]
            last_sh_idx = i
        if swing_low[i]:
            last_sl_price = df["low"].iloc[i]
            last_sl_idx = i

        if not np.isnan(last_sh_price):
            atr_val = df["atr"].iloc[i] if "atr" in df.columns and not np.isnan(df["atr"].iloc[i]) else 1.0
            swept = df["high"].iloc[i] > last_sh_price and df["low"].iloc[i] < last_sh_price
            if swept and (i - last_sh_idx) > cfg.mss_lookback:
                for j in range(i + 1, min(i + cfg.max_bars_after_sweep, n)):
                    if df["close"].iloc[j] < df["low"].iloc[i]:
                        if df.get("fvg_bear", pd.Series(False, index=df.index)).iloc[j] or \
                           df.get("ob_bear", pd.Series(False, index=df.index)).iloc[j]:
                            ls_short[j] = True
                            ls_sl_short[j] = df["high"].iloc[i] + atr_val * 0.2
                            break

        if not np.isnan(last_sl_price):
            atr_val = df["atr"].iloc[i] if "atr" in df.columns and not np.isnan(df["atr"].iloc[i]) else 1.0
            swept = df["low"].iloc[i] < last_sl_price and df["high"].iloc[i] > last_sl_price
            if swept and (i - last_sl_idx) > cfg.mss_lookback:
                for j in range(i + 1, min(i + cfg.max_bars_after_sweep, n)):
                    if df["close"].iloc[j] > df["high"].iloc[i]:
                        if df.get("fvg_bull", pd.Series(False, index=df.index)).iloc[j] or \
                           df.get("ob_bull", pd.Series(False, index=df.index)).iloc[j]:
                            ls_long[j] = True
                            ls_sl_long[j] = df["low"].iloc[i] - atr_val * 0.2
                            break

    df["ls_long"] = ls_long
    df["ls_short"] = ls_short
    df["ls_sl_long"] = ls_sl_long
    df["ls_sl_short"] = ls_sl_short

    return df

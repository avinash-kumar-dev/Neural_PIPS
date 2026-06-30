import numpy as np
import pandas as pd


def detect_pivot_points(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    df = df.copy()
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    pivot_high = np.full(n, np.nan)
    pivot_low = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        if highs[i] == np.max(highs[i - lookback:i + lookback + 1]):
            pivot_high[i] = highs[i]
        if lows[i] == np.min(lows[i - lookback:i + lookback + 1]):
            pivot_low[i] = lows[i]

    df["pivot_high"] = pivot_high
    df["pivot_low"] = pivot_low
    return df


def detect_ranges(df: pd.DataFrame, lookback: int = 20, min_range_pct: float = 0.002) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    close = df["close"].values

    range_high = np.full(n, np.nan)
    range_low = np.full(n, np.nan)
    in_range = np.zeros(n, dtype=bool)

    for i in range(lookback, n):
        window = close[i - lookback:i]
        h = np.max(window)
        l = np.min(window)
        rng = (h - l) / l if l > 0 else 0

        if rng < min_range_pct:
            in_range[i] = True
            range_high[i] = h
            range_low[i] = l

    df["range_high"] = range_high
    df["range_low"] = range_low
    df["in_range"] = in_range
    return df


def detect_po3_signals(df: pd.DataFrame, range_lookback: int = 20,
                        min_range_pct: float = 0.003,
                        sweep_min_atr: float = 0.3,
                        min_range_bars: int = 5) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values

    range_high = np.full(n, np.nan)
    range_low = np.full(n, np.nan)
    range_start = np.full(n, -1, dtype=int)

    entry_long = np.zeros(n, dtype=bool)
    entry_short = np.zeros(n, dtype=bool)
    entry_sl_long = np.full(n, np.nan)
    entry_sl_short = np.full(n, np.nan)

    current_range_high = np.nan
    current_range_low = np.nan
    current_range_start = -1
    range_bars = 0

    for i in range(range_lookback, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        window = close[i - range_lookback:i]
        h = np.max(window)
        l = np.min(window)
        rng = (h - l) / l if l > 0 else 0

        if rng < min_range_pct:
            range_bars += 1
            if np.isnan(current_range_high) or h > current_range_high:
                current_range_high = h
            if np.isnan(current_range_low) or l < current_range_low:
                current_range_low = l
            if current_range_start == -1:
                current_range_start = i
        else:
            if current_range_start != -1 and range_bars >= min_range_bars:
                sweep_above = high[i] > current_range_high
                sweep_below = low[i] < current_range_low

                if sweep_above:
                    sweep_dist = (high[i] - current_range_high) / atr[i]
                    if sweep_dist >= sweep_min_atr:
                        for j in range(i + 1, min(i + 15, n)):
                            if close[j] < current_range_high:
                                entry_short[j] = True
                                entry_sl_short[j] = high[i] + atr[i] * 0.5
                                break

                if sweep_below:
                    sweep_dist = (current_range_low - low[i]) / atr[i]
                    if sweep_dist >= sweep_min_atr:
                        for j in range(i + 1, min(i + 15, n)):
                            if close[j] > current_range_low:
                                entry_long[j] = True
                                entry_sl_long[j] = low[i] - atr[i] * 0.5
                                break

            current_range_high = np.nan
            current_range_low = np.nan
            current_range_start = -1
            range_bars = 0

    df["entry_long"] = entry_long
    df["entry_short"] = entry_short
    df["entry_sl_long"] = entry_sl_long
    df["entry_sl_short"] = entry_sl_short
    df["setup_long"] = "po3"
    df["setup_short"] = "po3"
    return df

import pandas as pd
import numpy as np


def compute_htf_bias(
    df_htf: pd.DataFrame,
    swing_lookback: int = 5,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> pd.DataFrame:
    result = df_htf.copy()
    n = len(result)

    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values

    left_max = pd.Series(highs).rolling(swing_lookback, min_periods=swing_lookback).max().shift(1).values
    left_min = pd.Series(lows).rolling(swing_lookback, min_periods=swing_lookback).min().shift(1).values

    bos_bullish = np.zeros(n, dtype=bool)
    bos_bearish = np.zeros(n, dtype=bool)
    choch_bullish = np.zeros(n, dtype=bool)
    choch_bearish = np.zeros(n, dtype=bool)

    swing_highs = []
    swing_lows = []

    for i in range(swing_lookback, n - swing_lookback):
        is_swing_high = True
        for j in range(1, swing_lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append((i, highs[i]))

        is_swing_low = True
        for j in range(1, swing_lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append((i, lows[i]))

    last_direction = 0
    sh_idx = 0
    sl_idx = 0

    for i in range(n):
        while sh_idx < len(swing_highs) and swing_highs[sh_idx][0] < i:
            sh_idx += 1
        while sl_idx < len(swing_lows) and swing_lows[sl_idx][0] < i:
            sl_idx += 1

        last_sh = None
        for j in range(sh_idx - 1, -1, -1):
            if swing_highs[j][0] < i:
                last_sh = swing_highs[j][1]
                break

        last_sl = None
        for j in range(sl_idx - 1, -1, -1):
            if swing_lows[j][0] < i:
                last_sl = swing_lows[j][1]
                break

        if last_sh is not None and closes[i] > last_sh:
            if last_direction == -1:
                choch_bullish[i] = True
            else:
                bos_bullish[i] = True
            last_direction = 1

        if last_sl is not None and closes[i] < last_sl:
            if last_direction == 1:
                choch_bearish[i] = True
            else:
                bos_bearish[i] = True
            last_direction = -1

    result["bos_bullish"] = bos_bullish
    result["bos_bearish"] = bos_bearish
    result["choch_bullish"] = choch_bullish
    result["choch_bearish"] = choch_bearish

    ema_f = pd.Series(closes).ewm(span=ema_fast, adjust=False).mean().values
    ema_s = pd.Series(closes).ewm(span=ema_slow, adjust=False).mean().values
    result["ema_fast"] = ema_f
    result["ema_slow"] = ema_s
    result["ema_bullish"] = ema_f > ema_s
    result["ema_bearish"] = ema_f < ema_s

    direction = np.zeros(n, dtype=int)
    current_dir = 0
    for i in range(n):
        if bos_bullish[i] or choch_bullish[i]:
            current_dir = 1
        elif bos_bearish[i] or choch_bearish[i]:
            current_dir = -1
        direction[i] = current_dir
    result["htf_direction"] = direction

    bias = np.zeros(n, dtype=int)
    for i in range(n):
        if direction[i] == 1 and (ema_f[i] > ema_s[i] or abs(ema_f[i] - ema_s[i]) / ema_s[i] < 0.001):
            bias[i] = 1
        elif direction[i] == -1 and (ema_f[i] < ema_s[i] or abs(ema_f[i] - ema_s[i]) / ema_s[i] < 0.001):
            bias[i] = -1
    result["htf_bias"] = bias

    return result


def align_htf_to_ltf(
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    htf_prefix: str,
) -> pd.DataFrame:
    result = df_ltf.copy()

    if "datetime" not in df_htf.columns:
        return result

    htf_dt = pd.to_datetime(df_htf["datetime"]).values.astype("datetime64[ns]")
    ltf_dt = pd.to_datetime(result["datetime"]).values.astype("datetime64[ns]")

    n = len(result)
    htf_bias = df_htf["htf_bias"].values
    htf_direction = df_htf["htf_direction"].values
    htf_ema_bull = df_htf["ema_bullish"].values
    htf_ema_bear = df_htf["ema_bearish"].values

    aligned_bias = np.zeros(n, dtype=int)
    aligned_direction = np.zeros(n, dtype=int)
    aligned_ema_bull = np.zeros(n, dtype=bool)
    aligned_ema_bear = np.zeros(n, dtype=bool)

    htf_idx = 0
    for i in range(n):
        t = ltf_dt[i]

        while htf_idx < len(htf_dt) - 1 and htf_dt[htf_idx + 1] <= t:
            htf_idx += 1

        if htf_idx < len(htf_dt):
            aligned_bias[i] = htf_bias[htf_idx]
            aligned_direction[i] = htf_direction[htf_idx]
            aligned_ema_bull[i] = htf_ema_bull[htf_idx]
            aligned_ema_bear[i] = htf_ema_bear[htf_idx]

    result[f"{htf_prefix}_bias"] = aligned_bias
    result[f"{htf_prefix}_direction"] = aligned_direction
    result[f"{htf_prefix}_ema_bullish"] = aligned_ema_bull
    result[f"{htf_prefix}_ema_bearish"] = aligned_ema_bear

    return result


def compute_cascade_bias(
    df: pd.DataFrame,
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    h4 = compute_htf_bias(df_h4, swing_lookback=5, ema_fast=12, ema_slow=26)
    h1 = compute_htf_bias(df_h1, swing_lookback=5, ema_fast=20, ema_slow=50)

    result = align_htf_to_ltf(result, h4, "h4")
    result = align_htf_to_ltf(result, h1, "h1")

    n = len(result)
    cascade_bias = np.zeros(n, dtype=int)

    h4_b = result["h4_bias"].values
    h1_b = result["h1_bias"].values

    for i in range(n):
        if h4_b[i] == 1 and h1_b[i] == 1:
            cascade_bias[i] = 1
        elif h4_b[i] == -1 and h1_b[i] == -1:
            cascade_bias[i] = -1
        elif h4_b[i] == 1 and h1_b[i] == 0:
            cascade_bias[i] = 1
        elif h4_b[i] == -1 and h1_b[i] == 0:
            cascade_bias[i] = -1
        elif h4_b[i] == 0 and h1_b[i] == 1:
            cascade_bias[i] = 1
        elif h4_b[i] == 0 and h1_b[i] == -1:
            cascade_bias[i] = -1

    result["cascade_bias"] = cascade_bias

    return result

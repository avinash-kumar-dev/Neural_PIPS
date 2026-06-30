import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ORBConfig:
    range_minutes: int = 30
    min_range_atr: float = 0.3
    max_range_atr: float = 2.0
    min_volume_ratio: float = 1.2
    confirmation_bars: int = 3
    max_entry_bars_after_break: int = 5


def detect_orb_signals(
    df: pd.DataFrame,
    session_start_hour: int,
    session_start_min: int = 0,
    cfg: ORBConfig = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = ORBConfig()

    df = df.copy()
    n = len(df)

    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    or_range = np.full(n, np.nan)
    orb_long = np.zeros(n, dtype=bool)
    orb_short = np.zeros(n, dtype=bool)
    orb_sl_long = np.full(n, np.nan)
    orb_sl_short = np.full(n, np.nan)

    dt = pd.to_datetime(df["datetime"])
    date = dt.dt.date.values
    hour = dt.dt.hour.values
    minute = dt.dt.minute.values

    unique_dates = np.unique(date)

    for d in unique_dates:
        day_mask = date == d
        day_idx = np.where(day_mask)[0]

        if len(day_idx) == 0:
            continue

        or_start = None
        or_end = None

        for idx in day_idx:
            h, m = hour[idx], minute[idx]
            if h == session_start_hour and m == session_start_min:
                or_start = idx
            if or_start is not None and or_end is None:
                minutes_since = (h - session_start_hour) * 60 + (m - session_start_min)
                if minutes_since >= cfg.range_minutes:
                    or_end = idx
                    break

        if or_start is None or or_end is None:
            continue

        day_highs = df["high"].iloc[or_start:or_end + 1].values
        day_lows = df["low"].iloc[or_start:or_end + 1].values

        day_or_high = np.max(day_highs)
        day_or_low = np.min(day_lows)
        day_or_range = day_or_high - day_or_low

        atr_vals = df["atr"].values

        for idx in day_idx:
            or_high[idx] = day_or_high
            or_low[idx] = day_or_low
            or_range[idx] = day_or_range

        break_out_idx = None
        for idx in day_idx:
            if idx <= or_end:
                continue

            h = hour[idx]
            m = minute[idx]
            minutes_after_or = (h - session_start_hour) * 60 + (m - session_start_min) - cfg.range_minutes

            if minutes_after_or < 5:
                continue

            current_atr = atr_vals[idx] if not np.isnan(atr_vals[idx]) else 1.0

            if day_or_range < cfg.min_range_atr * current_atr:
                break
            if day_or_range > cfg.max_range_atr * current_atr:
                break

            vol_ratio = df["volume_ratio"].iloc[idx] if "volume_ratio" in df.columns else 1.0

            if df["close"].iloc[idx] > day_or_high:
                if vol_ratio >= cfg.min_volume_ratio or df["bullish_engulfing"].iloc[idx] or df["pin_bar_bull"].iloc[idx]:
                    if break_out_idx is None or (idx - break_out_idx) <= cfg.max_entry_bars_after_break:
                        orb_long[idx] = True
                        orb_sl_long[idx] = day_or_low
                        if break_out_idx is None:
                            break_out_idx = idx

            if df["close"].iloc[idx] < day_or_low:
                if vol_ratio >= cfg.min_volume_ratio or df["bearish_engulfing"].iloc[idx] or df["pin_bar_bear"].iloc[idx]:
                    if break_out_idx is None or (idx - break_out_idx) <= cfg.max_entry_bars_after_break:
                        orb_short[idx] = True
                        orb_sl_short[idx] = day_or_high
                        if break_out_idx is None:
                            break_out_idx = idx

    df["orb_or_high"] = or_high
    df["orb_or_low"] = or_low
    df["orb_range"] = or_range
    df["orb_long"] = orb_long
    df["orb_short"] = orb_short
    df["orb_sl_long"] = orb_sl_long
    df["orb_sl_short"] = orb_sl_short

    return df


def compute_orb_targets(entry_price: float, sl: float, direction: int, or_range: float, pip_value: float) -> dict:
    risk = abs(entry_price - sl)

    if direction == 1:
        tp1 = entry_price + 1.5 * or_range
        tp2 = entry_price + 2.0 * or_range
        tp3 = entry_price + 3.0 * or_range
    else:
        tp1 = entry_price - 1.5 * or_range
        tp2 = entry_price - 2.0 * or_range
        tp3 = entry_price - 3.0 * or_range

    rr1 = abs(tp1 - entry_price) / risk if risk > 0 else 0
    rr2 = abs(tp2 - entry_price) / risk if risk > 0 else 0
    rr3 = abs(tp3 - entry_price) / risk if risk > 0 else 0

    return {
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "rr1": rr1, "rr2": rr2, "rr3": rr3,
        "risk": risk, "or_range": or_range,
    }

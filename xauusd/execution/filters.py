import pandas as pd
import numpy as np


def compute_session_filter(
    df: pd.DataFrame,
    sessions: dict = None,
) -> pd.DataFrame:
    result = df.copy()

    if sessions is None:
        sessions = {
            "london_open": (7, 10),
            "overlap": (13, 17),
            "ny_open": (13, 15),
        }

    if "datetime" not in result.columns:
        result["in_session"] = True
        return result

    dt = pd.to_datetime(result["datetime"])
    hour = dt.dt.hour

    in_any_session = np.zeros(len(result), dtype=bool)
    session_name = np.array([""] * len(result), dtype=object)

    for name, (start, end) in sessions.items():
        mask = (hour >= start) & (hour < end)
        in_any_session |= mask
        session_name[mask] = name

    result["in_session"] = in_any_session
    result["session_name"] = session_name
    return result


def compute_anti_clustering(
    df: pd.DataFrame,
    min_bars_between: int = 6,
    max_trades_per_session: int = 3,
    max_consecutive_losses: int = 3,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    long_entry = result.get("long_entry", pd.Series(False, index=result.index)).values
    short_entry = result.get("short_entry", pd.Series(False, index=result.index)).values

    long_blocked = np.zeros(n, dtype=bool)
    short_blocked = np.zeros(n, dtype=bool)

    last_trade_bar = -min_bars_between - 1
    trades_this_session = 0
    consecutive_losses = 0
    current_session = None

    for i in range(n):
        session = result.get("session_name", pd.Series("", index=result.index)).iloc[i] if "session_name" in result.columns else ""

        if session != current_session:
            current_session = session
            trades_this_session = 0
            consecutive_losses = 0

        bars_since_last = i - last_trade_bar
        cooldown_ok = bars_since_last >= min_bars_between
        session_limit_ok = trades_this_session < max_trades_per_session
        loss_streak_ok = consecutive_losses < max_consecutive_losses

        can_trade = cooldown_ok and session_limit_ok and loss_streak_ok

        if not can_trade:
            long_blocked[i] = True
            short_blocked[i] = True

        if long_entry[i] and not long_blocked[i]:
            last_trade_bar = i
            trades_this_session += 1
        elif short_entry[i] and not short_blocked[i]:
            last_trade_bar = i
            trades_this_session += 1

    result["long_blocked"] = long_blocked
    result["short_blocked"] = short_blocked
    result["long_entry_filtered"] = long_entry & ~long_blocked
    result["short_entry_filtered"] = short_entry & ~short_blocked
    return result


def compute_spread_filter(
    df: pd.DataFrame,
    max_spread: float = 0.5,
) -> pd.DataFrame:
    result = df.copy()
    if "spread" in result.columns:
        result["spread_ok"] = result["spread"] <= max_spread
    else:
        result["spread_ok"] = True
    return result

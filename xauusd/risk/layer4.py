import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


PIP_VALUE = 0.01  # XAUUSD: 1 pip = 0.01


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    return pd.Series(tr).rolling(period).mean()


def compute_structural_sl(
    df: pd.DataFrame,
    entry_index: int,
    direction: int,
    suggested_sl: float,
    buffer_pips: float = 1.0,
) -> float:
    buffer = buffer_pips * PIP_VALUE
    if direction == 1:
        sl = suggested_sl - buffer
    else:
        sl = suggested_sl + buffer
    return sl


@dataclass
class RiskCheck:
    sl_pips: float
    tp1_pips: float
    tp2_pips: float
    rr_ratio: float
    valid: bool
    reason: str


def check_risk_reward(
    entry_price: float,
    sl: float,
    next_structure_price: Optional[float],
    min_rr: float = 2.0,
) -> RiskCheck:
    sl_pips = abs(entry_price - sl) / PIP_VALUE

    if sl_pips <= 0:
        return RiskCheck(0, 0, 0, 0, False, "zero SL distance")

    tp2_pips = sl_pips * min_rr
    tp2_price = entry_price + tp2_pips * PIP_VALUE if sl < entry_price else entry_price - tp2_pips * PIP_VALUE

    if next_structure_price is not None:
        structure_distance = abs(next_structure_price - entry_price) / PIP_VALUE
        if tp2_pips > structure_distance:
            return RiskCheck(sl_pips, 0, 0, 0, False, f"2x SL ({tp2_pips:.1f}p) > next structure ({structure_distance:.1f}p)")

    tp1_pips = sl_pips

    rr_ratio = tp2_pips / sl_pips if sl_pips > 0 else 0

    return RiskCheck(sl_pips, tp1_pips, tp2_pips, rr_ratio, True, "OK")


def compute_position_size(
    equity: float,
    risk_pct: float,
    sl_pips: float,
    pip_value_per_lot: float = 10.0,
) -> float:
    if sl_pips <= 0:
        return 0
    risk_amount = equity * risk_pct
    lots = risk_amount / (sl_pips * pip_value_per_lot)
    return round(lots, 2)

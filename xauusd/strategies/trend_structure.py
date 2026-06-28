import pandas as pd
import numpy as np

from xauusd.structure.layer1 import compute_layer1
from xauusd.entry.layer2 import compute_layer2
from xauusd.confirmation.layer3 import compute_layer3


def run_trend_strategy(
    df: pd.DataFrame,
    swing_lookback: int = 3,
    displacement_mult: float = 1.5,
    ob_max_age: int = 20,
    fvg_min_size: float = 0.5,
    rsi_period: int = 14,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
    volume_ma_period: int = 20,
    body_ratio_min: float = 0.6,
    ema_fast: int = 50,
    ema_slow: int = 200,
) -> pd.DataFrame:
    l1 = compute_layer1(df, swing_lookback=swing_lookback,
                        ema_fast=ema_fast, ema_slow=ema_slow)

    l2 = compute_layer2(df, l1["layer1_bias"],
                        displacement_mult=displacement_mult,
                        ob_max_age=ob_max_age,
                        fvg_min_size=fvg_min_size,
                        rsi_period=rsi_period,
                        rsi_oversold=rsi_oversold,
                        rsi_overbought=rsi_overbought)

    l3 = compute_layer3(l2, volume_ma_period=volume_ma_period,
                        body_ratio_min=body_ratio_min)

    return l3

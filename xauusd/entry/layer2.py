import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from xauusd.entry.order_blocks import detect_order_blocks, check_ob_retest
from xauusd.entry.fvg import detect_fvg, check_fvg_retest
from xauusd.entry.indicator_triggers import compute_indicator_triggers
from xauusd.structure.layer1 import LONG_ONLY, SHORT_ONLY, NO_BIAS


@dataclass
class EntrySignal:
    index: int
    direction: int
    entry_price: float
    sl: float
    trigger_type: str
    confidence: float


def compute_layer2(
    df: pd.DataFrame,
    layer1_bias: pd.Series,
    displacement_mult: float = 1.5,
    ob_max_age: int = 20,
    fvg_min_size: float = 0.5,
    rsi_period: int = 14,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
) -> pd.DataFrame:
    result = df.copy()

    result = detect_order_blocks(result, displacement_mult=displacement_mult)
    result = check_ob_retest(result, max_age=ob_max_age)
    result = detect_fvg(result, min_size_atr=fvg_min_size)
    result = check_fvg_retest(result)
    result = compute_indicator_triggers(result, rsi_period=rsi_period,
                                         rsi_oversold=rsi_oversold,
                                         rsi_overbought=rsi_overbought)

    n = len(result)
    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    long_sl = np.full(n, np.nan)
    short_sl = np.full(n, np.nan)
    long_trigger = np.array([""] * n, dtype=object)
    short_trigger = np.array([""] * n, dtype=object)

    for i in range(n):
        bias = layer1_bias.iloc[i] if i < len(layer1_bias) else NO_BIAS

        rsi_ok_long = not result["rsi_overbought"].iloc[i] if "rsi_overbought" in result.columns else True
        rsi_ok_short = not result["rsi_oversold"].iloc[i] if "rsi_oversold" in result.columns else True
        macd_ok_long = result["macd_cross_bullish"].iloc[i] if "macd_cross_bullish" in result.columns else True
        macd_ok_short = result["macd_cross_bearish"].iloc[i] if "macd_cross_bearish" in result.columns else True
        indicator_long = rsi_ok_long or macd_ok_long
        indicator_short = rsi_ok_short or macd_ok_short

        if bias == LONG_ONLY:
            if result["ob_bull_signal"].iloc[i] and indicator_long:
                long_entry[i] = True
                long_sl[i] = result["ob_bull_sl"].iloc[i]
                long_trigger[i] = "OB"
            elif result["fvg_bull_hold"].iloc[i] and indicator_long:
                long_entry[i] = True
                long_sl[i] = result["fvg_bull_sl"].iloc[i]
                long_trigger[i] = "FVG_HOLD"
            elif result["fvg_bull_fill"].iloc[i] and indicator_long:
                long_entry[i] = True
                long_sl[i] = result["fvg_bull_sl"].iloc[i]
                long_trigger[i] = "FVG_FILL"

        elif bias == SHORT_ONLY:
            if result["ob_bear_signal"].iloc[i] and indicator_short:
                short_entry[i] = True
                short_sl[i] = result["ob_bear_sl"].iloc[i]
                short_trigger[i] = "OB"
            elif result["fvg_bear_hold"].iloc[i] and indicator_short:
                short_entry[i] = True
                short_sl[i] = result["fvg_bear_sl"].iloc[i]
                short_trigger[i] = "FVG_HOLD"
            elif result["fvg_bear_fill"].iloc[i] and indicator_short:
                short_entry[i] = True
                short_sl[i] = result["fvg_bear_sl"].iloc[i]
                short_trigger[i] = "FVG_FILL"

    result["long_entry"] = long_entry
    result["short_entry"] = short_entry
    result["long_sl"] = long_sl
    result["short_sl"] = short_sl
    result["long_trigger"] = long_trigger
    result["short_trigger"] = short_trigger
    return result

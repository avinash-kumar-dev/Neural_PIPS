import pandas as pd
import numpy as np

from xauusd.structure.layer1 import compute_layer1, LONG_ONLY, SHORT_ONLY, NO_BIAS
from xauusd.structure.regime import compute_regime
from xauusd.structure.premium_discount import compute_premium_discount
from xauusd.structure.ote import compute_ote_zones
from xauusd.structure.liquidity_pools import detect_liquidity_pools
from xauusd.structure.asian_range import compute_asian_range
from xauusd.entry.order_blocks import detect_order_blocks, check_ob_retest
from xauusd.entry.fvg import detect_fvg, check_fvg_retest
from xauusd.entry.bpr import detect_bpr
from xauusd.entry.breaker_blocks import detect_breaker_blocks
from xauusd.entry.ifvg import detect_ifvg
from xauusd.entry.unicorn import detect_unicorn
from xauusd.entry.rejection_blocks import detect_rejection_blocks
from xauusd.entry.vwap import compute_vwap
from xauusd.entry.indicator_triggers import compute_indicator_triggers
from xauusd.entry.m5_confirmation import compute_m5_confirmation
from xauusd.confirmation.layer3 import compute_layer3
from xauusd.execution.confluence import compute_confluence_score
from xauusd.execution.filters import compute_session_filter, compute_anti_clustering


def run_full_pipeline(
    df: pd.DataFrame,
    min_confluence: float = 70.0,
    min_rr: float = 2.0,
) -> pd.DataFrame:
    result = df.copy()

    result = compute_regime(result)
    result = compute_premium_discount(result)
    result = compute_ote_zones(result)
    result = compute_layer1(result)
    result = detect_liquidity_pools(result)
    result = compute_asian_range(result)

    result = detect_order_blocks(result)
    result = check_ob_retest(result)
    result = detect_fvg(result)
    result = check_fvg_retest(result)
    result = detect_bpr(result)
    result = detect_breaker_blocks(result)
    result = detect_ifvg(result)
    result = detect_unicorn(result)
    result = detect_rejection_blocks(result)
    result = compute_vwap(result)
    result = compute_indicator_triggers(result)

    n = len(result)
    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    long_sl = np.full(n, np.nan)
    short_sl = np.full(n, np.nan)
    long_score = np.zeros(n)
    short_score = np.zeros(n)
    long_trigger = np.array([""] * n, dtype=object)
    short_trigger = np.array([""] * n, dtype=object)

    for i in range(n):
        bias = result["layer1_bias"].iloc[i]

        if bias == NO_BIAS:
            continue

        row = result.iloc[i]

        if bias == LONG_ONLY:
            trigger = ""
            sl = np.nan

            if row.get("unicorn_bull", False):
                trigger = "UNICORN"
                sl = row.get("unicorn_bull_mid", np.nan) - 0.02
            elif row.get("has_bpr", False) and row.get("bpr_mid", np.nan) > 0:
                trigger = "BPR"
                sl = row.get("bpr_low", np.nan) - 0.01
            elif row.get("ob_bull_signal", False):
                trigger = "OB"
                sl = row.get("ob_bull_sl", np.nan)
            elif row.get("has_breaker_bull", False):
                trigger = "BREAKER"
                sl = row.get("breaker_bull_low", np.nan) - 0.01
            elif row.get("fvg_bull_hold", False):
                trigger = "FVG_HOLD"
                sl = row.get("fvg_bull_sl", np.nan)
            elif row.get("fvg_bull_fill", False):
                trigger = "FVG_FILL"
                sl = row.get("fvg_bull_sl", np.nan)
            elif row.get("has_ifvg_bull", False):
                trigger = "IFVG"
                sl = row.get("ifvg_bull_low", np.nan) - 0.01
            elif row.get("has_rej_bull", False):
                trigger = "REJECTION"
                sl = row.get("rej_bull_low", np.nan) - 0.01

            if trigger and not pd.isna(sl):
                score = compute_confluence_score(result.iloc[[i]], 1).iloc[0]
                if score >= min_confluence:
                    entry = row["close"]
                    sl_pips = abs(entry - sl) / 0.01
                    tp2 = entry + sl_pips * min_rr * 0.01

                    long_entry[i] = True
                    long_sl[i] = sl
                    long_score[i] = score
                    long_trigger[i] = trigger

        elif bias == SHORT_ONLY:
            trigger = ""
            sl = np.nan

            if row.get("unicorn_bear", False):
                trigger = "UNICORN"
                sl = row.get("unicorn_bear_mid", np.nan) + 0.02
            elif row.get("has_bpr", False) and row.get("bpr_mid", np.nan) > 0:
                trigger = "BPR"
                sl = row.get("bpr_high", np.nan) + 0.01
            elif row.get("ob_bear_signal", False):
                trigger = "OB"
                sl = row.get("ob_bear_sl", np.nan)
            elif row.get("has_breaker_bear", False):
                trigger = "BREAKER"
                sl = row.get("breaker_bear_high", np.nan) + 0.01
            elif row.get("fvg_bear_hold", False):
                trigger = "FVG_HOLD"
                sl = row.get("fvg_bear_sl", np.nan)
            elif row.get("fvg_bear_fill", False):
                trigger = "FVG_FILL"
                sl = row.get("fvg_bear_sl", np.nan)
            elif row.get("has_ifvg_bear", False):
                trigger = "IFVG"
                sl = row.get("ifvg_bear_high", np.nan) + 0.01
            elif row.get("has_rej_bear", False):
                trigger = "REJECTION"
                sl = row.get("rej_bear_high", np.nan) + 0.01

            if trigger and not pd.isna(sl):
                score = compute_confluence_score(result.iloc[[i]], -1).iloc[0]
                if score >= min_confluence:
                    entry = row["close"]
                    sl_pips = abs(entry - sl) / 0.01
                    tp2 = entry - sl_pips * min_rr * 0.01

                    short_entry[i] = True
                    short_sl[i] = sl
                    short_score[i] = score
                    short_trigger[i] = trigger

    result["long_entry"] = long_entry
    result["short_entry"] = short_entry
    result["long_sl"] = long_sl
    result["short_sl"] = short_sl
    result["long_score"] = long_score
    result["short_score"] = short_score
    result["long_trigger"] = long_trigger
    result["short_trigger"] = short_trigger

    result = compute_layer3(result)
    result = compute_session_filter(result)
    result = compute_anti_clustering(result)

    result["long_entry"] = result["long_entry_filtered"] & result["layer3_confirmed"] & result["in_session"]
    result["short_entry"] = result["short_entry_filtered"] & result["layer3_confirmed"] & result["in_session"]

    return result

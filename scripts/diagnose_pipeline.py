import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xauusd.structure.layer1 import compute_layer1, LONG_ONLY, SHORT_ONLY
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
from xauusd.confirmation.layer3 import compute_layer3
from xauusd.execution.confluence import compute_confluence_score
from xauusd.execution.filters import compute_session_filter, compute_anti_clustering


def diagnose(df: pd.DataFrame):
    n = len(df)
    print(f"Total bars: {n}")
    print("=" * 60)

    # Stage 1: Regime
    df = compute_regime(df)
    trending = df["regime_trending"].sum()
    ranging = df["regime_ranging"].sum()
    transition = df["regime_transition"].sum()
    print(f"Regime: {trending} trending, {ranging} ranging, {transition} transition")

    # Stage 2: Premium/Discount + OTE
    df = compute_premium_discount(df)
    df = compute_ote_zones(df)
    discount = df["pd_discount"].sum() if "pd_discount" in df.columns else 0
    premium = df["pd_premium"].sum() if "pd_premium" in df.columns else 0
    in_ote_bull = df.get("in_ote_bull", pd.Series(False, index=df.index)).sum()
    in_ote_bear = df.get("in_ote_bear", pd.Series(False, index=df.index)).sum()
    print(f"Premium/Discount: {discount} discount, {premium} premium")
    print(f"OTE: {in_ote_bull} bull, {in_ote_bear} bear")

    # Stage 3: Layer1
    df = compute_layer1(df)
    long_bias = (df["layer1_bias"] == LONG_ONLY).sum()
    short_bias = (df["layer1_bias"] == SHORT_ONLY).sum()
    no_bias = (df["layer1_bias"] == 0).sum()
    print(f"Layer1: {long_bias} LONG, {short_bias} SHORT, {no_bias} NO_BIAS")

    # Stage 4: Liquidity + Asian
    df = detect_liquidity_pools(df)
    df = compute_asian_range(df)

    # Stage 5: PD Arrays
    df = detect_order_blocks(df)
    df = check_ob_retest(df)
    ob_bull = df["ob_bull_signal"].sum()
    ob_bear = df["ob_bear_signal"].sum()
    print(f"OB: {ob_bull} bull, {ob_bear} bear")

    df = detect_fvg(df)
    df = check_fvg_retest(df)
    fvg_bull_hold = df["fvg_bull_hold"].sum()
    fvg_bear_hold = df["fvg_bear_hold"].sum()
    fvg_bull_fill = df["fvg_bull_fill"].sum()
    fvg_bear_fill = df["fvg_bear_fill"].sum()
    print(f"FVG hold: {fvg_bull_hold} bull, {fvg_bear_hold} bear")
    print(f"FVG fill: {fvg_bull_fill} bull, {fvg_bear_fill} bear")

    df = detect_bpr(df)
    bpr = df["has_bpr"].sum()
    print(f"BPR: {bpr}")

    df = detect_breaker_blocks(df)
    breaker_bull = df["has_breaker_bull"].sum()
    breaker_bear = df["has_breaker_bear"].sum()
    print(f"Breaker: {breaker_bull} bull, {breaker_bear} bear")

    df = detect_ifvg(df)
    ifvg_bull = df["has_ifvg_bull"].sum()
    ifvg_bear = df["has_ifvg_bear"].sum()
    print(f"IFVG: {ifvg_bull} bull, {ifvg_bear} bear")

    df = detect_unicorn(df)
    unicorn_bull = df["unicorn_bull"].sum()
    unicorn_bear = df["unicorn_bear"].sum()
    print(f"Unicorn: {unicorn_bull} bull, {unicorn_bear} bear")

    df = detect_rejection_blocks(df)
    rej_bull = df["has_rej_bull"].sum()
    rej_bear = df["has_rej_bear"].sum()
    print(f"Rejection: {rej_bull} bull, {rej_bear} bear")

    df = compute_vwap(df)
    df = compute_indicator_triggers(df)

    # Stage 6: Entry cascade
    bias = df["layer1_bias"].values
    long_mask = bias == LONG_ONLY
    short_mask = bias == SHORT_ONLY

    long_entries = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)

    for i in np.where(long_mask)[0]:
        row = df.iloc[i]
        if row.get("unicorn_bull", False): long_entries[i] = True
        elif row.get("has_bpr", False) and row.get("bpr_mid", np.nan) > 0: long_entries[i] = True
        elif row.get("ob_bull_signal", False): long_entries[i] = True
        elif row.get("has_breaker_bull", False): long_entries[i] = True
        elif row.get("fvg_bull_hold", False): long_entries[i] = True
        elif row.get("fvg_bull_fill", False): long_entries[i] = True
        elif row.get("has_ifvg_bull", False): long_entries[i] = True
        elif row.get("has_rej_bull", False): long_entries[i] = True

    for i in np.where(short_mask)[0]:
        row = df.iloc[i]
        if row.get("unicorn_bear", False): short_entries[i] = True
        elif row.get("has_bpr", False) and row.get("bpr_mid", np.nan) > 0: short_entries[i] = True
        elif row.get("ob_bear_signal", False): short_entries[i] = True
        elif row.get("has_breaker_bear", False): short_entries[i] = True
        elif row.get("fvg_bear_hold", False): short_entries[i] = True
        elif row.get("fvg_bear_fill", False): short_entries[i] = True
        elif row.get("has_ifvg_bear", False): short_entries[i] = True
        elif row.get("has_rej_bear", False): short_entries[i] = True

    print(f"\n--- Entry Cascade ---")
    print(f"LONG entries (bias+PD match): {long_entries.sum()}")
    print(f"SHORT entries (bias+PD match): {short_entries.sum()}")

    # Stage 7: Confluence scores
    long_scores = compute_confluence_score(df, 1)
    short_scores = compute_confluence_score(df, -1)

    long_pass_50 = (long_scores[long_entries] >= 50).sum() if long_entries.any() else 0
    long_pass_60 = (long_scores[long_entries] >= 60).sum() if long_entries.any() else 0
    long_pass_70 = (long_scores[long_entries] >= 70).sum() if long_entries.any() else 0
    short_pass_50 = (short_scores[short_entries] >= 50).sum() if short_entries.any() else 0
    short_pass_60 = (short_scores[short_entries] >= 60).sum() if short_entries.any() else 0
    short_pass_70 = (short_scores[short_entries] >= 70).sum() if short_entries.any() else 0

    print(f"\n--- Confluence Gate ---")
    print(f"LONG  pass@50: {long_pass_50}, @60: {long_pass_60}, @70: {long_pass_70}")
    print(f"SHORT pass@50: {short_pass_50}, @60: {short_pass_60}, @70: {short_pass_70}")

    # Stage 8: Layer3
    df = compute_layer3(df)
    layer3_pass_long = (long_entries & df["layer3_confirmed"]).sum()
    layer3_pass_short = (short_entries & df["layer3_confirmed"]).sum()
    print(f"\nLayer3: {layer3_pass_long} LONG, {layer3_pass_short} SHORT")

    # Stage 9: Session
    df = compute_session_filter(df)
    in_session = df["in_session"].sum()
    session_pass_long = (long_entries & df["layer3_confirmed"] & df["in_session"]).sum()
    session_pass_short = (short_entries & df["layer3_confirmed"] & df["in_session"]).sum()
    print(f"Session: {in_session} bars in session")
    print(f"After session: {session_pass_long} LONG, {session_pass_short} SHORT")

    # Stage 10: Anti-clustering
    df["long_entry"] = long_entries
    df["short_entry"] = short_entries
    df = compute_anti_clustering(df)
    final_long = df["long_entry_filtered"].sum()
    final_short = df["short_entry_filtered"].sum()
    print(f"\nAnti-clustering: {final_long} LONG, {final_short} SHORT")

    # Full pipeline final
    final_long_full = (df["long_entry_filtered"] & df["layer3_confirmed"] & df["in_session"]).sum()
    final_short_full = (df["short_entry_filtered"] & df["layer3_confirmed"] & df["in_session"]).sum()
    print(f"\n{'='*60}")
    print(f"FINAL: {final_long_full} LONG + {final_short_full} SHORT = {final_long_full + final_short_full} signals")

    # Score distribution for entries that had PD match
    if long_entries.any():
        print(f"\nLONG score distribution (PD-matched):")
        scores = long_scores[long_entries]
        print(f"  min: {scores.min():.1f}, max: {scores.max():.1f}, mean: {scores.mean():.1f}")
        for threshold in [40, 50, 60, 70, 80]:
            print(f"  >= {threshold}: {(scores >= threshold).sum()}")
    if short_entries.any():
        print(f"\nSHORT score distribution (PD-matched):")
        scores = short_scores[short_entries]
        print(f"  min: {scores.min():.1f}, max: {scores.max():.1f}, mean: {scores.mean():.1f}")
        for threshold in [40, 50, 60, 70, 80]:
            print(f"  >= {threshold}: {(scores >= threshold).sum()}")

    return df


if __name__ == "__main__":
    data_path = "xauusd/data/raw/xauusd_m15_test.parquet"
    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)

    if "datetime" not in df.columns:
        if "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["time"], unit="s")
        elif isinstance(df.index, pd.DatetimeIndex):
            df["datetime"] = df.index
            df = df.reset_index(drop=True)

    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    diagnose(df)

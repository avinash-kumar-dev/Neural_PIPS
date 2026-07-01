import sys
sys.path.insert(0, "/home/avinash-kumar/tradding_app")

import pandas as pd
import numpy as np
from pathlib import Path
from multi_asset.data.indicators import compute_basic_indicators, detect_swing_points, detect_bos
from multi_asset.backtest.engine import MultiAssetEngine, EngineConfig, compute_metrics
import pandas_ta as ta
import time

DATA_DIR = "/home/avinash-kumar/tradding_app/multi_asset/data/raw"
RESULTS_FILE = "/home/avinash-kumar/tradding_app/multi_asset/RESULTS_LOG.md"

INSTRUMENTS = {
    "US30": {"pip_value": 1.0, "spread": 1.0, "session": (13, 21), "variant": "A", "max_day": 5, "gap": 4},
    "US100": {"pip_value": 1.0, "spread": 1.0, "session": (13, 21), "variant": "A", "max_day": 5, "gap": 4},
    "XAUUSD": {"pip_value": 0.10, "spread": 3.0, "session": (0, 24), "variant": "B", "max_day": 5, "gap": 4},
    "GBPUSD": {"pip_value": 0.0001, "spread": 1.0, "session": (6, 21), "variant": "D", "max_day": 5, "gap": 4},
    "USDJPY": {"pip_value": 0.01, "spread": 1.0, "session": (0, 24), "variant": "A", "max_day": 5, "gap": 4},
}

VARIANT_LABELS = {"A": "Baseline", "B": "+ BB Expanding", "C": "+ MACD", "D": "+ Volume Spike", "E": "BB+Tight"}


def precompute(df):
    df = compute_basic_indicators(df, atr_period=14)
    df = detect_swing_points(df, lookback=5)
    df = detect_bos(df)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["st_dir"] = st.iloc[:, 1]
    st2 = ta.supertrend(df["high"], df["low"], df["close"], length=20, multiplier=4.0)
    df["st2_dir"] = st2.iloc[:, 1]
    df["bb_expanding"] = (df["bb_width"] > df["bb_width"].rolling(20).mean()).values
    df["macd_hist"] = ta.macd(df["close"])["MACDh_12_26_9"]
    df["macd_bull"] = df["macd_hist"] > 0
    df["macd_bear"] = df["macd_hist"] < 0
    df["vol_spike"] = (df["volume"] / df["volume"].rolling(20).mean()) > 2.0
    return df


def apply_variant(df, variant, ss, se):
    c = df["close"].values; e200 = df["ema_200"].values; adx = df["adx"].values
    rsi = df["rsi"].values; atr = df["atr"].values
    valid = (~np.isnan(e200)) & (~np.isnan(atr)) & (atr > 0) & (~np.isnan(adx)) & (~np.isnan(rsi))
    lt = valid & (c > e200) & ((df["st_dir"].values == 1) | (df["st2_dir"].values == 1))
    st_ = valid & (c < e200) & ((df["st_dir"].values == -1) | (df["st2_dir"].values == -1))
    lc = df["bos_bull"].values | df["bullish_engulfing"].values | df["pin_bar_bull"].values
    sc = df["bos_bear"].values | df["bearish_engulfing"].values | df["pin_bar_bear"].values
    adx_min, rsi_l, rsi_s = 20, 70, 30
    if variant == "E": adx_min, rsi_l, rsi_s = 25, 65, 35
    el = lt & lc & (adx > adx_min) & (rsi < rsi_l)
    es = st_ & sc & (adx > adx_min) & (rsi > rsi_s)
    if variant in ("B", "E"):
        el = el & df["bb_expanding"].values; es = es & df["bb_expanding"].values
    elif variant == "C":
        el = el & df["macd_bull"].values; es = es & df["macd_bear"].values
    elif variant == "D":
        el = el & df["vol_spike"].values; es = es & df["vol_spike"].values
    dt = pd.to_datetime(df["datetime"]); hours = dt.dt.hour.values
    in_s = (hours >= ss) & (hours < se)
    el = el & in_s; es = es & in_s
    combined = el | es; indices = np.where(combined)[0]
    filtered = []; last_idx = -5; day_counts = {}; dates = dt.dt.date.values
    for idx in indices:
        d = dates[idx]
        if d not in day_counts: day_counts[d] = 0
        if (idx - last_idx) < 4 or day_counts[d] >= 5: continue
        filtered.append(idx); last_idx = idx; day_counts[d] += 1
    n = len(df)
    ef_l = np.zeros(n, dtype=bool); ef_s = np.zeros(n, dtype=bool)
    sl_l = np.full(n, np.nan); sl_s = np.full(n, np.nan)
    lo = df["low"].values; hi = df["high"].values; at = df["atr"].values
    for idx in filtered:
        if el[idx]: ef_l[idx] = True; sl_l[idx] = lo[idx] - at[idx] * 1.5
        elif es[idx]: ef_s[idx] = True; sl_s[idx] = hi[idx] + at[idx] * 1.5
    df["entry_long"] = ef_l; df["entry_short"] = ef_s
    df["entry_sl_long"] = sl_l; df["entry_sl_short"] = sl_s
    df["setup_long"] = "supertrend"; df["setup_short"] = "supertrend"
    return df


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 7: COMBINED WALK-FORWARD — Best Strategy Per Instrument")
    print("=" * 70)
    print()
    for name, cfg in INSTRUMENTS.items():
        print(f"  {name}: Variant {cfg['variant']} ({VARIANT_LABELS[cfg['variant']]}), "
              f"session {cfg['session']}, max {cfg['max_day']}/day")
    print()

    all_data = {}
    for name in INSTRUMENTS:
        t0 = time.time()
        df = pd.read_parquet(Path(DATA_DIR) / f"{name}_M5.parquet")
        all_data[name] = precompute(df)
        print(f"  {name}: {len(df):,} bars ({time.time()-t0:.1f}s)")

    print()
    print("Running walk-forward...")
    print()

    all_wf = {}
    for name, cfg in INSTRUMENTS.items():
        t0 = time.time()
        df = all_data[name]
        bars_per_month = len(df) / 19.0
        train_bars = int(bars_per_month * 6)
        test_bars = int(bars_per_month * 1)
        step = test_bars

        ss, se = cfg["session"]
        wf = []
        for start in range(0, len(df) - train_bars - test_bars + 1, step):
            ts = df.iloc[start + train_bars: min(start + train_bars + test_bars, len(df))].copy()
            if len(ts) < 50: continue
            ts = apply_variant(ts, cfg["variant"], ss, se)
            tc = int(ts["entry_long"].sum()); ts2 = int(ts["entry_short"].sum())
            if tc + ts2 == 0:
                wf.append({"trades": 0, "pnl": 0, "wr": 0, "pf": 0}); continue
            sigs = ts[["entry_long", "entry_short", "entry_sl_long", "entry_sl_short",
                        "setup_long", "setup_short"]].copy()
            sigs["pip_value"] = cfg["pip_value"]
            engine = MultiAssetEngine(EngineConfig(
                spread_pips=cfg["spread"], slippage_pips=0.5, min_rr=2.0,
                breakeven_trigger_r=2.0, trailing_start_r=3.0, trailing_step_r=1.0))
            trades = engine.run({name: ts}, {name: sigs})
            m = compute_metrics(trades)
            wf.append({"trades": m["total_trades"], "pnl": m["total_pnl_pips"],
                        "wr": m["win_rate"], "pf": m["profit_factor"]})

        all_wf[name] = wf
        total_pnl = sum(w["pnl"] for w in wf)
        profitable = sum(1 for w in wf if w["pnl"] > 0)
        total_w = len(wf)
        avg_wr = np.mean([w["wr"] for w in wf if w["trades"] > 0]) if any(w["trades"] > 0 for w in wf) else 0
        print(f"  {name}: {profitable}/{total_w} windows ({profitable/total_w*100:.0f}%), "
              f"PnL {total_pnl:+.1f}, WR {avg_wr*100:.0f}% ({time.time()-t0:.1f}s)")

    print()
    print("=" * 70)
    print("COMBINED WALK-FORWARD RESULTS")
    print("=" * 70)

    max_windows = max(len(wf) for wf in all_wf.values())
    combined_per_window = []
    for i in range(max_windows):
        total_pnl = 0
        total_trades = 0
        for name, wf in all_wf.items():
            if i < len(wf):
                total_pnl += wf[i]["pnl"]
                total_trades += wf[i]["trades"]
        combined_per_window.append({"pnl": total_pnl, "trades": total_trades})

    print(f"\n{'Window':<8} {'Trades':>7} {'PnL':>10}")
    print("-" * 30)
    for i, w in enumerate(combined_per_window):
        if w["trades"] > 0:
            print(f"{i+1:<8} {w['trades']:>7} {w['pnl']:>+9.1f}")
    print("-" * 30)

    total_pnl = sum(w["pnl"] for w in combined_per_window)
    total_trades = sum(w["trades"] for w in combined_per_window)
    profitable = sum(1 for w in combined_per_window if w["pnl"] > 0)
    total_w = len(combined_per_window)

    print(f"{'TOTAL':<8} {total_trades:>7} {total_pnl:>+9.1f}")
    print(f"\nProfitable windows: {profitable}/{total_w} ({profitable/total_w*100:.0f}%)")

    months = total_w * 1
    pips_per_month = total_pnl / months if months > 0 else 0
    print(f"Period: {months} months")
    print(f"Pips/month: {pips_per_month:+.1f}")
    print(f"Target: 4,000 pips/month")

    if pips_per_month >= 4000:
        print(f"TARGET MET: {pips_per_month:+.0f} >= 4,000")
    else:
        gap = 4000 - pips_per_month
        print(f"Target gap: {gap:+.0f} pips/month short")

    with open(RESULTS_FILE, "a") as f:
        f.write("\n\n---\n\n# Step 7: Combined Walk-Forward — Best Per Instrument\n\n")
        f.write("| Instrument | Variant | WF Windows | Profitable | WF PnL |\n")
        f.write("|------------|---------|------------|------------|--------|\n")
        for name, cfg in INSTRUMENTS.items():
            wf = all_wf[name]
            total = sum(w["pnl"] for w in wf)
            wins = sum(1 for w in wf if w["pnl"] > 0)
            total_w = len(wf)
            f.write(f"| {name} | {cfg['variant']}: {VARIANT_LABELS[cfg['variant']]} | "
                    f"{total_w} | {wins}/{total_w} | {total:+.1f} |\n")
        f.write(f"\n**Combined: {total_pnl:+.1f} pips over {months} months = {pips_per_month:+.1f} pips/month**\n")
        f.write(f"\nProfitable windows: {profitable}/{total_w} ({profitable/total_w*100:.0f}%)\n")
        f.write(f"\n---\n")

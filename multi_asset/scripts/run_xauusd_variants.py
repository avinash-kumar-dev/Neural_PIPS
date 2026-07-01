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


def precompute(df):
    df = compute_basic_indicators(df, atr_period=14)
    df = detect_swing_points(df, lookback=5)
    df = detect_bos(df)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["st_dir"] = st.iloc[:, 1]
    st2 = ta.supertrend(df["high"], df["low"], df["close"], length=20, multiplier=4.0)
    df["st2_dir"] = st2.iloc[:, 1]

    df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).mean()).values
    df["bb_expanding"] = (df["bb_width"] > df["bb_width"].rolling(20).mean()).values

    df["macd_hist"] = ta.macd(df["close"])["MACDh_12_26_9"]
    df["macd_bull"] = df["macd_hist"] > 0
    df["macd_bear"] = df["macd_hist"] < 0

    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume_ratio"] > 2.0
    df["vol_spike_1_5"] = df["volume_ratio"] > 1.5

    df["atr_expanding"] = df["atr"] > df["atr"].rolling(20).mean()

    return df


def generate_signals(df, variant, ss=0, se=24, max_day=5, gap=4):
    c = df["close"].values
    e200 = df["ema_200"].values
    adx = df["adx"].values
    rsi = df["rsi"].values
    atr = df["atr"].values
    valid = (~np.isnan(e200)) & (~np.isnan(atr)) & (atr > 0) & (~np.isnan(adx)) & (~np.isnan(rsi))

    lt = valid & (c > e200) & ((df["st_dir"].values == 1) | (df["st2_dir"].values == 1))
    st_ = valid & (c < e200) & ((df["st_dir"].values == -1) | (df["st2_dir"].values == -1))

    lc = df["bos_bull"].values | df["bullish_engulfing"].values | df["pin_bar_bull"].values
    sc = df["bos_bear"].values | df["bearish_engulfing"].values | df["pin_bar_bear"].values

    adx_min = 20
    rsi_long_max = 70
    rsi_short_min = 30
    use_bb = False
    use_macd = False
    use_volume = False
    use_atr_exp = False
    tight_session = False
    vol_thresh = 2.0

    if variant == "A":
        pass
    elif variant == "B":
        use_bb = True
    elif variant == "C":
        use_macd = True
    elif variant == "D":
        use_volume = True
        vol_thresh = 2.0
    elif variant == "E":
        tight_session = True
        adx_min = 25
        rsi_long_max = 65
        rsi_short_min = 35

    el = lt & lc & (adx > adx_min) & (rsi < rsi_long_max)
    es = st_ & sc & (adx > adx_min) & (rsi > rsi_short_min)

    if use_bb:
        el = el & df["bb_expanding"].values
        es = es & df["bb_expanding"].values
    if use_macd:
        el = el & df["macd_bull"].values
        es = es & df["macd_bear"].values
    if use_volume:
        el = el & df["vol_spike"].values
        es = es & df["vol_spike"].values
    if use_atr_exp:
        el = el & df["atr_expanding"].values
        es = es & df["atr_expanding"].values

    dt = pd.to_datetime(df["datetime"])
    hours = dt.dt.hour.values
    if tight_session:
        in_s = (hours >= 13) & (hours < 16)
    else:
        in_s = (hours >= ss) & (hours < se)
    el = el & in_s
    es = es & in_s

    combined = el | es
    indices = np.where(combined)[0]
    filtered = []
    last_idx = -gap - 1
    day_counts = {}
    dates = dt.dt.date.values
    for idx in indices:
        d = dates[idx]
        if d not in day_counts:
            day_counts[d] = 0
        if (idx - last_idx) < gap or day_counts[d] >= max_day:
            continue
        filtered.append(idx)
        last_idx = idx
        day_counts[d] += 1

    n = len(df)
    el_f = np.zeros(n, dtype=bool)
    es_f = np.zeros(n, dtype=bool)
    sl_l = np.full(n, np.nan)
    sl_s = np.full(n, np.nan)
    lo = df["low"].values
    hi = df["high"].values

    for idx in filtered:
        if el[idx]:
            el_f[idx] = True
            sl_l[idx] = lo[idx] - atr[idx] * 1.5
        elif es[idx]:
            es_f[idx] = True
            sl_s[idx] = hi[idx] + atr[idx] * 1.5

    df["entry_long"] = el_f
    df["entry_short"] = es_f
    df["entry_sl_long"] = sl_l
    df["entry_sl_short"] = sl_s
    df["setup_long"] = "supertrend"
    df["setup_short"] = "supertrend"
    return df


def run_variant(name, df, variant, ss, se, max_day, gap):
    df = generate_signals(df.copy(), variant, ss, se, max_day, gap)
    lc = int(df["entry_long"].sum())
    sc = int(df["entry_short"].sum())

    if lc + sc == 0:
        return {"total_trades": 0, "win_rate": 0, "total_pnl_pips": 0, "profit_factor": 0,
                "rr": 0, "max_drawdown_pips": 0, "avg_win": 0, "avg_loss": 0,
                "long_signals": 0, "short_signals": 0}

    signals = df[["entry_long", "entry_short", "entry_sl_long", "entry_sl_short",
                   "setup_long", "setup_short"]].copy()
    signals["pip_value"] = 0.10

    engine = MultiAssetEngine(EngineConfig(
        spread_pips=3.0, slippage_pips=1.0, min_rr=2.0,
        breakeven_trigger_r=2.0, trailing_start_r=3.0, trailing_step_r=1.0,
    ))

    trades = engine.run({name: df}, {name: signals})
    m = compute_metrics(trades)
    m["long_signals"] = lc
    m["short_signals"] = sc
    return m


def run_walk_forward_variant(name, df, variant, cfg):
    bars_per_month = len(df) / 19.0
    train_bars = int(bars_per_month * 6)
    test_bars = int(bars_per_month * 1)
    step = test_bars

    wf_results = []
    for start in range(0, len(df) - train_bars - test_bars + 1, step):
        test_start = start + train_bars
        test_end = min(test_start + test_bars, len(df))
        test_slice = df.iloc[test_start:test_end].copy()

        if len(test_slice) < 50:
            continue

        m = run_variant(name, test_slice, variant, cfg["ss"], cfg["se"], cfg["max_day"], cfg["gap"])
        wf_results.append({
            "trades": m["total_trades"],
            "pnl": m["total_pnl_pips"],
            "wr": m["win_rate"],
            "pf": m["profit_factor"],
        })

    return wf_results


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: XAUUSD — Testing 5 Filter Variants")
    print("=" * 70)

    df = pd.read_parquet(Path(DATA_DIR) / "XAUUSD_M5.parquet")
    df = precompute(df)
    print(f"Loaded {len(df):,} bars")

    variants = {
        "A": {"label": "Baseline (ADX>20, RSI 30-70)", "ss": 0, "se": 24, "max_day": 5, "gap": 4},
        "B": {"label": "+ BB Expanding filter", "ss": 0, "se": 24, "max_day": 5, "gap": 4},
        "C": {"label": "+ MACD confirm", "ss": 0, "se": 24, "max_day": 5, "gap": 4},
        "D": {"label": "+ Volume spike 2x", "ss": 0, "se": 24, "max_day": 5, "gap": 4},
        "E": {"label": "Tight session 13-16 + ADX>25 + RSI tight", "ss": 13, "se": 16, "max_day": 2, "gap": 6},
    }

    results = {}
    for v, cfg in variants.items():
        t0 = time.time()
        m = run_variant("XAUUSD", df, v, cfg["ss"], cfg["se"], cfg["max_day"], cfg["gap"])
        elapsed = time.time() - t0
        results[v] = m
        print(f"\nVariant {v}: {cfg['label']}")
        print(f"  Signals: {m.get('long_signals',0)}L/{m.get('short_signals',0)}S")
        print(f"  Trades: {m['total_trades']}, WR: {m['win_rate']*100:.0f}%, "
              f"PnL: {m['total_pnl_pips']:+.1f}, PF: {m['profit_factor']:.2f}, "
              f"R:R: {m['rr']:.2f} ({elapsed:.1f}s)")

    print(f"\n{'='*70}")
    print("FULL-SAMPLE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Variant':<5} {'Trades':>7} {'WR':>6} {'PnL':>10} {'PF':>6} {'R:R':>6}")
    print("-" * 50)
    for v, m in results.items():
        print(f"{v:<5} {m['total_trades']:>7} {m['win_rate']*100:>5.0f}% {m['total_pnl_pips']:>+9.1f} {m['profit_factor']:>6.2f} {m['rr']:>6.2f}")

    print(f"\n{'='*70}")
    print("WALK-FORWARD VALIDATION")
    print(f"{'='*70}")

    wf_results = {}
    for v, cfg in variants.items():
        t0 = time.time()
        wf = run_walk_forward_variant("XAUUSD", df, v, cfg)
        wf_results[v] = wf

        total_pnl = sum(w["pnl"] for w in wf)
        total_trades = sum(w["trades"] for w in wf)
        profitable = sum(1 for w in wf if w["pnl"] > 0)
        total_w = len(wf)
        avg_wr = np.mean([w["wr"] for w in wf if w["trades"] > 0]) if any(w["trades"] > 0 for w in wf) else 0

        print(f"\nVariant {v}: {cfg['label']}")
        print(f"  Windows: {total_w}, Profitable: {profitable}/{total_w} ({profitable/total_w*100:.0f}%)")
        print(f"  Total: {total_trades} trades, PnL: {total_pnl:+.1f}, Avg WR: {avg_wr*100:.0f}%")
        for i, w in enumerate(wf):
            if w["trades"] > 0:
                print(f"    W{i+1}: {w['trades']} trades, PnL {w['pnl']:+.1f}, WR {w['wr']*100:.0f}%")
        print(f"  ({time.time()-t0:.1f}s)")

    print(f"\n{'='*70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<5} {'Windows':>8} {'Wins':>6} {'Total PnL':>10} {'Pips/Mo':>8}")
    print("-" * 50)
    for v, wf in wf_results.items():
        total_pnl = sum(w["pnl"] for w in wf)
        profitable = sum(1 for w in wf if w["pnl"] > 0)
        total_w = len(wf)
        pips_mo = total_pnl / total_w if total_w > 0 else 0
        print(f"{v:<5} {total_w:>8} {profitable:>6} {total_pnl:>+9.1f} {pips_mo:>+7.1f}")

    with open(RESULTS_FILE, "a") as f:
        f.write("\n\n---\n\n# Step 1: XAUUSD Variant Testing\n\n")
        f.write("## Full-Sample\n\n")
        f.write("| Variant | Trades | WR | PnL | PF | R:R |\n")
        f.write("|---------|--------|-----|-----|-----|-----|\n")
        for v, m in results.items():
            f.write(f"| {v}: {variants[v]['label']} | {m['total_trades']} | {m['win_rate']*100:.0f}% | {m['total_pnl_pips']:+.1f} | {m['profit_factor']:.2f} | {m['rr']:.2f} |\n")
        f.write("\n## Walk-Forward\n\n")
        f.write("| Variant | Windows | Profitable | Total PnL | Pips/Mo |\n")
        f.write("|---------|---------|------------|-----------|--------|\n")
        for v, wf in wf_results.items():
            total_pnl = sum(w["pnl"] for w in wf)
            profitable = sum(1 for w in wf if w["pnl"] > 0)
            total_w = len(wf)
            pips_mo = total_pnl / total_w if total_w > 0 else 0
            f.write(f"| {v}: {variants[v]['label']} | {total_w} | {profitable}/{total_w} | {total_pnl:+.1f} | {pips_mo:+.1f} |\n")
        f.write("\n---\n")

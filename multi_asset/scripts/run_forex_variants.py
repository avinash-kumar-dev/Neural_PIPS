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
    df["bb_expanding"] = (df["bb_width"] > df["bb_width"].rolling(20).mean()).values
    df["macd_hist"] = ta.macd(df["close"])["MACDh_12_26_9"]
    df["macd_bull"] = df["macd_hist"] > 0
    df["macd_bear"] = df["macd_hist"] < 0
    df["vol_spike"] = (df["volume"] / df["volume"].rolling(20).mean()) > 2.0
    return df


def apply_variant(df, variant, ss, se, max_day, gap):
    c = df["close"].values; e200 = df["ema_200"].values; adx = df["adx"].values
    rsi = df["rsi"].values; atr = df["atr"].values
    valid = (~np.isnan(e200)) & (~np.isnan(atr)) & (atr > 0) & (~np.isnan(adx)) & (~np.isnan(rsi))

    lt = valid & (c > e200) & ((df["st_dir"].values == 1) | (df["st2_dir"].values == 1))
    st_ = valid & (c < e200) & ((df["st_dir"].values == -1) | (df["st2_dir"].values == -1))
    lc = df["bos_bull"].values | df["bullish_engulfing"].values | df["pin_bar_bull"].values
    sc = df["bos_bear"].values | df["bearish_engulfing"].values | df["pin_bar_bear"].values

    adx_min, rsi_l, rsi_s = 20, 70, 30

    el = lt & lc & (adx > adx_min) & (rsi < rsi_l)
    es = st_ & sc & (adx > adx_min) & (rsi > rsi_s)

    if variant == "B":
        el = el & df["bb_expanding"].values
        es = es & df["bb_expanding"].values
    elif variant == "C":
        el = el & df["macd_bull"].values
        es = es & df["macd_bear"].values
    elif variant == "D":
        el = el & df["vol_spike"].values
        es = es & df["vol_spike"].values
    elif variant == "E":
        adx_min, rsi_l, rsi_s = 25, 65, 35
        el = lt & lc & (adx > adx_min) & (rsi < rsi_l)
        es = st_ & sc & (adx > adx_min) & (rsi > rsi_s)
        el = el & df["bb_expanding"].values
        es = es & df["bb_expanding"].values

    dt = pd.to_datetime(df["datetime"])
    hours = dt.dt.hour.values
    in_s = (hours >= ss) & (hours < se)
    el = el & in_s; es = es & in_s

    combined = el | es
    indices = np.where(combined)[0]
    filtered = []
    last_idx = -gap - 1
    day_counts = {}
    dates = dt.dt.date.values
    for idx in indices:
        d = dates[idx]
        if d not in day_counts: day_counts[d] = 0
        if (idx - last_idx) < gap or day_counts[d] >= max_day: continue
        filtered.append(idx); last_idx = idx; day_counts[d] += 1

    n = len(df)
    el_f = np.zeros(n, dtype=bool); es_f = np.zeros(n, dtype=bool)
    sl_l = np.full(n, np.nan); sl_s = np.full(n, np.nan)
    lo = df["low"].values; hi = df["high"].values
    for idx in filtered:
        if el[idx]: el_f[idx]=True; sl_l[idx]=lo[idx]-atr[idx]*1.5
        elif es[idx]: es_f[idx]=True; sl_s[idx]=hi[idx]+atr[idx]*1.5

    df["entry_long"]=el_f; df["entry_short"]=es_f; df["entry_sl_long"]=sl_l; df["entry_sl_short"]=sl_s
    df["setup_long"]="supertrend"; df["setup_short"]="supertrend"
    return df


def run_variant(name, pip_value, spread, df, variant, ss, se, max_day, gap):
    df = apply_variant(df.copy(), variant, ss, se, max_day, gap)
    lc = int(df["entry_long"].sum()); sc = int(df["entry_short"].sum())
    if lc + sc == 0:
        return {"total_trades": 0, "win_rate": 0, "total_pnl_pips": 0, "profit_factor": 0,
                "rr": 0, "max_drawdown_pips": 0, "long_signals": 0, "short_signals": 0}
    signals = df[["entry_long","entry_short","entry_sl_long","entry_sl_short","setup_long","setup_short"]].copy()
    signals["pip_value"] = pip_value
    engine = MultiAssetEngine(EngineConfig(spread_pips=spread, slippage_pips=0.5, min_rr=2.0,
                                            breakeven_trigger_r=2.0, trailing_start_r=3.0, trailing_step_r=1.0))
    trades = engine.run({name: df}, {name: signals})
    m = compute_metrics(trades)
    m["long_signals"] = lc; m["short_signals"] = sc
    return m


def run_wf(name, pip_value, spread, df, variant, ss, se, max_day, gap):
    bars_per_month = len(df) / 19.0
    train_bars = int(bars_per_month * 6)
    test_bars = int(bars_per_month * 1)
    step = test_bars
    wf = []
    for start in range(0, len(df) - train_bars - test_bars + 1, step):
        ts = df.iloc[start + train_bars: min(start + train_bars + test_bars, len(df))].copy()
        if len(ts) < 50: continue
        m = run_variant(name, pip_value, spread, ts, variant, ss, se, max_day, gap)
        wf.append({"trades": m["total_trades"], "pnl": m["total_pnl_pips"], "wr": m["win_rate"]})
    return wf


if __name__ == "__main__":
    instruments = [
        ("EURUSD", 0.0001, 1.0, 6, 21),
        ("GBPUSD", 0.0001, 1.0, 6, 21),
        ("USDJPY", 0.01, 1.0, 0, 21),
    ]

    variants = ["A", "B", "C", "D", "E"]
    variant_labels = {
        "A": "Baseline",
        "B": "+ BB Expanding",
        "C": "+ MACD Confirm",
        "D": "+ Volume Spike",
        "E": "BB + ADX>25 + RSI tight",
    }

    all_results = {}

    for inst_name, pip_val, spread, ss, se in instruments:
        print(f"\n{'='*70}")
        print(f"STEP: {inst_name} — Testing 5 Filter Variants")
        print(f"{'='*70}")

        df = pd.read_parquet(Path(DATA_DIR) / f"{inst_name}_M5.parquet")
        df = precompute(df)
        print(f"Loaded {len(df):,} bars")

        results = {}
        for v in variants:
            t0 = time.time()
            m = run_variant(inst_name, pip_val, spread, df, v, ss, se, 5, 4)
            elapsed = time.time() - t0
            results[v] = m
            print(f"  {v} ({variant_labels[v]}): {m['total_trades']} trades, "
                  f"WR {m['win_rate']*100:.0f}%, PnL {m['total_pnl_pips']:+.1f}, "
                  f"PF {m['profit_factor']:.2f} ({elapsed:.1f}s)")

        print(f"\n  Walk-forward...")
        wf_results = {}
        for v in variants:
            t0 = time.time()
            wf = run_wf(inst_name, pip_val, spread, df, v, ss, se, 5, 4)
            wf_results[v] = wf
            total_pnl = sum(w["pnl"] for w in wf)
            profitable = sum(1 for w in wf if w["pnl"] > 0)
            total_w = len(wf)
            print(f"    {v}: {profitable}/{total_w} windows ({profitable/total_w*100:.0f}%), "
                  f"PnL {total_pnl:+.1f} ({time.time()-t0:.1f}s)")

        all_results[inst_name] = {"full": results, "wf": wf_results}

        with open(RESULTS_FILE, "a") as f:
            f.write(f"\n## {inst_name} Variants\n\n")
            f.write("| Variant | Trades | WR | PnL | PF | WF Windows | WF PnL |\n")
            f.write("|---------|--------|-----|-----|-----|------------|--------|\n")
            for v in variants:
                fr = results[v]
                wf = wf_results[v]
                wf_pnl = sum(w["pnl"] for w in wf)
                wf_wins = sum(1 for w in wf if w["pnl"] > 0)
                wf_total = len(wf)
                f.write(f"| {v}: {variant_labels[v]} | {fr['total_trades']} | {fr['win_rate']*100:.0f}% | "
                        f"{fr['total_pnl_pips']:+.1f} | {fr['profit_factor']:.2f} | "
                        f"{wf_wins}/{wf_total} | {wf_pnl:+.1f} |\n")
            f.write("\n---\n")

    print(f"\n{'='*70}")
    print("SUMMARY — All Instruments")
    print(f"{'='*70}")
    for inst_name in all_results:
        print(f"\n{inst_name}:")
        for v in variants:
            fr = all_results[inst_name]["full"][v]
            wf = all_results[inst_name]["wf"][v]
            wf_pnl = sum(w["pnl"] for w in wf)
            wf_wins = sum(1 for w in wf if w["pnl"] > 0)
            wf_total = len(wf)
            winner = " <-- WINNER" if v == "B" else ""
            print(f"  {v}: Full PnL {fr['total_pnl_pips']:+.1f}, "
                  f"WF {wf_wins}/{wf_total} windows, WF PnL {wf_pnl:+.1f}{winner}")

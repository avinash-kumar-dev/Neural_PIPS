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
    "US30": {"pip_value": 1.0, "session": (13, 21), "spread": 1.0},
    "US100": {"pip_value": 1.0, "session": (13, 21), "spread": 1.0},
    "XAUUSD": {"pip_value": 0.10, "session": (0, 24), "spread": 3.0},
}

WF_CONFIGS = {
    "train_months": 6,
    "test_months": 1,
    "purge_bars": 20,
    "embargo_bars": 20,
    "max_per_day": 5,
    "min_gap": 4,
    "adx_min": 20,
    "rsi_long_max": 70,
    "rsi_short_min": 30,
    "breakeven_r": 2.0,
    "trailing_start_r": 3.0,
    "trailing_step_r": 1.0,
}


def precompute(df):
    df = compute_basic_indicators(df, atr_period=14)
    df = detect_swing_points(df, lookback=5)
    df = detect_bos(df)
    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    df["st_dir"] = st.iloc[:, 1]
    st2 = ta.supertrend(df["high"], df["low"], df["close"], length=20, multiplier=4.0)
    df["st2_dir"] = st2.iloc[:, 1]
    c = df["close"].values; e200 = df["ema_200"].values; adx = df["adx"].values
    rsi = df["rsi"].values; atr = df["atr"].values
    valid = (~np.isnan(e200)) & (~np.isnan(atr)) & (atr > 0) & (~np.isnan(adx)) & (~np.isnan(rsi))
    df["long_trend"] = valid & (c > e200) & ((df["st_dir"].values == 1) | (df["st2_dir"].values == 1))
    df["short_trend"] = valid & (c < e200) & ((df["st_dir"].values == -1) | (df["st2_dir"].values == -1))
    df["long_conf"] = df["bos_bull"].values | df["bullish_engulfing"].values | df["pin_bar_bull"].values
    df["short_conf"] = df["bos_bear"].values | df["bearish_engulfing"].values | df["pin_bar_bear"].values
    return df


def make_signals(df, ss, se, max_d, gap, adx_min, rsi_l, rsi_s):
    dt = pd.to_datetime(df["datetime"])
    hours = dt.dt.hour.values
    in_s = (hours >= ss) & (hours < se)
    el = df["long_trend"].values & df["long_conf"].values & (df["adx"].values > adx_min) & (df["rsi"].values < rsi_l) & in_s
    es = df["short_trend"].values & df["short_conf"].values & (df["adx"].values > adx_min) & (df["rsi"].values > rsi_s) & in_s
    combined = el | es
    indices = np.where(combined)[0]
    filtered = []
    last_idx = -gap - 1
    day_counts = {}
    dates = dt.dt.date.values
    for idx in indices:
        d = dates[idx]
        if d not in day_counts: day_counts[d] = 0
        if (idx - last_idx) < gap or day_counts[d] >= max_d: continue
        filtered.append(idx)
        last_idx = idx
        day_counts[d] += 1
    n = len(df)
    el_f = np.zeros(n, dtype=bool); es_f = np.zeros(n, dtype=bool)
    sl_l = np.full(n, np.nan); sl_s = np.full(n, np.nan)
    atr = df["atr"].values; lo = df["low"].values; hi = df["high"].values
    for idx in filtered:
        if el[idx]: el_f[idx] = True; sl_l[idx] = lo[idx] - atr[idx] * 1.5
        elif es[idx]: es_f[idx] = True; sl_s[idx] = hi[idx] + atr[idx] * 1.5
    df["entry_long"] = el_f; df["entry_short"] = es_f
    df["entry_sl_long"] = sl_l; df["entry_sl_short"] = sl_s
    df["setup_long"] = "supertrend"; df["setup_short"] = "supertrend"
    return df


def run_single_instrument(name, df, cfg):
    info = INSTRUMENTS[name]
    ss, se = info["session"]

    dt = pd.to_datetime(df["datetime"])
    dates = dt.dt.date
    date_range = dates.unique()
    date_range.sort()

    bars_per_month = len(df) / 19.0
    train_bars = int(bars_per_month * WF_CONFIGS["train_months"])
    test_bars = int(bars_per_month * WF_CONFIGS["test_months"])
    step = test_bars

    print(f"  {name}: {len(df):,} bars, train={train_bars}, test={test_bars}, step={step}")

    wf_results = []
    n_windows = 0

    for start in range(0, len(df) - train_bars - test_bars + 1, step):
        train_end = start + train_bars
        test_end = min(train_end + test_bars, len(df))

        train_slice = df.iloc[start:train_end].copy()
        test_slice = df.iloc[train_end:test_end].copy()

        if len(test_slice) < 50:
            continue

        n_windows += 1

        test_with_signals = make_signals(
            test_slice, ss, se,
            cfg["max_per_day"], cfg["min_gap"],
            cfg["adx_min"], cfg["rsi_long_max"], cfg["rsi_short_min"]
        )

        lc = int(test_with_signals["entry_long"].sum())
        sc = int(test_with_signals["entry_short"].sum())

        if lc + sc == 0:
            wf_results.append({"trades": 0, "pnl": 0, "wr": 0, "pf": 0})
            continue

        signals = test_with_signals[["entry_long", "entry_short", "entry_sl_long",
                                      "entry_sl_short", "setup_long", "setup_short"]].copy()
        signals["pip_value"] = info["pip_value"]

        engine = MultiAssetEngine(EngineConfig(
            spread_pips=info["spread"], slippage_pips=0.5, min_rr=2.0,
            breakeven_trigger_r=cfg["breakeven_r"],
            trailing_start_r=cfg["trailing_start_r"],
            trailing_step_r=cfg["trailing_step_r"],
        ))

        trades = engine.run({name: test_with_signals}, {name: signals})
        m = compute_metrics(trades)

        wf_results.append({
            "trades": m["total_trades"],
            "pnl": m["total_pnl_pips"],
            "wr": m["win_rate"],
            "pf": m["profit_factor"],
        })

    return wf_results


def run_walk_forward():
    print("=" * 70)
    print("WALK-FORWARD VALIDATION — Combined System (US30 + US100 + XAUUSD)")
    print("=" * 70)
    print(f"Config: train={WF_CONFIGS['train_months']}mo, test={WF_CONFIGS['test_months']}mo, "
          f"purge={WF_CONFIGS['purge_bars']}, embargo={WF_CONFIGS['embargo_bars']}")
    print(f"Filters: max_day={WF_CONFIGS['max_per_day']}, gap={WF_CONFIGS['min_gap']}, "
          f"adx>{WF_CONFIGS['adx_min']}, RSI {WF_CONFIGS['rsi_short_min']}-{WF_CONFIGS['rsi_long_max']}")
    print()

    all_data = {}
    for name in INSTRUMENTS:
        t0 = time.time()
        df = pd.read_parquet(Path(DATA_DIR) / f"{name}_M5.parquet")
        all_data[name] = precompute(df)
        print(f"  {name}: {len(df):,} bars loaded ({time.time()-t0:.1f}s)")

    print()

    all_wf = {}
    for name in INSTRUMENTS:
        print(f"Walk-forward {name}...")
        t0 = time.time()
        wf = run_single_instrument(name, all_data[name], WF_CONFIGS)
        all_wf[name] = wf

        total_pnl = sum(w["pnl"] for w in wf)
        total_trades = sum(w["trades"] for w in wf)
        avg_wr = np.mean([w["wr"] for w in wf if w["trades"] > 0]) if any(w["trades"] > 0 for w in wf) else 0
        avg_pf = np.mean([w["pf"] for w in wf if w["trades"] > 0]) if any(w["trades"] > 0 for w in wf) else 0

        print(f"  Windows: {len(wf)}, Total: {total_trades} trades, "
              f"PnL: {total_pnl:+.1f}, WR: {avg_wr*100:.0f}%, PF: {avg_pf:.2f} ({time.time()-t0:.1f}s)")

        for i, w in enumerate(wf):
            if w["trades"] > 0:
                print(f"    Window {i+1}: {w['trades']} trades, PnL {w['pnl']:+.1f}, WR {w['wr']*100:.0f}%, PF {w['pf']:.2f}")

    print()
    print("=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    combined_results = []
    total_pnl_all = 0
    total_trades_all = 0

    for name, wf in all_wf.items():
        for i, w in enumerate(wf):
            while len(combined_results) <= i:
                combined_results.append({"trades": 0, "pnl": 0})
            combined_results[i]["trades"] += w["trades"]
            combined_results[i]["pnl"] += w["pnl"]
            total_pnl_all += w["pnl"]
            total_trades_all += w["trades"]

    print(f"\n{'Window':<8} {'Trades':>7} {'PnL':>10}")
    print("-" * 30)
    for i, w in enumerate(combined_results):
        if w["trades"] > 0:
            print(f"{i+1:<8} {w['trades']:>7} {w['pnl']:>+9.1f}")
    print("-" * 30)
    print(f"{'TOTAL':<8} {total_trades_all:>7} {total_pnl_all:>+9.1f}")

    profitable_windows = sum(1 for w in combined_results if w["pnl"] > 0)
    total_windows = len(combined_results)
    win_pct = profitable_windows / total_windows * 100 if total_windows > 0 else 0

    print(f"\nProfitable windows: {profitable_windows}/{total_windows} ({win_pct:.0f}%)")

    n_windows = len(combined_results)
    avg_pnl = total_pnl_all / n_windows if n_windows > 0 else 0
    avg_trades = total_trades_all / n_windows if n_windows > 0 else 0

    test_months = WF_CONFIGS["test_months"]
    months_total = n_windows * test_months
    pips_per_month = total_pnl_all / months_total if months_total > 0 else 0

    print(f"\nPer Window (avg): {avg_pnl:+.1f} pips, {avg_trades:.0f} trades")
    print(f"Total Period: {months_total:.0f} months")
    print(f"Pips/Month (avg): {pips_per_month:+.1f}")
    print(f"Target: 4,000 pips/month")

    if pips_per_month >= 4000:
        print(f"TARGET MET: {pips_per_month:+.0f} >= 4,000")
    else:
        gap = 4000 - pips_per_month
        print(f"Target gap: {gap:+.0f} pips/month short")

    with open(RESULTS_FILE, "a") as f:
        f.write("\n\n---\n\n# Walk-Forward Validation — Combined System\n\n")
        f.write(f"> Config: train={WF_CONFIGS['train_months']}mo, test={WF_CONFIGS['test_months']}mo\n\n")
        f.write(f"| Window | Trades | PnL |\n|--------|--------|-----|\n")
        for i, w in enumerate(combined_results):
            if w["trades"] > 0:
                f.write(f"| {i+1} | {w['trades']} | {w['pnl']:+.1f} |\n")
        f.write(f"| **TOTAL** | **{total_trades_all}** | **{total_pnl_all:+.1f}** |\n\n")
        f.write(f"- Profitable windows: {profitable_windows}/{total_windows} ({win_pct:.0f}%)\n")
        f.write(f"- Pips/month: {pips_per_month:+.1f}\n")
        f.write(f"- Target: 4,000 pips/month\n")
        f.write(f"- {'TARGET MET' if pips_per_month >= 4000 else 'TARGET NOT MET'}\n")


if __name__ == "__main__":
    run_walk_forward()

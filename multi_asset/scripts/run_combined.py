import sys
sys.path.insert(0, "/home/avinash-kumar/tradding_app")

import pandas as pd
import numpy as np
from pathlib import Path
from multi_asset.data.indicators import compute_basic_indicators, detect_swing_points, detect_bos
from multi_asset.backtest.engine import MultiAssetEngine, EngineConfig, compute_metrics
import pandas_ta as ta
import time

RESULTS_FILE = "/home/avinash-kumar/tradding_app/multi_asset/RESULTS_LOG.md"
DATA_DIR = "/home/avinash-kumar/tradding_app/multi_asset/data/raw"
XAUUSD_DATA = "/home/avinash-kumar/tradding_app/multi_asset/data/raw/XAUUSD_M5.parquet"


def log_result(name, metrics, config_note=""):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n## {name}\n> {config_note}\n\n")
        f.write(f"| Trades | WR | PnL | PF | R:R | MaxDD |\n")
        f.write(f"|--------|-----|-----|-----|-----|-------|\n")
        f.write(f"| {metrics['total_trades']} | {metrics['win_rate']*100:.0f}% | {metrics['total_pnl_pips']:+.1f} | {metrics['profit_factor']:.2f} | {metrics['rr']:.2f} | {metrics['max_drawdown_pips']:.1f} |\n")
        if metrics.get("instruments"):
            f.write(f"\n| Inst | Trades | WR | PnL |\n|------|--------|-----|-----|\n")
            for inst, s in metrics["instruments"].items():
                f.write(f"| {inst} | {s['trades']} | {s['wr']*100:.0f}% | {s['pnl']:+.1f} |\n")
        f.write("\n---\n")


def run_indices(pip_vals, sessions, max_d=5, gap=4, adx_min=20, rsi_l=70, rsi_s=30):
    print("\n" + "=" * 60)
    print("SYSTEM 1: US30 + US100 (Supertrend + EMA200 + BOS)")
    print("=" * 60)

    results = {}
    for name in ["US30", "US100"]:
        t0 = time.time()
        df = pd.read_parquet(Path(DATA_DIR) / f"{name}_M5.parquet")
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
        lt = valid & (c > e200) & ((df["st_dir"].values == 1) | (df["st2_dir"].values == 1))
        st_ = valid & (c < e200) & ((df["st_dir"].values == -1) | (df["st2_dir"].values == -1))
        lc = df["bos_bull"].values | df["bullish_engulfing"].values | df["pin_bar_bull"].values
        sc = df["bos_bear"].values | df["bearish_engulfing"].values | df["pin_bar_bear"].values
        el = lt & lc & (adx > adx_min) & (rsi < rsi_l)
        es = st_ & sc & (adx > adx_min) & (rsi > rsi_s)

        dt = pd.to_datetime(df["datetime"])
        hours = dt.dt.hour.values
        ss, se = sessions[name]
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
            if (idx - last_idx) < gap or day_counts[d] >= max_d: continue
            filtered.append(idx)
            last_idx = idx
            day_counts[d] += 1

        n = len(df)
        el_f = np.zeros(n, dtype=bool); es_f = np.zeros(n, dtype=bool)
        sl_l = np.full(n, np.nan); sl_s = np.full(n, np.nan)
        for idx in filtered:
            if el[idx]: el_f[idx] = True; sl_l[idx] = df["low"].iloc[idx] - atr[idx] * 1.5
            elif es[idx]: es_f[idx] = True; sl_s[idx] = df["high"].iloc[idx] + atr[idx] * 1.5

        df["entry_long"] = el_f; df["entry_short"] = es_f
        df["entry_sl_long"] = sl_l; df["entry_sl_short"] = sl_s
        df["setup_long"] = "supertrend"; df["setup_short"] = "supertrend"

        signals = df[["entry_long", "entry_short", "entry_sl_long", "entry_sl_short",
                       "setup_long", "setup_short"]].copy()
        signals["pip_value"] = pip_vals[name]

        engine = MultiAssetEngine(EngineConfig(
            spread_pips=1.0, slippage_pips=0.5, min_rr=2.0,
            breakeven_trigger_r=2.0, trailing_start_r=3.0, trailing_step_r=1.0,
        ))
        trades = engine.run({name: df}, {name: signals})
        m = compute_metrics(trades)
        results[name] = m
        print(f"  {name}: {m['total_trades']} trades, WR {m['win_rate']*100:.0f}%, "
              f"PnL {m['total_pnl_pips']:+.1f}, PF {m['profit_factor']:.2f} ({time.time()-t0:.1f}s)")

    return results


def run_xauusd():
    print("\n" + "=" * 60)
    print("SYSTEM 2: XAUUSD (ICT/SMC — Cascade + Confluence)")
    print("=" * 60)

    t0 = time.time()

    try:
        from xauusd.pipeline import XAUUSDPipeline
        pipeline = XAUUSDPipeline()

        df = pd.read_parquet(XAUUSD_DATA)
        print(f"  Loaded {len(df):,} M5 bars")

        signals_df = pipeline.generate_signals(df)
        lc = int(signals_df.get("entry_long", pd.Series(dtype=bool)).sum()) if "entry_long" in signals_df else 0
        sc = int(signals_df.get("entry_short", pd.Series(dtype=bool)).sum()) if "entry_short" in signals_df else 0
        print(f"  Signals: {lc}L / {sc}S")

        if lc + sc == 0:
            print("  No signals generated")
            return {}

        engine = MultiAssetEngine(EngineConfig(
            spread_pips=3.0, slippage_pips=1.0, min_rr=2.0,
            breakeven_trigger_r=2.0, trailing_start_r=3.0, trailing_step_r=1.0,
        ))

        trades = engine.run({"XAUUSD": df}, {"XAUUSD": signals_df})
        m = compute_metrics(trades)
        print(f"  XAUUSD: {m['total_trades']} trades, WR {m['win_rate']*100:.0f}%, "
              f"PnL {m['total_pnl_pips']:+.1f}, PF {m['profit_factor']:.2f} ({time.time()-t0:.1f}s)")
        return {"XAUUSD": m}

    except Exception as e:
        print(f"  XAUUSD pipeline error: {e}")
        print("  Running simple Supertrend on XAUUSD as fallback...")

        df = pd.read_parquet(XAUUSD_DATA)
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
        lt = valid & (c > e200) & ((df["st_dir"].values == 1) | (df["st2_dir"].values == 1))
        st_ = valid & (c < e200) & ((df["st_dir"].values == -1) | (df["st2_dir"].values == -1))
        lc = df["bos_bull"].values | df["bullish_engulfing"].values | df["pin_bar_bull"].values
        sc = df["bos_bear"].values | df["bearish_engulfing"].values | df["pin_bar_bear"].values
        el = lt & lc & (adx > 25) & (rsi < 65)
        es = st_ & sc & (adx > 25) & (rsi > 35)

        n = len(df)
        el_f = np.zeros(n, dtype=bool); es_f = np.zeros(n, dtype=bool)
        sl_l = np.full(n, np.nan); sl_s = np.full(n, np.nan)
        for i in range(n):
            if el[i]: el_f[i] = True; sl_l[i] = df["low"].iloc[i] - atr[i] * 1.5
            elif es[i]: es_f[i] = True; sl_s[i] = df["high"].iloc[i] + atr[i] * 1.5

        df["entry_long"] = el_f; df["entry_short"] = es_f
        df["entry_sl_long"] = sl_l; df["entry_sl_short"] = sl_s
        df["setup_long"] = "supertrend"; df["setup_short"] = "supertrend"

        signals = df[["entry_long", "entry_short", "entry_sl_long", "entry_sl_short",
                       "setup_long", "setup_short"]].copy()
        signals["pip_value"] = 0.10

        engine = MultiAssetEngine(EngineConfig(
            spread_pips=3.0, slippage_pips=1.0, min_rr=2.0,
            breakeven_trigger_r=2.0, trailing_start_r=3.0, trailing_step_r=1.0,
        ))
        trades = engine.run({"XAUUSD": df}, {"XAUUSD": signals})
        m = compute_metrics(trades)
        print(f"  XAUUSD: {m['total_trades']} trades, WR {m['win_rate']*100:.0f}%, "
              f"PnL {m['total_pnl_pips']:+.1f}, PF {m['profit_factor']:.2f} ({time.time()-t0:.1f}s)")
        return {"XAUUSD": m}


if __name__ == "__main__":
    pip_vals = {"US30": 1.0, "US100": 1.0}
    sessions = {"US30": (13, 21), "US100": (13, 21)}

    indices_results = run_indices(pip_vals, sessions, max_d=5, gap=4, adx_min=20)
    xauusd_results = run_xauusd()

    all_results = {**indices_results, **xauusd_results}

    print("\n" + "=" * 60)
    print("COMBINED RESULTS — BOTH SYSTEMS RUNNING SIMULTANEOUSLY")
    print("=" * 60)
    print(f"{'Instrument':<12} {'Trades':>7} {'WR':>6} {'PnL':>10} {'PF':>6}")
    print("-" * 50)
    total_pnl = 0
    total_trades = 0
    for name, m in all_results.items():
        total_pnl += m["total_pnl_pips"]
        total_trades += m["total_trades"]
        print(f"{name:<12} {m['total_trades']:>7} {m['win_rate']*100:>5.0f}% {m['total_pnl_pips']:>+9.1f} {m['profit_factor']:>6.2f}")
    print("-" * 50)
    print(f"{'TOTAL':<12} {total_trades:>7} {'':>6} {total_pnl:>+9.1f}")

    months = 19.0
    pips_per_month = total_pnl / months
    print(f"\nPer Month (avg): {pips_per_month:+.0f} pips/month")
    print(f"Target: 4,000 pips/month")
    print(f"Progress: {pips_per_month / 4000 * 100:.0f}%")

    log_result("COMBINED: XAUUSD + US30 + US100", {
        "total_trades": total_trades,
        "win_rate": np.mean([m["win_rate"] for m in all_results.values()]),
        "total_pnl_pips": total_pnl,
        "profit_factor": np.mean([m["profit_factor"] for m in all_results.values()]),
        "rr": np.mean([m["rr"] for m in all_results.values()]),
        "max_drawdown_pips": max(m["max_drawdown_pips"] for m in all_results.values()),
        "instruments": {n: {"trades": m["total_trades"], "wr": m["win_rate"], "pnl": m["total_pnl_pips"]} for n, m in all_results.items()}
    }, "XAUUSD ICT/SMC + US30/US100 Supertrend — running simultaneously")

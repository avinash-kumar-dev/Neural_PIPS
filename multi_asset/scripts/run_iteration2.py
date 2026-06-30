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
    df["adx"] = adx; df["rsi"] = rsi; df["atr"] = atr
    return df


def make_signals(df, session_start, session_end, max_per_day, min_gap, adx_min, rsi_long_max, rsi_short_min):
    dt = pd.to_datetime(df["datetime"])
    hours = dt.dt.hour.values
    in_session = (hours >= session_start) & (hours < session_end)

    el = df["long_trend"].values & df["long_conf"].values & (df["adx"].values > adx_min) & (df["rsi"].values < rsi_long_max) & in_session
    es = df["short_trend"].values & df["short_conf"].values & (df["adx"].values > adx_min) & (df["rsi"].values > rsi_short_min) & in_session

    combined = el | es
    indices = np.where(combined)[0]

    filtered = []
    last_idx = -min_gap - 1
    day_counts = {}
    dates = dt.dt.date.values
    for idx in indices:
        d = dates[idx]
        if d not in day_counts:
            day_counts[d] = 0
        if (idx - last_idx) < min_gap or day_counts[d] >= max_per_day:
            continue
        filtered.append(idx)
        last_idx = idx
        day_counts[d] += 1

    n = len(df)
    el_f = np.zeros(n, dtype=bool)
    es_f = np.zeros(n, dtype=bool)
    sl_l = np.full(n, np.nan)
    sl_s = np.full(n, np.nan)
    atr = df["atr"].values
    lo = df["low"].values; hi = df["high"].values

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


if __name__ == "__main__":
    print("Pre-computing indicators for all instruments...")
    t_total = time.time()
    data = {}
    for name in ["US30", "US100", "EURUSD", "GBPUSD", "USDJPY"]:
        t0 = time.time()
        df = pd.read_parquet(Path(DATA_DIR) / f"{name}_M5.parquet")
        df = precompute(df)
        data[name] = df
        print(f"  {name}: {time.time()-t0:.1f}s")

    configs = [
        {"label": "V1: 3/day, gap6, adx20", "max_day": 3, "gap": 6, "adx": 20, "rsi_l": 70, "rsi_s": 30},
        {"label": "V2: 3/day, gap6, adx25", "max_day": 3, "gap": 6, "adx": 25, "rsi_l": 65, "rsi_s": 35},
        {"label": "V3: 5/day, gap4, adx25", "max_day": 5, "gap": 4, "adx": 25, "rsi_l": 65, "rsi_s": 35},
        {"label": "V4: 2/day, gap10, adx30", "max_day": 2, "gap": 10, "adx": 30, "rsi_l": 60, "rsi_s": 40},
    ]

    sessions = {"US30": (13, 21), "US100": (13, 21), "EURUSD": (6, 21), "GBPUSD": (6, 21), "USDJPY": (0, 21)}
    pip_vals = {"US30": 1.0, "US100": 1.0, "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01}

    with open(RESULTS_FILE, "a") as f:
        f.write("\n\n---\n\n# Iteration 2: Vectorized Signals + Filters\n\n")

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Config: {cfg['label']}")
        print(f"{'='*50}")

        results = {}
        for name in ["US30", "US100", "EURUSD", "GBPUSD", "USDJPY"]:
            ss, se = sessions[name]
            pv = pip_vals[name]
            df = data[name].copy()
            df = make_signals(df, ss, se, cfg["max_day"], cfg["gap"], cfg["adx"], cfg["rsi_l"], cfg["rsi_s"])

            lc = int(df["entry_long"].sum())
            sc = int(df["entry_short"].sum())
            if lc + sc == 0:
                continue

            signals = df[["entry_long", "entry_short", "entry_sl_long", "entry_sl_short",
                           "setup_long", "setup_short"]].copy()
            signals["pip_value"] = pv

            engine = MultiAssetEngine(EngineConfig(
                spread_pips=1.0, slippage_pips=0.5, min_rr=2.0,
                breakeven_trigger_r=2.0, trailing_start_r=3.0, trailing_step_r=1.0,
            ))
            trades = engine.run({name: df}, {name: signals})
            m = compute_metrics(trades)
            results[name] = m
            print(f"  {name}: {lc}L/{sc}S → {m['total_trades']} trades, "
                  f"WR {m['win_rate']*100:.0f}%, PnL {m['total_pnl_pips']:+.1f}, PF {m['profit_factor']:.2f}")

        total_pnl = sum(m["total_pnl_pips"] for m in results.values())
        total_trades = sum(m["total_trades"] for m in results.values())
        avg_wr = np.mean([m["win_rate"] for m in results.values()]) * 100 if results else 0
        avg_pf = np.mean([m["profit_factor"] for m in results.values()]) if results else 0
        print(f"\n  TOTAL: {total_trades} trades, {total_pnl:+.1f} pips, Avg WR {avg_wr:.0f}%, Avg PF {avg_pf:.2f}")

        log_result(f"V2 Total ({cfg['label']})", {
            "total_trades": total_trades, "win_rate": avg_wr / 100, "total_pnl_pips": total_pnl,
            "profit_factor": avg_pf, "rr": 0, "max_drawdown_pips": 0,
            "instruments": {n: {"trades": m["total_trades"], "wr": m["win_rate"], "pnl": m["total_pnl_pips"]} for n, m in results.items()}
        }, cfg["label"])

    print(f"\nTotal time: {time.time()-t_total:.1f}s")

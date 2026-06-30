import sys
sys.path.insert(0, "/home/avinash-kumar/tradding_app")

import pandas as pd
import numpy as np
from pathlib import Path
from multi_asset.data.indicators import compute_basic_indicators
from multi_asset.setups.po3 import detect_po3_signals
from multi_asset.backtest.engine import MultiAssetEngine, EngineConfig, compute_metrics
import time

RESULTS_FILE = "/home/avinash-kumar/tradding_app/multi_asset/RESULTS_LOG.md"
DATA_DIR = "/home/avinash-kumar/tradding_app/multi_asset/data/raw"


if __name__ == "__main__":
    with open(RESULTS_FILE, "a") as f:
        f.write("\n\n---\n\n# PO3 Setup — Iteration 1\n\n")

    configs = [
        {"label": "V1: range20, sweep0.3, range5", "range_lb": 20, "min_range": 0.003, "sweep_atr": 0.3, "min_bars": 5},
        {"label": "V2: range15, sweep0.2, range5", "range_lb": 15, "min_range": 0.002, "sweep_atr": 0.2, "min_bars": 5},
        {"label": "V3: range20, sweep0.5, range10", "range_lb": 20, "min_range": 0.003, "sweep_atr": 0.5, "min_bars": 10},
    ]

    sessions = {"US30": (13, 21), "US100": (13, 21), "EURUSD": (6, 21), "GBPUSD": (6, 21), "USDJPY": (0, 21)}
    pip_vals = {"US30": 1.0, "US100": 1.0, "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01}

    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"PO3 Config: {cfg['label']}")
        print(f"{'='*50}")

        results = {}
        for name in ["US30", "US100", "EURUSD", "GBPUSD", "USDJPY"]:
            t0 = time.time()
            df = pd.read_parquet(Path(DATA_DIR) / f"{name}_M5.parquet")
            df = compute_basic_indicators(df, atr_period=14)

            df = detect_po3_signals(df,
                                     range_lookback=cfg["range_lb"],
                                     min_range_pct=cfg["min_range"],
                                     sweep_min_atr=cfg["sweep_atr"],
                                     min_range_bars=cfg["min_bars"])

            ss, se = sessions[name]
            dt = pd.to_datetime(df["datetime"])
            hours = dt.dt.hour.values
            in_session = (hours >= ss) & (hours < se)
            df["entry_long"] = df["entry_long"].values & in_session
            df["entry_short"] = df["entry_short"].values & in_session

            lc = int(df["entry_long"].sum())
            sc = int(df["entry_short"].sum())

            if lc + sc == 0:
                print(f"  {name}: 0 signals")
                continue

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
            print(f"  {name}: {lc}L/{sc}S → {m['total_trades']} trades, "
                  f"WR {m['win_rate']*100:.0f}%, PnL {m['total_pnl_pips']:+.1f}, "
                  f"PF {m['profit_factor']:.2f}, R:R {m['rr']:.2f} ({time.time()-t0:.1f}s)")

        total = sum(m["total_pnl_pips"] for m in results.values())
        total_trades = sum(m["total_trades"] for m in results.values())
        avg_wr = np.mean([m["win_rate"] for m in results.values()]) * 100 if results else 0
        avg_pf = np.mean([m["profit_factor"] for m in results.values()]) if results else 0
        print(f"\n  TOTAL: {total_trades} trades, {total:+.1f} pips, Avg WR {avg_wr:.0f}%, Avg PF {avg_pf:.2f}")

        with open(RESULTS_FILE, "a") as f:
            f.write(f"\n### {cfg['label']}\n\n")
            f.write(f"| Inst | Trades | WR | PnL | PF | R:R |\n|------|--------|-----|-----|-----|-----|\n")
            for n, m in results.items():
                f.write(f"| {n} | {m['total_trades']} | {m['win_rate']*100:.0f}% | {m['total_pnl_pips']:+.1f} | {m['profit_factor']:.2f} | {m['rr']:.2f} |\n")
            f.write(f"| **TOTAL** | **{total_trades}** | | **{total:+.1f}** | **{avg_pf:.2f}** | |\n\n")

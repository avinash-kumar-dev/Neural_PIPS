import sys
sys.path.insert(0, "/home/avinash-kumar/tradding_app")

import pandas as pd
import numpy as np
from pathlib import Path
from multi_asset.data.fetcher import MultiAssetFetcher
from multi_asset.data.indicators import compute_basic_indicators
from multi_asset.setups.orb import detect_orb_signals, ORBConfig
from multi_asset.backtest.engine import MultiAssetEngine, EngineConfig, compute_metrics

RESULTS_FILE = "/home/avinash-kumar/tradding_app/multi_asset/RESULTS_LOG.md"
DATA_DIR = "/home/avinash-kumar/tradding_app/multi_asset/data/raw"


def log_result(name: str, metrics: dict, config_note: str = ""):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n## {name}\n")
        f.write(f"> Config: {config_note}\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total Trades | {metrics['total_trades']} |\n")
        f.write(f"| Win Rate | {metrics['win_rate']*100:.1f}% |\n")
        f.write(f"| Total PnL | {metrics['total_pnl_pips']:+.1f} pips |\n")
        f.write(f"| Profit Factor | {metrics['profit_factor']:.2f} |\n")
        f.write(f"| R:R | {metrics['rr']:.2f} |\n")
        f.write(f"| Max Drawdown | {metrics['max_drawdown_pips']:.1f} pips |\n")
        f.write(f"| Avg Win | {metrics['avg_win']:.1f} pips |\n")
        f.write(f"| Avg Loss | {metrics['avg_loss']:.1f} pips |\n")
        if metrics.get("instruments"):
            f.write(f"\n### Per Instrument\n\n")
            f.write(f"| Instrument | Trades | WR | PnL |\n|------------|--------|-----|-----|\n")
            for inst, stats in metrics["instruments"].items():
                f.write(f"| {inst} | {stats['trades']} | {stats['wr']*100:.0f}% | {stats['pnl']:+.1f} |\n")
        f.write("\n---\n")


def fetch_all_data():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    instruments = [
        ("US30", "US30", "M5", 50000),
        ("USTECH100M", "US100", "M5", 50000),
        ("EURUSD", "EURUSD", "M5", 50000),
        ("GBPUSD", "GBPUSD", "M5", 50000),
        ("USDJPY", "USDJPY", "M5", 50000),
    ]

    with MultiAssetFetcher() as fetcher:
        for symbol, name, tf, bars in instruments:
            path = Path(DATA_DIR) / f"{name}_{tf}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                print(f"  {name}: loaded from cache ({len(df)} bars)")
            else:
                print(f"  Fetching {name} ({symbol}) {tf}...")
                df = fetcher.fetch_ohlcv(symbol, tf, bars)
                df.to_parquet(path, index=False)
                print(f"  {name}: saved {len(df)} bars")


def run_orb_on_instrument(name: str, pip_value: float, session_hour: int, session_min: int, range_min: int):
    path = Path(DATA_DIR) / f"{name}_M5.parquet"
    df = pd.read_parquet(path)

    df = compute_basic_indicators(df, atr_period=14)

    cfg = ORBConfig(range_minutes=range_min)
    df = detect_orb_signals(df, session_start_hour=session_hour, session_start_min=session_min, cfg=cfg)

    long_count = df["orb_long"].sum()
    short_count = df["orb_short"].sum()

    if long_count == 0 and short_count == 0:
        return {"total_trades": 0, "wins": 0, "losses": 0, "be": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "rr": 0, "total_pnl_pips": 0,
                "profit_factor": 0, "max_drawdown_pips": 0, "instruments": {}}

    signals = df.copy()
    signals["entry_long"] = df["orb_long"]
    signals["entry_short"] = df["orb_short"]
    signals["entry_sl_long"] = df["orb_sl_long"]
    signals["entry_sl_short"] = df["orb_sl_short"]
    signals["setup_long"] = "orb"
    signals["setup_short"] = "orb"
    signals["pip_value"] = pip_value

    engine = MultiAssetEngine(EngineConfig(
        spread_pips=1.0,
        slippage_pips=0.5,
        min_rr=2.0,
        breakeven_trigger_r=2.0,
        trailing_start_r=3.0,
        trailing_step_r=1.0,
    ))

    trades = engine.run({name: df}, {name: signals})
    metrics = compute_metrics(trades)
    return metrics


if __name__ == "__main__":
    with open(RESULTS_FILE, "w") as f:
        f.write("# Multi-Asset A1 Strategy — Iteration Results\n\n")
        f.write(f"> Started: {pd.Timestamp.now()}\n\n---\n")

    print("Phase 1: Fetching all data...")
    fetch_all_data()

    tests = [
        ("US30", 1.0, 13, 30, 30),
        ("US100", 1.0, 13, 30, 30),
        ("EURUSD", 0.0001, 7, 0, 60),
        ("GBPUSD", 0.0001, 8, 0, 60),
        ("USDJPY", 0.01, 0, 0, 60),
    ]

    results = {}
    for name, pip_val, sess_h, sess_m, range_m in tests:
        print(f"\n{'='*60}")
        print(f"ORB on {name} (session {sess_h}:{sess_m:02d}, {range_m}-min range)")
        print(f"{'='*60}")
        try:
            m = run_orb_on_instrument(name, pip_val, sess_h, sess_m, range_m)
            results[name] = m
            print(f"  Trades: {m['total_trades']}, WR: {m['win_rate']*100:.0f}%, PnL: {m['total_pnl_pips']:+.1f}, PF: {m['profit_factor']:.2f}")
            log_result(f"ORB {name} M5 (Iteration 1)", m,
                       f"{range_m}-min OR, session {sess_h}:{sess_m:02d}, BE=2R, Trail=3R, pip={pip_val}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUMMARY — ORB Strategy Iteration 1")
    print(f"{'='*60}")
    print(f"{'Instrument':<12} {'Trades':>7} {'WR':>6} {'PnL':>10} {'PF':>6} {'R:R':>6}")
    print("-" * 55)
    for name, m in results.items():
        print(f"{name:<12} {m['total_trades']:>7} {m['win_rate']*100:>5.0f}% {m['total_pnl_pips']:>+9.1f} {m['profit_factor']:>6.2f} {m['rr']:>6.2f}")

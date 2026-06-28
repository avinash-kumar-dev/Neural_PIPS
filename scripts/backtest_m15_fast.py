#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import time
from xauusd.structure.layer1 import compute_layer1
from xauusd.entry.layer2 import compute_layer2
from xauusd.confirmation.layer3 import compute_layer3
from xauusd.backtest.engine import BacktestEngine, BacktestConfig, compute_metrics

PIP_VALUE = 0.01


def run_test(df, params, label=""):
    t0 = time.time()
    l1 = compute_layer1(df, swing_lookback=params.get("swing_lb", 5))
    l2 = compute_layer2(df, l1["layer1_bias"],
                        displacement_mult=params.get("disp", 1.2),
                        ob_max_age=params.get("ob_age", 15),
                        fvg_min_size=params.get("fvg_min", 0.3),
                        rsi_oversold=params.get("rsi_os", 35),
                        rsi_overbought=params.get("rsi_ob", 65))
    l3 = compute_layer3(l2, volume_ma_period=params.get("vol_ma", 15),
                        body_ratio_min=params.get("body_min", 0.5))

    longs = (l3["long_entry"] & l3["layer3_confirmed"]).sum()
    shorts = (l3["short_entry"] & l3["layer3_confirmed"]).sum()

    config = BacktestConfig(
        spread_pips=params.get("spread", 0.15),
        commission_per_lot=params.get("comm", 7.0),
        slippage_pips=params.get("slip", 1.5),
        risk_per_trade=0.03,
        min_rr=params.get("min_rr", 2.0),
    )
    engine = BacktestEngine(config)
    trades = engine.run(df, l3)
    m = compute_metrics(trades)

    months = (df["datetime"].max() - df["datetime"].min()).days / 30
    tpm = m["total_trades"] / months if months > 0 else 0
    ppm = m.get("total_pnl_pips", 0) / months if months > 0 else 0
    elapsed = time.time() - t0

    print(f"{label:40s} | {m['total_trades']:4d} trades ({tpm:4.0f}/mo) | "
          f"WR {m.get('win_rate', 0):.0%} | R:R {m.get('avg_rr', 0):.1f} | "
          f"PnL {m.get('total_pnl_pips', 0):8.0f}p ({ppm:5.0f}/mo) | "
          f"PF {m.get('profit_factor', 0):.2f} | {elapsed:.1f}s")
    return m, ppm


def main():
    df_full = pd.read_parquet("xauusd/data/raw/xauusd_m15.parquet")

    # Use 1 year for speed
    cutoff = df_full["datetime"].max() - pd.Timedelta(days=365)
    df = df_full[df_full["datetime"] >= cutoff].copy().reset_index(drop=True)
    print(f"M15 data: {len(df)} bars, {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Months: {(df['datetime'].max() - df['datetime'].min()).days / 30:.1f}")
    print()

    configs = [
        {"swing_lb": 3, "disp": 1.0, "ob_age": 20, "fvg_min": 0.2, "rsi_os": 35, "rsi_ob": 65, "vol_ma": 15, "body_min": 0.4, "min_rr": 2.0, "slip": 1.5, "label": "Aggressive M15"},
        {"swing_lb": 3, "disp": 1.0, "ob_age": 20, "fvg_min": 0.2, "rsi_os": 35, "rsi_ob": 65, "vol_ma": 15, "body_min": 0.4, "min_rr": 1.5, "slip": 1.5, "label": "Relaxed R:R (1.5)"},
        {"swing_lb": 3, "disp": 1.0, "ob_age": 25, "fvg_min": 0.2, "rsi_os": 35, "rsi_ob": 65, "vol_ma": 10, "body_min": 0.4, "min_rr": 2.0, "slip": 1.5, "label": "Wide OB age + fast vol"},
        {"swing_lb": 2, "disp": 0.8, "ob_age": 25, "fvg_min": 0.15, "rsi_os": 40, "rsi_ob": 60, "vol_ma": 10, "body_min": 0.35, "min_rr": 2.0, "slip": 1.5, "label": "Ultra aggressive"},
        {"swing_lb": 2, "disp": 0.8, "ob_age": 25, "fvg_min": 0.15, "rsi_os": 40, "rsi_ob": 60, "vol_ma": 10, "body_min": 0.35, "min_rr": 1.5, "slip": 1.5, "label": "Ultra + 1.5RR"},
        {"swing_lb": 5, "disp": 1.2, "ob_age": 20, "fvg_min": 0.3, "rsi_os": 30, "rsi_ob": 70, "vol_ma": 20, "body_min": 0.5, "min_rr": 2.0, "slip": 2.0, "label": "Conservative M15"},
    ]

    print(f"{'Config':40s} | {'Trades':>8s}    | Metrics...")
    print("-" * 130)

    best_ppm = 0
    best_cfg = None
    for cfg in configs:
        label = cfg.pop("label")
        m, ppm = run_test(df, cfg, label)
        cfg["label"] = label
        if ppm > best_ppm and m['total_trades'] > 0:
            best_ppm = ppm
            best_cfg = cfg

    if best_cfg:
        print(f"\nBest: {best_cfg.get('label', '?')} at {best_ppm:.0f} pips/month")


if __name__ == "__main__":
    main()

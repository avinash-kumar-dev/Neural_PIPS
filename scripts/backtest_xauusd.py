#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

import pandas as pd
from xauusd.strategies.trend_structure import run_trend_strategy
from xauusd.backtest.engine import BacktestEngine, BacktestConfig, compute_metrics


def main():
    df = pd.read_parquet("xauusd/data/raw/xauusd_h4.parquet")
    print(f"Data: {len(df)} bars, {df['datetime'].min()} to {df['datetime'].max()}")

    signals = run_trend_strategy(df, swing_lookback=3)
    long_entries = (signals["long_entry"] & signals["layer3_confirmed"]).sum()
    short_entries = (signals["short_entry"] & signals["layer3_confirmed"]).sum()
    print(f"Signals: {long_entries} long, {short_entries} short")

    config = BacktestConfig(
        spread_pips=0.15,
        commission_per_lot=7.0,
        slippage_pips=2.0,
        risk_per_trade=0.03,
        min_rr=2.0,
        initial_equity=100000.0,
    )

    engine = BacktestEngine(config)
    trades = engine.run(df, signals)
    metrics = compute_metrics(trades)

    print(f"\n=== Backtest Results (H4, 3-year) ===")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Wins: {metrics['wins']}, Losses: {metrics['losses']}, Open: {metrics['open']}")
    print(f"Win rate: {metrics['win_rate']:.1%}")
    print(f"Avg win: {metrics['avg_win_pips']:.1f} pips")
    print(f"Avg loss: {metrics['avg_loss_pips']:.1f} pips")
    print(f"Avg R:R: {metrics['avg_rr']:.2f}")
    print(f"Total PnL: {metrics['total_pnl_pips']:.1f} pips")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown_pips']:.1f} pips")

    print(f"\nSample trades:")
    for t in trades[:5]:
        print(f"  {t.entry_time} {'LONG' if t.direction == 1 else 'SHORT'} "
              f"entry={t.entry_price:.2f} sl={t.sl:.2f} tp={t.tp2:.2f} "
              f"trigger={t.trigger} outcome={t.outcome} pnl={t.pnl_pips:.1f}p")


if __name__ == "__main__":
    main()

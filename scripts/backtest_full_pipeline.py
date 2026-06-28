import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xauusd.pipeline import run_full_pipeline
from xauusd.backtest.engine import BacktestEngine, BacktestConfig, compute_metrics


def run_backtest(
    data_path: str,
    min_confluence: float = 70.0,
    min_rr: float = 2.0,
    initial_equity: float = 1000.0,
    risk_per_trade: float = 0.03,
) -> dict:
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} bars")

    if "datetime" not in df.columns:
        if "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["time"], unit="s")
        elif isinstance(df.index, pd.DatetimeIndex):
            df["datetime"] = df.index
            df = df.reset_index(drop=True)

    print(f"Running full pipeline with min_confluence={min_confluence}...")
    signals = run_full_pipeline(df, min_confluence=min_confluence, min_rr=min_rr)

    long_signals = signals["long_entry"].sum()
    short_signals = signals["short_entry"].sum()
    print(f"Generated {long_signals} LONG + {short_signals} SHORT = {long_signals + short_signals} total signals")

    config = BacktestConfig(
        spread_pips=0.15,
        commission_per_lot=7.0,
        slippage_pips=2.0,
        risk_per_trade=risk_per_trade,
        min_rr=min_rr,
        initial_equity=initial_equity,
    )

    engine = BacktestEngine(config)
    trades = engine.run(df, signals)

    metrics = compute_metrics(trades)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Wins: {metrics['wins']}")
    print(f"Losses: {metrics['losses']}")
    print(f"Open: {metrics['open']}")
    print(f"Win rate: {metrics['win_rate']:.1%}")
    print(f"Avg win: {metrics['avg_win_pips']:.1f} pips")
    print(f"Avg loss: {metrics['avg_loss_pips']:.1f} pips")
    print(f"Avg R:R: {metrics['avg_rr']:.2f}")
    print(f"Total PnL: {metrics['total_pnl_pips']:.1f} pips")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown_pips']:.1f} pips")
    print("=" * 60)

    if trades:
        print("\nTrade details:")
        for t in trades[:20]:
            print(f"  {t.entry_time} | {'LONG' if t.direction == 1 else 'SHORT':5s} | "
                  f"Entry: {t.entry_price:.2f} | SL: {t.sl:.2f} | "
                  f"TP2: {t.tp2:.2f} | {t.outcome:4s} | {t.pnl_pips:+.1f}p | "
                  f"Trigger: {t.trigger} | Lots: {t.lots}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="xauusd/data/raw/xauusd_m15.parquet")
    parser.add_argument("--min-confluence", type=float, default=70.0)
    parser.add_argument("--min-rr", type=float, default=2.0)
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--risk", type=float, default=0.03)
    args = parser.parse_args()

    run_backtest(
        data_path=args.data,
        min_confluence=args.min_confluence,
        min_rr=args.min_rr,
        initial_equity=args.equity,
        risk_per_trade=args.risk,
    )

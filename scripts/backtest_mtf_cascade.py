import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xauusd.pipeline_mtf import run_mtf_pipeline
from xauusd.backtest.engine_mtf import MTFBacktestEngine, MTFBacktestConfig, compute_mtf_metrics


def run_backtest(
    m15_path: str,
    h4_path: str,
    h1_path: str,
    m5_path: str = None,
    m1_path: str = None,
    min_confluence: float = 35.0,
    initial_equity: float = 1000.0,
    risk_per_trade: float = 0.03,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    print(f"Loading M15 data from {m15_path}...")
    df_m15 = pd.read_parquet(m15_path)
    print(f"  Loaded {len(df_m15)} M15 bars")

    print(f"Loading H4 data from {h4_path}...")
    df_h4 = pd.read_parquet(h4_path)
    print(f"  Loaded {len(df_h4)} H4 bars")

    print(f"Loading H1 data from {h1_path}...")
    df_h1 = pd.read_parquet(h1_path)
    print(f"  Loaded {len(df_h1)} H1 bars")

    df_m5 = None
    if m5_path:
        print(f"Loading M5 data from {m5_path}...")
        df_m5 = pd.read_parquet(m5_path)
        print(f"  Loaded {len(df_m5)} M5 bars")

    df_m1 = None
    if m1_path:
        print(f"Loading M1 data from {m1_path}...")
        df_m1 = pd.read_parquet(m1_path)
        print(f"  Loaded {len(df_m1)} M1 bars")

    for df, name in [(df_m15, "M15"), (df_h4, "H4"), (df_h1, "H1")]:
        if "datetime" not in df.columns:
            if "time" in df.columns:
                df["datetime"] = pd.to_datetime(df["time"], unit="s")
            elif isinstance(df.index, pd.DatetimeIndex):
                df["datetime"] = df.index
                df = df.reset_index(drop=True)

    if start_date and end_date:
        dt_m15 = pd.to_datetime(df_m15["datetime"])
        mask = (dt_m15 >= start_date) & (dt_m15 <= end_date)
        df_m15 = df_m15[mask].reset_index(drop=True)
        print(f"  Filtered M15 to {len(df_m15)} bars ({start_date} to {end_date})")

    print(f"\nRunning MTF pipeline (H4→H1→M15) with min_confluence={min_confluence}...")
    signals = run_mtf_pipeline(
        df_m15=df_m15,
        df_h4=df_h4,
        df_h1=df_h1,
        df_m5=df_m5,
        df_m1=df_m1,
        min_confluence=min_confluence,
    )

    long_signals = signals["long_entry"].sum()
    short_signals = signals["short_entry"].sum()
    print(f"Generated {long_signals} LONG + {short_signals} SHORT = {long_signals + short_signals} total signals")

    config = MTFBacktestConfig(
        spread_pips=0.15,
        commission_per_lot=7.0,
        slippage_pips=2.0,
        risk_per_trade=risk_per_trade,
        initial_equity=initial_equity,
        min_sl_pips=30.0,
        max_sl_pips=200.0,
        breakeven_trigger_r=1.5,
        trailing_start_r=2.0,
        trailing_step_r=1.0,
        trailing_tp_ratio=2.0,
        max_bars=300,
        stale_exit_bars=100,
    )

    engine = MTFBacktestEngine(config)
    trades = engine.run(df_m15, signals)

    metrics = compute_mtf_metrics(trades)

    print("\n" + "=" * 60)
    print("MTF BACKTEST RESULTS (H4→H1→M15 Cascade + Trailing TP)")
    print("=" * 60)
    print(f"Total trades: {metrics.get('total_trades', 0)}")
    print(f"Wins: {metrics.get('wins', 0)}")
    print(f"Losses: {metrics.get('losses', 0)}")
    print(f"BE: {metrics.get('be', 0)}")
    print(f"Open: {metrics.get('open', 0)}")
    print(f"Win rate: {metrics.get('win_rate', 0):.1%}")
    print(f"Avg win: {metrics.get('avg_win_pips', 0):.1f} pips")
    print(f"Avg loss: {metrics.get('avg_loss_pips', 0):.1f} pips")
    print(f"Avg R:R: {metrics.get('avg_rr', 0):.2f}")
    print(f"Total PnL: {metrics.get('total_pnl_pips', 0):.1f} pips")
    print(f"Profit factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Max drawdown: {metrics.get('max_drawdown_pips', 0):.1f} pips")
    print(f"Avg bars held: {metrics.get('avg_bars_held', 0):.0f}")
    print("=" * 60)

    if trades:
        print("\nTrade details (first 20):")
        for t in trades[:20]:
            print(f"  {t.entry_time} | {'LONG' if t.direction == 1 else 'SHORT':5s} | "
                  f"Entry: {t.entry_price:.2f} | SL: {t.sl:.2f} | "
                  f"TP: {t.tp:.2f} | {t.outcome:4s} | {t.pnl_pips:+.1f}p | "
                  f"Trigger: {t.trigger} | Lots: {t.lots}")

        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]
        bes = [t for t in trades if t.outcome == "BE"]
        if wins:
            max_win = max(wins, key=lambda t: t.pnl_pips)
            print(f"\nLargest WIN: {max_win.pnl_pips:+.1f}p ({max_win.trigger}, held {max_win.bars_held} bars)")
        if losses:
            max_loss = max(losses, key=lambda t: abs(t.pnl_pips))
            print(f"Largest LOSS: {max_loss.pnl_pips:+.1f}p ({max_loss.trigger}, held {max_loss.bars_held} bars)")
        if bes:
            print(f"BE trades: {len(bes)} (avg {np.mean([t.pnl_pips for t in bes]):+.1f}p)")

        total_days = (pd.to_datetime(trades[-1].entry_time) - pd.to_datetime(trades[0].entry_time)).days
        if total_days > 0:
            trades_per_month = metrics.get('total_trades', 0) / (total_days / 30.44)
            pips_per_month = metrics.get('total_pnl_pips', 0) / (total_days / 30.44)
            print(f"\n--- RATE ---")
            print(f"Trades/month: {trades_per_month:.1f}")
            print(f"Pips/month: {pips_per_month:.0f}")
            print(f"Period: {total_days} days ({trades[0].entry_time.date()} to {trades[-1].entry_time.date()})")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m15", default="xauusd/data/raw/xauusd_m15.parquet")
    parser.add_argument("--h4", default="xauusd/data/raw/xauusd_h4.parquet")
    parser.add_argument("--h1", default="xauusd/data/raw/xauusd_h1.parquet")
    parser.add_argument("--m5", default=None)
    parser.add_argument("--m1", default=None)
    parser.add_argument("--min-confluence", type=float, default=35.0)
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--risk", type=float, default=0.03)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    run_backtest(
        m15_path=args.m15,
        h4_path=args.h4,
        h1_path=args.h1,
        m5_path=args.m5,
        m1_path=args.m1,
        min_confluence=args.min_confluence,
        initial_equity=args.equity,
        risk_per_trade=args.risk,
        start_date=args.start,
        end_date=args.end,
    )

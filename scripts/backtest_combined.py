#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

import pandas as pd
from xauusd.strategies.voting import compute_voting_signals
from xauusd.strategies.trend_structure import run_trend_strategy
from xauusd.strategies.breakout_retest import compute_breakout_retest
from xauusd.strategies.liquidity_sweep import compute_liquidity_sweep
from xauusd.backtest.engine import BacktestEngine, BacktestConfig, compute_metrics


def run_backtest(df, signals, label):
    config = BacktestConfig(
        spread_pips=0.15, commission_per_lot=7.0,
        slippage_pips=2.0, risk_per_trade=0.03, min_rr=2.0,
    )
    engine = BacktestEngine(config)
    trades = engine.run(df, signals)
    metrics = compute_metrics(trades)
    print(f"\n=== {label} ===")
    print(f"Trades: {metrics['total_trades']} (W:{metrics['wins']} L:{metrics['losses']} O:{metrics['open']})")
    print(f"Win rate: {metrics['win_rate']:.1%}")
    print(f"Avg R:R: {metrics['avg_rr']:.2f}")
    print(f"Total PnL: {metrics['total_pnl_pips']:.1f} pips")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown_pips']:.1f} pips")
    return metrics


def main():
    df = pd.read_parquet("xauusd/data/raw/xauusd_h4.parquet")
    print(f"Data: {len(df)} bars, {df['datetime'].min()} to {df['datetime'].max()}")

    trend_signals = run_trend_strategy(df)
    trend_long = (trend_signals["long_entry"] & trend_signals["layer3_confirmed"]).sum()
    trend_short = (trend_signals["short_entry"] & trend_signals["layer3_confirmed"]).sum()
    print(f"Trend module: {trend_long} long, {trend_short} short")

    breakout = compute_breakout_retest(df)
    br_long = breakout["retest_bull"].sum()
    br_short = breakout["retest_bear"].sum()
    print(f"Breakout module: {br_long} long, {br_short} short")

    sweep = compute_liquidity_sweep(df)
    ls_long = sweep["sweep_high_signal"].sum()
    ls_short = sweep["sweep_low_signal"].sum()
    print(f"Liquidity sweep module: {ls_long} long, {ls_short} short")

    voting = compute_voting_signals(df, min_agreement=2)
    vote_long = voting["long_entry"].sum()
    vote_short = voting["short_entry"].sum()
    print(f"Combined voting (2/3): {vote_long} long, {vote_short} short")

    run_backtest(df, trend_signals, "Trend Module Only")
    m = run_backtest(df, voting, "Combined (Voting 2/3)")


if __name__ == "__main__":
    main()

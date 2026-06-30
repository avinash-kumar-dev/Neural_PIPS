import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Trade:
    instrument: str = ""
    direction: int = 0
    entry_time: pd.Timestamp = None
    entry_price: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    exit_time: pd.Timestamp = None
    exit_price: float = 0.0
    pnl_pips: float = 0.0
    outcome: str = "OPEN"
    setup: str = ""
    rr: float = 0.0
    bars_held: int = 0
    costs_pips: float = 0.0
    pip_value: float = 0.0001


@dataclass
class EngineConfig:
    spread_pips: float = 0.5
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0
    risk_per_trade_pct: float = 1.0
    initial_equity: float = 10000.0
    min_rr: float = 2.0
    breakeven_trigger_r: float = 2.0
    trailing_start_r: float = 3.0
    trailing_step_r: float = 1.0
    max_bars: int = 300
    stale_exit_bars: int = 80
    max_concurrent: int = 3


class MultiAssetEngine:
    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()

    def run(self, instruments_data: dict, signals: dict) -> list[Trade]:
        all_trades = []
        equity = self.config.initial_equity

        max_idx = max(len(df) for df in instruments_data.values())

        for i in range(max_idx):
            active = [t for t in all_trades if t.outcome == "OPEN"]

            for name, df in instruments_data.items():
                if i >= len(df):
                    continue
                bar = df.iloc[i]

                for trade in [t for t in active if t.instrument == name]:
                    self._manage_trade(trade, bar, equity)

                if name not in signals:
                    continue
                sig_df = signals[name]
                if i >= len(sig_df):
                    continue
                sig = sig_df.iloc[i]

                active_count = sum(1 for t in all_trades if t.outcome == "OPEN")
                if active_count >= self.config.max_concurrent:
                    continue

                pip_val = 0.0001
                if "pip_value" in sig.index:
                    pip_val = sig["pip_value"]
                elif name in instruments_data:
                    pip_val = 0.01 if "US" in name else 0.0001

                for direction in [1, -1]:
                    entry_key = "entry_long" if direction == 1 else "entry_short"
                    sl_key = "entry_sl_long" if direction == 1 else "entry_sl_short"
                    setup_key = "setup_long" if direction == 1 else "setup_short"

                    if not sig.get(entry_key, False):
                        continue

                    sl = sig.get(sl_key, np.nan)
                    if pd.isna(sl) or sl == 0:
                        continue

                    entry_price = bar["close"]
                    sl_dist = abs(entry_price - sl)
                    sl_pips = sl_dist / pip_val

                    if sl_pips < 5 or sl_pips > 500:
                        continue

                    costs = self.config.spread_pips + self.config.slippage_pips + self.config.commission_per_lot / 100
                    tp1 = entry_price + direction * sl_dist * 2.0
                    tp2 = entry_price + direction * sl_dist * 3.0

                    trade = Trade(
                        instrument=name,
                        direction=direction,
                        entry_time=bar["datetime"],
                        entry_price=entry_price + direction * self.config.slippage_pips * pip_val,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        setup=sig.get(setup_key, "unknown"),
                        costs_pips=costs,
                        rr=2.0,
                        pip_value=pip_val,
                    )
                    all_trades.append(trade)

        for t in all_trades:
            if t.outcome == "OPEN":
                t.outcome = "OPEN"

        return all_trades

    def _manage_trade(self, trade: Trade, bar: pd.Series, equity: float):
        trade.bars_held += 1
        sl_dist = abs(trade.entry_price - trade.sl)
        if sl_dist == 0:
            return

        pip_val = trade.pip_value

        if trade.direction == 1:
            mfe = bar["high"] - trade.entry_price
            mfe_r = mfe / sl_dist

            if mfe_r >= self.config.breakeven_trigger_r:
                new_sl = trade.entry_price + pip_val
                if new_sl > trade.sl:
                    trade.sl = new_sl

            if mfe_r >= self.config.trailing_start_r:
                new_sl = trade.entry_price + (mfe_r - self.config.trailing_step_r) * sl_dist
                if new_sl > trade.sl:
                    trade.sl = new_sl

            if bar["low"] <= trade.sl:
                self._close_trade(trade, bar)

        else:
            mfe = trade.entry_price - bar["low"]
            mfe_r = mfe / sl_dist

            if mfe_r >= self.config.breakeven_trigger_r:
                new_sl = trade.entry_price - pip_val
                if new_sl < trade.sl:
                    trade.sl = new_sl

            if mfe_r >= self.config.trailing_start_r:
                new_sl = trade.entry_price - (mfe_r - self.config.trailing_step_r) * sl_dist
                if new_sl < trade.sl:
                    trade.sl = new_sl

            if bar["high"] >= trade.sl:
                self._close_trade(trade, bar)

        if trade.outcome == "OPEN" and trade.bars_held >= self.config.stale_exit_bars:
            sl_dist = abs(trade.entry_price - trade.sl)
            if trade.direction == 1:
                pnl = bar["close"] - trade.entry_price
            else:
                pnl = trade.entry_price - bar["close"]
            if pnl < sl_dist * 0.3:
                trade.exit_time = bar["datetime"]
                trade.exit_price = bar["close"]
                trade.pnl_pips = pnl / pip_val - trade.costs_pips
                trade.outcome = "WIN" if trade.pnl_pips > 0 else "LOSS"

        if trade.outcome == "OPEN" and trade.bars_held >= self.config.max_bars:
            if trade.direction == 1:
                pnl = bar["close"] - trade.entry_price
            else:
                pnl = trade.entry_price - bar["close"]
            trade.exit_time = bar["datetime"]
            trade.exit_price = bar["close"]
            trade.pnl_pips = pnl / pip_val - trade.costs_pips
            trade.outcome = "WIN" if trade.pnl_pips > 0 else "LOSS"

    def _close_trade(self, trade: Trade, bar: pd.Series):
        trade.exit_time = bar["datetime"]
        trade.exit_price = trade.sl
        pip_val = trade.pip_value
        trade.pnl_pips = (trade.exit_price - trade.entry_price) * trade.direction / pip_val - trade.costs_pips

        if abs(trade.pnl_pips) < 1.0:
            trade.outcome = "BE"
        elif trade.pnl_pips > 0:
            trade.outcome = "WIN"
        else:
            trade.outcome = "LOSS"


def compute_metrics(trades: list[Trade]) -> dict:
    closed = [t for t in trades if t.outcome != "OPEN"]
    if not closed:
        return {"total_trades": 0, "wins": 0, "losses": 0, "be": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "rr": 0, "total_pnl_pips": 0,
                "profit_factor": 0, "max_drawdown_pips": 0, "instruments": {}}

    wins = [t for t in closed if t.outcome == "WIN"]
    losses = [t for t in closed if t.outcome == "LOSS"]
    bes = [t for t in closed if t.outcome == "BE"]

    total = len(closed)
    wr = len(wins) / total if total > 0 else 0
    avg_win = np.mean([t.pnl_pips for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t.pnl_pips) for t in losses]) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    total_pnl = sum(t.pnl_pips for t in closed)
    gross_profit = sum(t.pnl_pips for t in wins)
    gross_loss = sum(abs(t.pnl_pips) for t in losses)
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    pnls = [t.pnl_pips for t in closed]
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = np.max(dd) if len(dd) > 0 else 0

    by_instrument = {}
    for t in closed:
        if t.instrument not in by_instrument:
            by_instrument[t.instrument] = []
        by_instrument[t.instrument].append(t)

    instrument_stats = {}
    for inst, inst_trades in by_instrument.items():
        inst_wins = [t for t in inst_trades if t.outcome == "WIN"]
        inst_pnl = sum(t.pnl_pips for t in inst_trades)
        instrument_stats[inst] = {
            "trades": len(inst_trades),
            "wr": len(inst_wins) / len(inst_trades) if inst_trades else 0,
            "pnl": inst_pnl,
        }

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "be": len(bes),
        "win_rate": wr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "rr": rr,
        "total_pnl_pips": total_pnl,
        "profit_factor": pf,
        "max_drawdown_pips": max_dd,
        "instruments": instrument_stats,
    }

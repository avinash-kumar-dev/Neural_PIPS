import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

PIP_VALUE = 0.10


@dataclass
class Trade:
    entry_time: pd.Timestamp
    direction: int
    entry_price: float
    sl: float
    tp1: float
    tp2: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_pips: float = 0.0
    outcome: str = "OPEN"
    trigger: str = ""
    module: str = ""
    costs_pips: float = 0.0
    lots: float = 0.01
    be_hit: bool = False
    max_favorable: float = 0.0
    bars_held: int = 0


@dataclass
class BacktestConfig:
    spread_pips: float = 0.15
    commission_per_lot: float = 7.0
    slippage_pips: float = 2.0
    risk_per_trade: float = 0.03
    min_rr: float = 3.0
    initial_equity: float = 100000.0
    min_sl_pips: float = 100.0
    max_sl_pips: float = 500.0
    trailing_start_r: float = 2.0
    trailing_step_r: float = 1.0
    max_bars: int = 300
    breakeven_at_1r: bool = True


class BacktestEngine:
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> list[Trade]:
        trades = []
        equity = self.config.initial_equity
        open_trade = None

        for i in range(len(df)):
            if open_trade is not None:
                bar = df.iloc[i]
                trade = open_trade
                trade.bars_held += 1

                sl_distance = abs(trade.entry_price - trade.sl)

                if trade.direction == 1:
                    current_profit = bar["high"] - trade.entry_price
                    if current_profit > trade.max_favorable:
                        trade.max_favorable = current_profit

                    mfe_r = trade.max_favorable / sl_distance if sl_distance > 0 else 0

                    if self.config.breakeven_at_1r and not trade.be_hit and mfe_r >= 1.0:
                        trade.sl = trade.entry_price + 1.0 * PIP_VALUE
                        trade.be_hit = True

                    if self.config.trailing_start_r > 0 and mfe_r >= self.config.trailing_start_r:
                        new_sl = trade.entry_price + (mfe_r - self.config.trailing_step_r) * sl_distance
                        if new_sl > trade.sl:
                            trade.sl = new_sl

                    if bar["low"] <= trade.sl:
                        trade.exit_time = bar["datetime"]
                        trade.exit_price = trade.sl
                        trade.pnl_pips = (trade.sl - trade.entry_price) / PIP_VALUE - trade.costs_pips
                        trade.outcome = "WIN" if trade.sl > trade.entry_price else ("BE" if abs(trade.sl - trade.entry_price) < 1.0 * PIP_VALUE else "LOSS")
                        equity += trade.pnl_pips * PIP_VALUE * trade.lots * 100
                        trades.append(trade)
                        open_trade = None

                else:
                    current_profit = trade.entry_price - bar["low"]
                    if current_profit > trade.max_favorable:
                        trade.max_favorable = current_profit

                    mfe_r = trade.max_favorable / sl_distance if sl_distance > 0 else 0

                    if self.config.breakeven_at_1r and not trade.be_hit and mfe_r >= 1.0:
                        trade.sl = trade.entry_price - 1.0 * PIP_VALUE
                        trade.be_hit = True

                    if self.config.trailing_start_r > 0 and mfe_r >= self.config.trailing_start_r:
                        new_sl = trade.entry_price - (mfe_r - self.config.trailing_step_r) * sl_distance
                        if new_sl < trade.sl:
                            trade.sl = new_sl

                    if bar["high"] >= trade.sl:
                        trade.exit_time = bar["datetime"]
                        trade.exit_price = trade.sl
                        trade.pnl_pips = (trade.entry_price - trade.sl) / PIP_VALUE - trade.costs_pips
                        trade.outcome = "WIN" if trade.sl < trade.entry_price else ("BE" if abs(trade.sl - trade.entry_price) < 1.0 * PIP_VALUE else "LOSS")
                        equity += trade.pnl_pips * PIP_VALUE * trade.lots * 100
                        trades.append(trade)
                        open_trade = None

                if open_trade is not None and trade.bars_held >= self.config.max_bars:
                    if trade.direction == 1:
                        trade.exit_time = bar["datetime"]
                        trade.exit_price = bar["close"]
                        trade.pnl_pips = (bar["close"] - trade.entry_price) / PIP_VALUE - trade.costs_pips
                    else:
                        trade.exit_time = bar["datetime"]
                        trade.exit_price = bar["close"]
                        trade.pnl_pips = (trade.entry_price - bar["close"]) / PIP_VALUE - trade.costs_pips
                    trade.outcome = "WIN" if trade.pnl_pips > 0 else "LOSS"
                    equity += trade.pnl_pips * PIP_VALUE * trade.lots * 100
                    trades.append(trade)
                    open_trade = None

            if open_trade is None and i < len(signals):
                row = signals.iloc[i]
                direction = 0
                sl = np.nan
                trigger = ""

                if row.get("long_entry", False) and row.get("layer3_confirmed", False):
                    direction = 1
                    sl = row.get("long_sl", np.nan)
                    trigger = row.get("long_trigger", "")
                elif row.get("short_entry", False) and row.get("layer3_confirmed", False):
                    direction = -1
                    sl = row.get("short_sl", np.nan)
                    trigger = row.get("short_trigger", "")

                if direction != 0 and not pd.isna(sl):
                    entry_price = df.iloc[i]["close"]
                    entry_price += direction * self.config.slippage_pips * PIP_VALUE

                    sl_pips = abs(entry_price - sl) / PIP_VALUE
                    if sl_pips < self.config.min_sl_pips or sl_pips > self.config.max_sl_pips:
                        continue

                    tp2_pips = sl_pips * self.config.min_rr
                    if direction == 1:
                        tp1 = entry_price + sl_pips * 1.5 * PIP_VALUE
                        tp2 = entry_price + tp2_pips * PIP_VALUE
                    else:
                        tp1 = entry_price - sl_pips * 1.5 * PIP_VALUE
                        tp2 = entry_price - tp2_pips * PIP_VALUE

                    costs_pips = self.config.spread_pips + self.config.slippage_pips + (self.config.commission_per_lot / (PIP_VALUE * 100))

                    risk_amount = equity * self.config.risk_per_trade
                    lots = risk_amount / (sl_pips * PIP_VALUE * 100) if sl_pips > 0 else 0
                    lots = round(max(lots, 0.01), 2)

                    open_trade = Trade(
                        entry_time=df.iloc[i]["datetime"],
                        direction=direction,
                        entry_price=entry_price,
                        sl=sl,
                        tp1=tp1,
                        tp2=tp2,
                        trigger=trigger,
                        module="trend",
                        costs_pips=costs_pips,
                        lots=lots,
                    )

        if open_trade is not None:
            open_trade.outcome = "OPEN"
            trades.append(open_trade)

        return trades


def compute_metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {"total_trades": 0}

    closed = [t for t in trades if t.outcome != "OPEN"]
    wins = [t for t in closed if t.outcome == "WIN"]
    losses = [t for t in closed if t.outcome == "LOSS"]
    bes = [t for t in closed if t.outcome == "BE"]

    total = len(closed)
    win_count = len(wins)
    loss_count = len(losses)
    be_count = len(bes)

    win_rate = win_count / total if total > 0 else 0
    avg_win = np.mean([t.pnl_pips for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t.pnl_pips) for t in losses]) if losses else 0
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

    total_pnl = sum(t.pnl_pips for t in closed)
    gross_profit = sum(t.pnl_pips for t in wins)
    gross_loss = sum(abs(t.pnl_pips) for t in losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    pnls = [t.pnl_pips for t in closed]
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    avg_bars = np.mean([t.bars_held for t in closed]) if closed else 0

    return {
        "total_trades": total,
        "wins": win_count,
        "losses": loss_count,
        "be": be_count,
        "open": len(trades) - total,
        "win_rate": win_rate,
        "avg_win_pips": avg_win,
        "avg_loss_pips": avg_loss,
        "avg_rr": avg_rr,
        "total_pnl_pips": total_pnl,
        "profit_factor": profit_factor,
        "max_drawdown_pips": max_drawdown,
        "gross_profit_pips": gross_profit,
        "gross_loss_pips": gross_loss,
        "avg_bars_held": avg_bars,
    }

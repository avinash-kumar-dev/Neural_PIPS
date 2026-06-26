import pandas as pd
import numpy as np
from datetime import datetime


class Backtester:
    def __init__(self, config=None):
        cfg = config or {}
        self.initial_capital = cfg.get('initial_capital', 100000)
        self.risk_per_trade = cfg.get('risk_per_trade', 0.02)
        self.spread_pips = cfg.get('spread_pips', 0.5)

    def run_simple(self, signals_df, price_df):
        signals = signals_df[signals_df['signal'] != 'NO-TRADE'].copy()

        trades = []
        equity = self.initial_capital
        equity_curve = [equity]

        for _, signal in signals.iterrows():
            entry_price = self._get_entry_price(signal, price_df)
            if entry_price is None:
                continue

            tp_pips = signal['tp_pips']
            sl_pips = signal['sl_pips']

            risk_amount = equity * self.risk_per_trade
            sl_distance = sl_pips / 10000
            position_size = risk_amount / sl_distance

            result = self._simulate_trade(signal, entry_price, tp_pips, sl_pips, price_df)
            trades.append(result)

            pnl = result['pnl']
            equity += pnl
            equity_curve.append(equity)

        trades_df = pd.DataFrame(trades)
        metrics = self._compute_metrics(trades_df, equity_curve)
        return trades_df, metrics

    def _get_entry_price(self, signal, price_df):
        ts = signal['timestamp']
        mask = price_df['time'] == ts
        if mask.any():
            return price_df.loc[mask, 'close'].iloc[0]
        return None

    def _simulate_trade(self, signal, entry_price, tp_pips, sl_pips, price_df):
        ts = signal['timestamp']
        signal_dir = 1 if signal['signal'] == 'LONG' else -1

        tp_price = entry_price + signal_dir * tp_pips / 10000
        sl_price = entry_price - signal_dir * sl_pips / 10000

        idx = price_df[price_df['time'] >= ts].index
        if len(idx) == 0:
            return self._empty_trade(signal, entry_price)

        start_idx = idx[0]
        end_idx = min(start_idx + 20, len(price_df))

        for i in range(start_idx + 1, end_idx):
            high = price_df['high'].iloc[i]
            low = price_df['low'].iloc[i]

            if signal_dir == 1:
                if high >= tp_price:
                    return self._win_trade(signal, entry_price, tp_price, tp_pips, i - start_idx)
                if low <= sl_price:
                    return self._loss_trade(signal, entry_price, sl_price, sl_pips, i - start_idx)
            else:
                if low <= tp_price:
                    return self._win_trade(signal, entry_price, tp_price, tp_pips, i - start_idx)
                if high >= sl_price:
                    return self._loss_trade(signal, entry_price, sl_price, sl_pips, i - start_idx)

        exit_price = price_df['close'].iloc[end_idx - 1]
        pnl_pips = signal_dir * (exit_price - entry_price) * 10000
        return {
            'timestamp': ts,
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'entry': entry_price,
            'exit': exit_price,
            'tp_pips': tp_pips,
            'sl_pips': sl_pips,
            'pnl_pips': pnl_pips,
            'pnl': pnl_pips / 10000 * 100000,
            'result': 'win' if pnl_pips > 0 else 'loss',
            'bars_held': i - start_idx,
            'actual_label': signal['actual_label'],
        }

    def _win_trade(self, signal, entry, exit_price, pnl_pips, bars):
        return {
            'timestamp': signal['timestamp'],
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'entry': entry,
            'exit': exit_price,
            'tp_pips': signal['tp_pips'],
            'sl_pips': signal['sl_pips'],
            'pnl_pips': pnl_pips,
            'pnl': pnl_pips / 10000 * 100000,
            'result': 'win',
            'bars_held': bars,
            'actual_label': signal['actual_label'],
        }

    def _loss_trade(self, signal, entry, exit_price, loss_pips, bars):
        return {
            'timestamp': signal['timestamp'],
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'entry': entry,
            'exit': exit_price,
            'tp_pips': signal['tp_pips'],
            'sl_pips': signal['sl_pips'],
            'pnl_pips': -loss_pips,
            'pnl': -loss_pips / 10000 * 100000,
            'result': 'loss',
            'bars_held': bars,
            'actual_label': signal['actual_label'],
        }

    def _empty_trade(self, signal, entry):
        return {
            'timestamp': signal['timestamp'],
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'entry': entry,
            'exit': entry,
            'tp_pips': signal['tp_pips'],
            'sl_pips': signal['sl_pips'],
            'pnl_pips': 0,
            'pnl': 0,
            'result': 'no_trade',
            'bars_held': 0,
            'actual_label': signal['actual_label'],
        }

    def _compute_metrics(self, trades_df, equity_curve):
        if len(trades_df) == 0:
            return {'total_trades': 0}

        wins = trades_df[trades_df['result'] == 'win']
        losses = trades_df[trades_df['result'] == 'loss']

        total = len(trades_df)
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = win_count / total * 100
        avg_win = wins['pnl_pips'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pips'].mean() if len(losses) > 0 else 0
        profit_factor = abs(wins['pnl_pips'].sum() / losses['pnl_pips'].sum()) if len(losses) > 0 and losses['pnl_pips'].sum() != 0 else float('inf')

        total_pnl = trades_df['pnl_pips'].sum()
        avg_confidence = trades_df['confidence'].mean()

        equity_series = pd.Series(equity_curve)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = drawdown.min()

        return {
            'total_trades': total,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'avg_win_pips': avg_win,
            'avg_loss_pips': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl_pips': total_pnl,
            'avg_confidence': avg_confidence,
            'max_drawdown_pct': max_drawdown,
            'final_equity': equity_curve[-1],
            'return_pct': (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100,
        }

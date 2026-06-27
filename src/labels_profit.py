import pandas as pd
import numpy as np


class ProfitAwareLabeler:
    """Generate profit-aware labels by simulating both long and short outcomes.

    Labels:
        1 = WIN: at least one direction hits TP first
        0 = LOSS: both directions hit SL first
        2 = TIMEOUT: neither TP nor SL hit within horizon
    """

    def __init__(self, tp_mult=4.5, sl_mult=1.5, horizon=20, min_tp_pips=9, min_sl_pips=3):
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.horizon = horizon
        self.min_tp_pips = min_tp_pips
        self.min_sl_pips = min_sl_pips

    def compute_atr(self, df, period=14):
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        atr = np.full(len(df), np.nan)
        if len(tr) >= period:
            atr[period] = np.mean(tr[:period])
            alpha = 1.0 / period
            for i in range(period + 1, len(df)):
                atr[i] = atr[i - 1] * (1 - alpha) + tr[i - 1] * alpha

        return atr

    def compute_atr_pips(self, df, period=14):
        atr = self.compute_atr(df, period)
        return atr * 10000

    def simulate_forward(self, high, low, entry_idx, tp_level, sl_level, horizon):
        """Simulate forward bars to check if TP or SL is hit first."""
        entry = (tp_level + sl_level) / 2

        for j in range(entry_idx + 1, min(entry_idx + 1 + horizon, len(high))):
            bar_high = high[j]
            bar_low = low[j]

            if tp_level > sl_level:
                tp_hit = bar_high >= tp_level
                sl_hit = bar_low <= sl_level
            else:
                tp_hit = bar_low <= tp_level
                sl_hit = bar_high >= sl_level

            if tp_hit and sl_hit:
                return 'both'
            if tp_hit:
                return 'tp'
            if sl_hit:
                return 'sl'

        return 'timeout'

    def label(self, df):
        """Generate profit-aware labels for all bars.

        Returns:
            labels: Series with 1=WIN, 0=LOSS, 2=TIMEOUT
            metadata: DataFrame with per-bar outcome details
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        atr_pips = self.compute_atr_pips(df)

        labels = np.full(len(df), 2, dtype=int)
        best_direction = np.full(len(df), -1, dtype=int)
        long_outcome = ['timeout'] * len(df)
        short_outcome = ['timeout'] * len(df)

        for i in range(len(df) - self.horizon):
            if np.isnan(atr_pips[i]) or atr_pips[i] <= 0:
                continue

            entry = close[i]
            atr_val = atr_pips[i] / 10000

            tp_long = entry + max(self.tp_mult * atr_val, self.min_tp_pips / 10000)
            sl_long = entry - max(self.sl_mult * atr_val, self.min_sl_pips / 10000)
            tp_short = entry - max(self.tp_mult * atr_val, self.min_tp_pips / 10000)
            sl_short = entry + max(self.sl_mult * atr_val, self.min_sl_pips / 10000)

            long_res = self.simulate_forward(high, low, i, tp_long, sl_long, self.horizon)
            short_res = self.simulate_forward(high, low, i, tp_short, sl_short, self.horizon)

            long_outcome[i] = long_res
            short_outcome[i] = short_res

            long_win = long_res == 'tp'
            short_win = short_res == 'tp'
            long_lose = long_res == 'sl'
            short_lose = short_res == 'sl'

            if long_win or short_win:
                labels[i] = 1
                if long_win and not short_win:
                    best_direction[i] = 1
                elif short_win and not long_win:
                    best_direction[i] = 0
                else:
                    best_direction[i] = -1
            elif long_lose and short_lose:
                labels[i] = 0
                best_direction[i] = -1
            else:
                labels[i] = 2
                best_direction[i] = -1

        metadata = pd.DataFrame({
            'profit_label': labels,
            'best_direction': best_direction,
            'long_outcome': long_outcome,
            'short_outcome': short_outcome,
            'atr_pips': atr_pips,
        }, index=df.index)

        return pd.Series(labels, index=df.index, name='profit_label'), metadata


class TradeOutcomeSimulator:
    """Simulate actual trade outcomes for backtesting profit-aware signals."""

    def __init__(self, tp_mult=4.5, sl_mult=1.5, horizon=20, min_tp_pips=9, min_sl_pips=3):
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.horizon = horizon
        self.min_tp_pips = min_tp_pips
        self.min_sl_pips = min_sl_pips

    def simulate_trade(self, entry_price, direction, atr_pips, future_high, future_low):
        """Simulate a single trade's outcome.

        Returns:
            dict with outcome, pnl_pips, bars_held
        """
        tp_pips = max(self.tp_mult * atr_pips, self.min_tp_pips)
        sl_pips = max(self.sl_mult * atr_pips, self.min_sl_pips)

        if direction == 1:
            tp_price = entry_price + tp_pips / 10000
            sl_price = entry_price - sl_pips / 10000
        else:
            tp_price = entry_price - tp_pips / 10000
            sl_price = entry_price + sl_pips / 10000

        for bar_idx in range(min(self.horizon, len(future_high))):
            if direction == 1:
                if future_high[bar_idx] >= tp_price:
                    return {'outcome': 'win', 'pnl_pips': tp_pips, 'bars_held': bar_idx + 1}
                if future_low[bar_idx] <= sl_price:
                    return {'outcome': 'loss', 'pnl_pips': -sl_pips, 'bars_held': bar_idx + 1}
            else:
                if future_low[bar_idx] <= tp_price:
                    return {'outcome': 'win', 'pnl_pips': tp_pips, 'bars_held': bar_idx + 1}
                if future_high[bar_idx] >= sl_price:
                    return {'outcome': 'loss', 'pnl_pips': -sl_pips, 'bars_held': bar_idx + 1}

        return {'outcome': 'timeout', 'pnl_pips': 0, 'bars_held': self.horizon}

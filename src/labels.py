import pandas as pd
import numpy as np


class TripleBarrierLabeler:
    def __init__(self, tp_mult=4.5, sl_mult=1.5, horizon=20, min_tp_pips=9, max_tp_pips=45, min_sl_pips=3, max_sl_pips=15):
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.horizon = horizon
        self.min_tp_pips = min_tp_pips
        self.max_tp_pips = max_tp_pips
        self.min_sl_pips = min_sl_pips
        self.max_sl_pips = max_sl_pips

    def compute_atr(self, df, period=14):
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def compute_tp_sl(self, df):
        atr = self.compute_atr(df)
        tp_raw = atr * self.tp_mult
        sl_raw = atr * self.sl_mult

        tp_raw = tp_raw.clip(lower=self.min_tp_pips / 10000, upper=self.max_tp_pips / 10000)
        sl_raw = sl_raw.clip(lower=self.min_sl_pips / 10000, upper=self.max_sl_pips / 10000)

        return tp_raw, sl_raw, atr

    def label_bars(self, df):
        tp_raw, sl_raw, atr = self.compute_tp_sl(df)

        labels = pd.Series(-1, index=df.index, dtype=int)
        touch_bar = pd.Series(np.nan, index=df.index)
        touch_type = pd.Series('', index=df.index, dtype=str)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        tp_vals = tp_raw.values
        sl_vals = sl_raw.values

        for i in range(len(df) - 1):
            if np.isnan(tp_vals[i]) or np.isnan(sl_vals[i]):
                continue

            entry = close[i]
            tp_level = entry + tp_vals[i]
            sl_level = entry - sl_vals[i]

            end_idx = min(i + self.horizon + 1, len(df))

            hit_tp = False
            hit_sl = False
            first_touch_bar = -1
            first_touch_type = ''

            for j in range(i + 1, end_idx):
                if high[j] >= tp_level and not hit_tp:
                    hit_tp = True
                    if first_touch_bar == -1:
                        first_touch_bar = j - i
                        first_touch_type = 'tp'
                if low[j] <= sl_level and not hit_sl:
                    hit_sl = True
                    if first_touch_bar == -1:
                        first_touch_bar = j - i
                        first_touch_type = 'sl'

            if hit_tp and not hit_sl:
                labels.iloc[i] = 1
                touch_bar.iloc[i] = first_touch_bar
                touch_type.iloc[i] = 'tp'
            elif hit_sl and not hit_tp:
                labels.iloc[i] = 0
                touch_bar.iloc[i] = first_touch_bar
                touch_type.iloc[i] = 'sl'
            elif hit_tp and hit_sl:
                if first_touch_type == 'tp':
                    labels.iloc[i] = 1
                    touch_bar.iloc[i] = first_touch_bar
                    touch_type.iloc[i] = 'tp'
                else:
                    labels.iloc[i] = 0
                    touch_bar.iloc[i] = first_touch_bar
                    touch_type.iloc[i] = 'sl'
            else:
                labels.iloc[i] = -1
                touch_bar.iloc[i] = np.nan
                touch_type.iloc[i] = 'none'

        return labels, touch_bar, touch_type, tp_raw, sl_raw, atr

    def label_dataframe(self, df):
        labels, touch_bar, touch_type, tp_raw, sl_raw, atr = self.label_bars(df)

        result = df.copy()
        result['label'] = labels
        result['touch_bar'] = touch_bar
        result['touch_type'] = touch_type
        result['tp_pips'] = tp_raw * 10000
        result['sl_pips'] = sl_raw * 10000
        result['atr_pips'] = atr * 10000
        result['tp_sl_ratio'] = result['tp_pips'] / result['sl_pips']

        return result

    def get_stats(self, df_labeled):
        labels = df_labeled['label']
        total = len(labels[labels != -1])
        long_count = (labels == 1).sum()
        short_count = (labels == 0).sum()
        no_trade = (labels == -1).sum()

        stats = {
            'total_labeled': total,
            'long_signals': long_count,
            'short_signals': short_count,
            'no_trade': no_trade,
            'long_pct': long_count / total * 100 if total > 0 else 0,
            'short_pct': short_count / total * 100 if total > 0 else 0,
            'avg_tp_pips': df_labeled.loc[labels != -1, 'tp_pips'].mean(),
            'avg_sl_pips': df_labeled.loc[labels != -1, 'sl_pips'].mean(),
            'avg_atr_pips': df_labeled.loc[labels != -1, 'atr_pips'].mean(),
            'avg_touch_bar': df_labeled.loc[labels != -1, 'touch_bar'].mean(),
        }
        return stats

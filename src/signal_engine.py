import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timezone


class SignalEngine:
    def __init__(self, config=None):
        cfg = config or {}
        self.check_interval = cfg.get('check_interval', 300)
        self.confidence_threshold = cfg.get('confidence_threshold', 78)
        self.min_tp_pips = cfg.get('min_tp_pips', 9)
        self.min_sl_pips = cfg.get('min_sl_pips', 3)
        self.feature_pipeline = cfg.get('feature_pipeline', None)
        self.ml_gate = cfg.get('ml_gate', 0.85)
        self.confluence_min = cfg.get('confluence_min', 2)
        self.max_spread_pips = cfg.get('max_spread_pips', 0.5)
        self.min_atr_ratio = cfg.get('min_atr_ratio', 0.5)
        self.max_atr_ratio = cfg.get('max_atr_ratio', 2.5)
        self.min_atr_pips = cfg.get('min_atr_pips', 3.0)
        self.max_atr_pips = cfg.get('max_atr_pips', 20.0)

        sessions = cfg.get('sessions', None)
        if sessions:
            self.sessions = sessions
        else:
            self.sessions = [
                {'start': cfg.get('session_start', 13), 'end': cfg.get('session_end', 16)},
            ]

    def in_session(self, hour):
        for s in self.sessions:
            if s['start'] <= hour < s['end']:
                return True
        return False

    def generate_signal(self, features_row, raw_features_row, model, meta_or_weights):
        if 'time' not in features_row:
            return self._no_trade('no_time')

        hour = features_row['time'].hour if hasattr(features_row['time'], 'hour') else pd.to_datetime(features_row['time']).hour

        if not self.in_session(hour):
            return self._no_trade('outside_session')

        ml_pred, ml_confidence, individual_preds = self._predict(features_row, model, meta_or_weights)

        n_models = len(individual_preds)
        agree_count = sum(1 for p in individual_preds if (p > 0.5) == (ml_pred == 1))
        if agree_count < self.confluence_min:
            return self._no_trade(f'low_confluence_{agree_count}/{n_models}')

        h4_regime = raw_features_row.get('h4_regime', 0)
        if h4_regime > 0 and ml_pred == 0:
            return self._no_trade('h4_bullish_short_blocked')
        if h4_regime < 0 and ml_pred == 1:
            return self._no_trade('h4_bearish_long_blocked')

        ml_conf_for_gate = (1 - ml_confidence) if ml_pred == 0 else ml_confidence
        if ml_conf_for_gate < self.ml_gate:
            return self._no_trade(f'low_ml_confidence_{ml_conf_for_gate:.2f}')

        atr_pips = raw_features_row.get('atr_pips', None)
        if atr_pips is None:
            atr_14 = raw_features_row.get('atr_14', 0)
            atr_pips = atr_14 * 10000

        if atr_pips < self.min_atr_pips:
            return self._no_trade(f'atr_too_low_{atr_pips:.1f}')
        if atr_pips > self.max_atr_pips:
            return self._no_trade(f'atr_too_high_{atr_pips:.1f}')

        atr_ratio = raw_features_row.get('atr_ratio', 1.0)
        if atr_ratio < self.min_atr_ratio:
            return self._no_trade(f'atr_ratio_low_{atr_ratio:.2f}')
        if atr_ratio > self.max_atr_ratio:
            return self._no_trade(f'atr_ratio_high_{atr_ratio:.2f}')

        spread_pips = raw_features_row.get('spread_pips', 0)
        if spread_pips > self.max_spread_pips:
            return self._no_trade(f'spread_high_{spread_pips:.1f}')

        from src.confidence_scorer import ConfidenceScorer
        scorer = ConfidenceScorer({'threshold': self.confidence_threshold})
        confidence, scores = scorer.score(features_row, raw_features_row)

        if confidence < self.confidence_threshold:
            return self._no_trade(f'low_confidence_{confidence:.0f}')

        tp_pips = max(atr_pips * 4.5, self.min_tp_pips)
        sl_pips = max(atr_pips * 1.5, self.min_sl_pips)

        signal_dir = 'LONG' if ml_pred == 1 else 'SHORT'

        return {
            'signal': signal_dir,
            'confidence': confidence,
            'ml_confidence': ml_confidence,
            'confluence': f'{agree_count}/{n_models}',
            'tp_pips': tp_pips,
            'sl_pips': sl_pips,
            'tp_sl_ratio': tp_pips / sl_pips,
            'atr_pips': atr_pips,
            'scores': scores,
            'timestamp': features_row['time'],
            'reason': 'signal_generated',
        }

    def _predict(self, features_row, model, meta_or_weights):
        feat_cols = [c for c in features_row.index if c != 'time']
        X = features_row[feat_cols].values.reshape(1, -1).astype(float)

        if self.feature_pipeline is not None:
            X = self.feature_pipeline.transform(X)

        individual_preds = []
        for m in model:
            proba = m.predict_proba(X)[:, 1]
            individual_preds.append(float(proba[0]))

        base_proba = np.column_stack(individual_preds)

        if isinstance(meta_or_weights, np.ndarray):
            meta_pred = base_proba @ meta_or_weights
        else:
            meta_pred = np.clip(meta_or_weights.predict(base_proba), 0, 1)

        return int(meta_pred[0] > 0.5), float(meta_pred[0]), individual_preds

    def _no_trade(self, reason):
        return {
            'signal': 'NO-TRADE',
            'confidence': 0,
            'ml_confidence': 0,
            'confluence': '0/0',
            'tp_pips': 0,
            'sl_pips': 0,
            'tp_sl_ratio': 0,
            'atr_pips': 0,
            'scores': {},
            'timestamp': None,
            'reason': reason,
        }

    def run_backtest(self, features_df, raw_features_df, labels_df, model, meta_model):
        signals = []
        for idx in features_df.index:
            feat_row = features_df.loc[idx]
            raw_row = raw_features_df.loc[idx]
            label_row = labels_df.loc[idx]

            feat_with_time = feat_row.copy()
            if 'time' not in feat_with_time and 'time' in label_row:
                feat_with_time['time'] = label_row['time']

            signal = self.generate_signal(feat_with_time, raw_row, model, meta_model)
            signal['actual_label'] = label_row.get('label', -1)
            signal['actual_tp_pips'] = label_row.get('tp_pips', 0)
            signal['actual_sl_pips'] = label_row.get('sl_pips', 0)
            signals.append(signal)

        return pd.DataFrame(signals)

    def get_stats(self, signals_df):
        generated = signals_df[signals_df['signal'] != 'NO-TRADE']
        no_trade = signals_df[signals_df['signal'] == 'NO-TRADE']

        correct = 0
        total_trades = 0
        for _, row in generated.iterrows():
            if row['actual_label'] != -1:
                total_trades += 1
                if row['signal'] == 'LONG' and row['actual_label'] == 1:
                    correct += 1
                elif row['signal'] == 'SHORT' and row['actual_label'] == 0:
                    correct += 1

        return {
            'total_bars': len(signals_df),
            'signals_generated': len(generated),
            'no_trade': len(no_trade),
            'signal_rate': len(generated) / len(signals_df) * 100,
            'correct_signals': correct,
            'total_evaluated': total_trades,
            'accuracy': correct / total_trades * 100 if total_trades > 0 else 0,
            'avg_confidence': generated['confidence'].mean() if len(generated) > 0 else 0,
            'avg_tp_pips': generated['tp_pips'].mean() if len(generated) > 0 else 0,
            'avg_sl_pips': generated['sl_pips'].mean() if len(generated) > 0 else 0,
        }

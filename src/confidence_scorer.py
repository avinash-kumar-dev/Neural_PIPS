import pandas as pd
import numpy as np


class ConfidenceScorer:
    def __init__(self, config=None):
        cfg = config or {}
        self.weights = cfg.get('weights', {
            'trend': 0.20,
            'momentum': 0.15,
            'volatility': 0.15,
            'structure': 0.15,
            'pattern': 0.10,
            'mtf': 0.10,
            'session': 0.10,
            'spread': 0.05,
        })
        self.threshold = cfg.get('threshold', 80)

    def score(self, row, raw_features=None):
        feat = raw_features if raw_features is not None else row

        scores = {
            'trend': self._score_trend(feat),
            'momentum': self._score_momentum(feat),
            'volatility': self._score_volatility(feat),
            'structure': self._score_structure(feat),
            'pattern': self._score_pattern(feat),
            'mtf': self._score_mtf(feat),
            'session': self._score_session(row),
            'spread': self._score_spread(feat),
        }

        weighted = sum(scores[k] * self.weights[k] for k in scores)
        return weighted, scores

    def score_batch(self, df, raw_features=None):
        results = []
        for idx in df.index:
            row = df.loc[idx]
            feat = raw_features.loc[idx] if raw_features is not None else row
            score, breakdown = self.score(row, feat)
            results.append({
                'confidence': score,
                'scores': breakdown,
                'pass': score >= self.threshold,
            })
        return results

    def _score_trend(self, feat):
        score = 50
        adx = feat.get('adx', 0)
        if adx >= 40:
            score += 30
        elif adx >= 25:
            score += 15
        elif adx >= 20:
            score += 5

        if feat.get('h4_bullish', 0) == 1 or feat.get('h4_bearish', 0) == 1:
            score += 20

        return min(score, 100)

    def _score_momentum(self, feat):
        score = 50
        rsi = feat.get('rsi_14', 50)
        if 40 <= rsi <= 60:
            score += 10
        elif 30 <= rsi <= 70:
            score += 5

        if feat.get('macd_cross', 0) != 0:
            score += 15

        if feat.get('stoch_rsi_k', 50) > 50:
            score += 5

        return min(score, 100)

    def _score_volatility(self, feat):
        score = 50
        atr_ratio = feat.get('atr_ratio', 1.0)
        if 0.8 <= atr_ratio <= 1.2:
            score += 20
        elif 0.5 <= atr_ratio <= 1.5:
            score += 10

        bb_pos = feat.get('bb_position', 0.5)
        if 0.2 <= bb_pos <= 0.8:
            score += 15

        return min(score, 100)

    def _score_structure(self, feat):
        score = 50
        if feat.get('ema_8_above', 0) == 1:
            score += 10
        if feat.get('ema_21_above', 0) == 1:
            score += 10
        if feat.get('ema_50_above', 0) == 1:
            score += 10

        bb_pos = feat.get('bb_position', 0.5)
        if 0.3 <= bb_pos <= 0.7:
            score += 10

        return min(score, 100)

    def _score_pattern(self, feat):
        score = 50
        body = feat.get('body_ratio', 0)
        if body >= 0.5:
            score += 20
        elif body >= 0.3:
            score += 10

        vol = feat.get('volume_ratio', 1.0)
        if vol >= 1.5:
            score += 15
        elif vol >= 1.0:
            score += 5

        return min(score, 100)

    def _score_mtf(self, feat):
        score = 50
        alignment = feat.get('mtf_alignment', 0)
        if alignment >= 3:
            score += 30
        elif alignment >= 2:
            score += 15

        if feat.get('h1_above_ema50', 0) == 1:
            score += 10

        return min(score, 100)

    def _score_session(self, row):
        score = 50
        hour = row['time'].hour if hasattr(row['time'], 'hour') else pd.to_datetime(row['time']).hour

        if 13 <= hour < 16:
            score += 30
        elif 12 <= hour < 17:
            score += 15

        dow = row['time'].weekday() if hasattr(row['time'], 'weekday') else pd.to_datetime(row['time']).weekday()
        if dow < 5:
            score += 10

        return min(score, 100)

    def _score_spread(self, feat):
        score = 80
        vol = feat.get('volume_ratio', 1.0)
        if vol >= 1.2:
            score += 10

        return min(score, 100)

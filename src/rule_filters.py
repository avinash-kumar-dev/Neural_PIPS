import pandas as pd
import numpy as np


class RuleFilters:
    def __init__(self, config=None):
        cfg = config or {}
        self.min_adx = cfg.get('min_adx', 20)
        self.max_spread_pips = cfg.get('max_spread_pips', 0.5)
        self.min_body_ratio = cfg.get('min_body_ratio', 0.3)
        self.min_volume_ratio = cfg.get('min_volume_ratio', 0.8)
        self.session_start = cfg.get('session_start', 13)
        self.session_end = cfg.get('session_end', 16)
        self.min_mtf_alignment = cfg.get('min_mtf_alignment', 2)

    def apply_all(self, df, raw_features=None):
        result = df.copy()
        result['filter_reject'] = 0
        result['filter_reason'] = ''

        feat = raw_features if raw_features is not None else df

        r1, _ = self._gate_session(result)
        r2, _ = self._gate_adx(feat)
        r3, _ = self._gate_body_ratio(feat)
        r4, _ = self._gate_volume(feat)
        r5, _ = self._gate_mtf_alignment(feat)

        reject = r1 | r2 | r3 | r4 | r5
        result.loc[reject, 'filter_reject'] = 1

        reasons = pd.Series('', index=result.index)
        reasons[r1] = 'outside_session'
        reasons[(~r1) & r2] = 'adx_too_low'
        reasons[(~r1) & (~r2) & r3] = 'body_too_small'
        reasons[(~r1) & (~r2) & (~r3) & r4] = 'volume_too_low'
        reasons[(~r1) & (~r2) & (~r3) & (~r4) & r5] = 'mtf_misaligned'
        result.loc[reject, 'filter_reason'] = reasons[reject]

        return result

    def _gate_session(self, df):
        hours = pd.to_datetime(df['time']).dt.hour
        reject = ~((hours >= self.session_start) & (hours < self.session_end))
        return reject, 'outside_session'

    def _gate_adx(self, feat):
        if 'adx' not in feat.columns:
            return pd.Series(False, index=feat.index), ''
        reject = feat['adx'] < self.min_adx
        return reject, 'adx_too_low'

    def _gate_body_ratio(self, feat):
        if 'body_ratio' not in feat.columns:
            return pd.Series(False, index=feat.index), ''
        reject = feat['body_ratio'] < self.min_body_ratio
        return reject, 'body_too_small'

    def _gate_volume(self, feat):
        if 'volume_ratio' not in feat.columns:
            return pd.Series(False, index=feat.index), ''
        reject = feat['volume_ratio'] < self.min_volume_ratio
        return reject, 'volume_too_low'

    def _gate_mtf_alignment(self, feat):
        if 'mtf_alignment' not in feat.columns:
            return pd.Series(False, index=feat.index), ''
        reject = feat['mtf_alignment'] < self.min_mtf_alignment
        return reject, 'mtf_misaligned'

    def get_stats(self, df_filtered):
        total = len(df_filtered)
        rejected = df_filtered['filter_reject'].sum()
        passed = total - rejected

        reason_counts = df_filtered[df_filtered['filter_reject'] == 1]['filter_reason'].value_counts()

        return {
            'total': total,
            'rejected': int(rejected),
            'passed': int(passed),
            'rejection_rate': rejected / total * 100,
            'reason_counts': reason_counts.to_dict(),
        }

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os


class PerformanceMonitor:
    def __init__(self, config=None):
        cfg = config or {}
        self.min_accuracy = cfg.get('min_accuracy', 0.35)
        self.drift_threshold = cfg.get('drift_threshold', 0.10)
        self.retrain_window = cfg.get('retrain_window', 100)
        self.data_dir = cfg.get('data_dir', 'data/monitoring')
        os.makedirs(self.data_dir, exist_ok=True)

    def track_signal(self, signal, actual_outcome=None):
        record = {
            'timestamp': signal['timestamp'],
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'tp_pips': signal['tp_pips'],
            'sl_pips': signal['sl_pips'],
            'actual_outcome': actual_outcome,
            'predicted_correctly': None,
        }

        if actual_outcome is not None:
            if signal['signal'] == 'LONG' and actual_outcome == 1:
                record['predicted_correctly'] = True
            elif signal['signal'] == 'SHORT' and actual_outcome == 0:
                record['predicted_correctly'] = True
            else:
                record['predicted_correctly'] = False

        self._append_record(record)
        return record

    def get_accuracy(self, window=None):
        records = self._load_records()
        if len(records) == 0:
            return None

        evaluated = [r for r in records if r['predicted_correctly'] is not None]
        if window:
            evaluated = evaluated[-window:]

        if len(evaluated) == 0:
            return None

        correct = sum(1 for r in evaluated if r['predicted_correctly'])
        return correct / len(evaluated)

    def check_drift(self, window=50):
        records = self._load_records()
        evaluated = [r for r in records if r['predicted_correctly'] is not None]

        if len(evaluated) < window * 2:
            return False, None

        recent = evaluated[-window:]
        previous = evaluated[-window*2:-window]

        recent_acc = sum(1 for r in recent if r['predicted_correctly']) / len(recent)
        previous_acc = sum(1 for r in previous if r['predicted_correctly']) / len(previous)

        drift = previous_acc - recent_acc
        needs_retrain = drift > self.drift_threshold

        return needs_retrain, {
            'recent_accuracy': recent_acc,
            'previous_accuracy': previous_acc,
            'drift': drift,
            'threshold': self.drift_threshold,
        }

    def should_retrain(self):
        accuracy = self.get_accuracy(window=self.retrain_window)
        if accuracy is not None and accuracy < self.min_accuracy:
            return True, f'Accuracy {accuracy:.1%} below minimum {self.min_accuracy:.1%}'

        needs_drift, drift_info = self.check_drift()
        if needs_drift:
            return True, f'Drift detected: {drift_info}'

        return False, 'Model performing adequately'

    def get_summary(self):
        records = self._load_records()
        if len(records) == 0:
            return {'total_signals': 0}

        evaluated = [r for r in records if r['predicted_correctly'] is not None]
        correct = sum(1 for r in evaluated if r['predicted_correctly'])

        return {
            'total_signals': len(records),
            'evaluated': len(evaluated),
            'correct': correct,
            'accuracy': correct / len(evaluated) if evaluated else None,
            'avg_confidence': np.mean([r['confidence'] for r in records]),
            'long_signals': sum(1 for r in records if r['signal'] == 'LONG'),
            'short_signals': sum(1 for r in records if r['signal'] == 'SHORT'),
        }

    def _append_record(self, record):
        filepath = os.path.join(self.data_dir, 'signal_log.jsonl')
        with open(filepath, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    def _load_records(self):
        filepath = os.path.join(self.data_dir, 'signal_log.jsonl')
        if not os.path.exists(filepath):
            return []
        records = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def reset(self):
        filepath = os.path.join(self.data_dir, 'signal_log.jsonl')
        if os.path.exists(filepath):
            os.remove(filepath)

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import joblib
import os
import json
from datetime import datetime


class EnsembleTrainer:
    def __init__(self, config=None):
        cfg = config or {}
        self.n_windows = cfg.get('n_windows', 30)
        self.train_months = cfg.get('train_months', 6)
        self.test_months = cfg.get('test_months', 1)
        self.embargo_bars = cfg.get('embargo_bars', 20)
        self.min_train_bars = cfg.get('min_train_bars', 50000)
        self.models_dir = cfg.get('models_dir', 'models')

    def train_walk_forward(self, X, y, timestamps):
        self._import_models()

        windows = self._create_windows(timestamps)
        results = []
        all_predictions = []

        for i, (train_idx, test_idx) in enumerate(windows):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            ts_test = timestamps[test_idx]

            meta_model, base_models = self._train_window(X_train, y_train)

            base_preds = np.column_stack([m.predict_proba(X_test)[:, 1] for m in base_models])
            meta_pred = np.clip(meta_model.predict(base_preds), 0, 1)

            window_result = self._evaluate_window(y_test, meta_pred, base_preds, ts_test)
            window_result['window'] = i
            results.append(window_result)

            all_predictions.append(pd.DataFrame({
                'timestamp': ts_test,
                'y_true': y_test,
                'y_pred': meta_pred,
                'window': i,
            }))

            print(f'  Window {i+1}/{len(windows)}: acc={window_result["accuracy"]:.3f} '
                  f'prec={window_result["precision"]:.3f} '
                  f'recall={window_result["recall"]:.3f}')

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        summary = self._summarize_results(results)

        return summary, predictions_df, results

    def _create_windows(self, timestamps):
        timestamps = pd.to_datetime(timestamps)
        min_date = timestamps.min()
        max_date = timestamps.max()
        total_days = (max_date - min_date).days

        train_days = self.train_months * 30
        test_days = self.test_months * 30
        window_days = train_days + test_days

        windows = []
        for i in range(self.n_windows):
            window_start = min_date + pd.Timedelta(days=i * test_days)
            train_end = window_start + pd.Timedelta(days=train_days)
            test_end = train_end + pd.Timedelta(days=test_days)

            if test_end > max_date:
                break

            train_mask = (timestamps >= window_start) & (timestamps < train_end)
            test_mask = (timestamps >= train_end) & (timestamps < test_end)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) >= self.min_train_bars and len(test_idx) > 0:
                windows.append((train_idx, test_idx))

        return windows

    def _train_window(self, X_train, y_train):
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, verbosity=0
        )
        lgbm = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1
        )

        xgb.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)

        base_models = [xgb, lgbm]

        try:
            from catboost import CatBoostClassifier
            catboost = CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1,
                random_seed=42, verbose=0
            )
            catboost.fit(X_train, y_train)
            base_models.append(catboost)
        except ImportError:
            pass

        meta_features = np.column_stack([m.predict_proba(X_train)[:, 1] for m in base_models])

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_features, y_train)

        return meta_model, base_models

    def _evaluate_window(self, y_true, y_pred, base_preds, timestamps):
        y_pred_binary = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        ll = log_loss(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'log_loss': ll,
            'n_samples': len(y_true),
            'n_positive': int(y_true.sum()),
            'date_range': f'{timestamps.min()} to {timestamps.max()}',
        }

    def _summarize_results(self, results):
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'log_loss']
        summary = {}
        for m in metrics:
            values = [r[m] for r in results]
            summary[f'{m}_mean'] = np.mean(values)
            summary[f'{m}_std'] = np.std(values)
            summary[f'{m}_min'] = np.min(values)
            summary[f'{m}_max'] = np.max(values)

        summary['n_windows'] = len(results)
        summary['total_samples'] = sum(r['n_samples'] for r in results)
        return summary

    def save_models(self, meta_model, base_models, summary, prefix='ensemble'):
        os.makedirs(self.models_dir, exist_ok=True)

        model_names = ['xgboost', 'lightgbm']
        if len(base_models) > 2:
            model_names.append('catboost')

        for i, model in enumerate(base_models):
            path = os.path.join(self.models_dir, f'{prefix}_{model_names[i]}.joblib')
            joblib.dump(model, path)

        joblib.dump(meta_model, os.path.join(self.models_dir, f'{prefix}_meta.joblib'))

        with open(os.path.join(self.models_dir, f'{prefix}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def load_models(self, prefix='ensemble'):
        meta_model = joblib.load(os.path.join(self.models_dir, f'{prefix}_meta.joblib'))
        base_models = [
            joblib.load(os.path.join(self.models_dir, f'{prefix}_xgboost.joblib')),
            joblib.load(os.path.join(self.models_dir, f'{prefix}_lightgbm.joblib')),
        ]
        catboost_path = os.path.join(self.models_dir, f'{prefix}_catboost.joblib')
        if os.path.exists(catboost_path):
            base_models.append(joblib.load(catboost_path))
        return meta_model, base_models

    def predict(self, X, meta_model, base_models):
        base_preds = np.column_stack([m.predict_proba(X)[:, 1] for m in base_models])
        meta_pred = meta_model.predict(base_preds)
        return meta_pred, base_preds

    def _import_models(self):
        pass

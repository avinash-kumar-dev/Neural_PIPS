import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_pipeline import FeaturePipeline
from src.labels import TripleBarrierLabeler
from src.rule_filters import RuleFilters
from src.feature_engineering import TimeFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODELS_DIR = 'models/v7'


def load_data():
    print("Loading data...")
    m5 = pd.read_parquet('data/eurusd/raw/m5_ohlcv.parquet')
    old_features = pd.read_parquet('data/eurusd/features_m5_v6_raw.parquet')

    if 'time' not in old_features.columns:
        old_features['time'] = m5['time'].values

    print(f"  M5: {len(m5):,} bars | {m5.time.min()} to {m5.time.max()}")
    print(f"  Old features: {old_features.shape[1]} cols")

    time_feat = TimeFeatures().compute(m5['time'])
    print(f"  Time features: {time_feat.shape[1]} cols")

    feat_cols_old = [c for c in old_features.columns if c != 'time']
    raw_features = pd.concat([
        old_features[feat_cols_old].reset_index(drop=True),
        time_feat.reset_index(drop=True),
        old_features[['time']].reset_index(drop=True),
    ], axis=1)

    print(f"  Combined features: {raw_features.shape[1] - 1} cols")
    print(f"  Feature cols: {list(raw_features.columns[:5])}... + {list(raw_features.columns[-5:])}")

    return m5, raw_features


def generate_labels(m5, raw_features):
    print("\nGenerating labels...")
    t0 = time.time()

    labeler = TripleBarrierLabeler(
        tp_mult=4.5, sl_mult=1.5, horizon=20,
        min_tp_pips=9, min_sl_pips=3
    )

    labeled = labeler.label_dataframe(m5)
    stats = labeler.get_stats(labeled)
    print(f"  Total labeled: {stats['total_labeled']:,}")
    print(f"  LONG: {stats['long_signals']:,} ({stats['long_pct']:.1f}%)")
    print(f"  SHORT: {stats['short_signals']:,} ({stats['short_pct']:.1f}%)")
    print(f"  NO-TRADE: {stats['no_trade']:,}")

    label_time = time.time() - t0
    print(f"  Label generation: {label_time:.1f}s")

    return labeled


def apply_filters(labeled, raw_features):
    print("\nApplying rule filters...")
    rf = RuleFilters()
    filtered = rf.apply_all(labeled, raw_features)
    stats = rf.get_stats(filtered)
    print(f"  Passed: {stats['passed']:,} / {stats['total']:,} ({100-stats['rejection_rate']:.1f}%)")

    return filtered


def prepare_training_data(raw_features, filtered):
    print("\nPreparing training data...")

    timestamps = pd.to_datetime(filtered['time'])
    june_mask = timestamps >= pd.Timestamp('2026-06-01')
    train_mask = ~june_mask & (filtered['label'] != -1)

    feat_cols = [c for c in raw_features.columns if c != 'time']
    X = raw_features.loc[train_mask, feat_cols].values.astype(float)
    y = filtered.loc[train_mask, 'label'].values.astype(int)
    ts = timestamps[train_mask].values

    print(f"  Training bars: {len(X):,}")
    print(f"  Features: {len(feat_cols)}")
    print(f"  LONG: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
    print(f"  SHORT: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")

    long_count = (y == 1).sum()
    short_count = (y == 0).sum()
    scale_pos_weight = short_count / long_count
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    june_mask_valid = june_mask & (filtered['label'] != -1)
    X_june = raw_features.loc[june_mask_valid, feat_cols].values.astype(float)
    y_june = filtered.loc[june_mask_valid, 'label'].values.astype(int)
    ts_june = timestamps[june_mask_valid].values

    print(f"  June holdout: {len(X_june):,} bars")

    return X, y, ts, X_june, y_june, ts_june, feat_cols, scale_pos_weight


def train_walk_forward(X, y, ts, n_windows=12, embargo_bars=48, scale_pos_weight=5.5):
    print(f"\nWalk-forward training ({n_windows} windows, embargo={embargo_bars})...")
    print("  Class imbalance fix: scale_pos_weight={:.2f}".format(scale_pos_weight))
    print("  Meta-learner: weighted probability average (Nelder-Mead)")
    t0 = time.time()

    timestamps = pd.to_datetime(ts)
    min_date = timestamps.min()
    max_date = timestamps.max()

    train_months = 6
    test_months = 1
    train_days = train_months * 30
    test_days = test_months * 30

    windows = []
    for i in range(n_windows):
        window_start = min_date + pd.Timedelta(days=i * test_days)
        train_end = window_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days) + pd.Timedelta(days=embargo_bars // 24)

        if test_end > max_date:
            break

        train_mask = (timestamps >= window_start) & (timestamps < train_end)
        test_mask = (timestamps >= train_end + pd.Timedelta(days=embargo_bars // 24)) & (timestamps < test_end)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) >= 20000 and len(test_idx) > 0:
            windows.append((train_idx, test_idx))

    print(f"  Windows created: {len(windows)}")

    if len(windows) == 0:
        raise ValueError("No walk-forward windows created.")

    results = []
    all_predictions = []

    for i, (train_idx, test_idx) in enumerate(windows):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        ts_test = timestamps[test_idx]

        pipeline = FeaturePipeline(use_robust_scaler=True)
        X_train_scaled = pipeline.fit_transform(X_train)
        X_test_scaled = pipeline.transform(X_test)

        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        xgb = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss', random_state=42, verbosity=0
        )
        xgb.fit(X_train_scaled, y_train)

        lgbm = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42, verbosity=-1
        )
        lgbm.fit(X_train_scaled, y_train)

        try:
            from catboost import CatBoostClassifier
            cb = CatBoostClassifier(
                iterations=300, depth=8, learning_rate=0.05,
                class_weights={0: 1.0, 1: scale_pos_weight},
                random_seed=42, verbose=0
            )
            cb.fit(X_train_scaled, y_train)
            base_models = [xgb, lgbm, cb]
        except ImportError:
            base_models = [xgb, lgbm]

        train_proba = np.column_stack([m.predict_proba(X_train_scaled)[:, 1] for m in base_models])
        weights = np.array([1.0 / len(base_models)] * len(base_models))

        from scipy.optimize import minimize

        def neg_log_loss(w):
            w = np.abs(w) / np.sum(np.abs(w))
            meta_proba = train_proba @ w
            meta_proba = np.clip(meta_proba, 1e-7, 1 - 1e-7)
            ll = y_train * np.log(meta_proba) + (1 - y_train) * np.log(1 - meta_proba)
            return -np.mean(ll)

        opt = minimize(neg_log_loss, weights, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-6})
        weights = np.abs(opt.x) / np.sum(np.abs(opt.x))

        print(f"  Window {i+1}: weights = [{', '.join(f'{w:.3f}' for w in weights)}]")

        test_proba = np.column_stack([m.predict_proba(X_test_scaled)[:, 1] for m in base_models])
        meta_pred = test_proba @ weights
        ml_pred = (meta_pred > 0.5).astype(int)

        xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
        lgbm_proba = lgbm.predict_proba(X_test_scaled)[:, 1]
        catboost_proba = base_models[2].predict_proba(X_test_scaled)[:, 1] if len(base_models) > 2 else lgbm_proba
        ensemble_std = np.std([xgb_proba, lgbm_proba, catboost_proba], axis=0)

        accuracy = accuracy_score(y_test, ml_pred)
        precision = precision_score(y_test, ml_pred, zero_division=0)
        recall = recall_score(y_test, ml_pred, zero_division=0)
        f1 = f1_score(y_test, ml_pred, zero_division=0)

        long_acc = ((ml_pred == 1) & (y_test == 1)).sum() / max((ml_pred == 1).sum(), 1) * 100
        short_acc = ((ml_pred == 0) & (y_test == 0)).sum() / max((ml_pred == 0).sum(), 1) * 100

        results.append({
            'window': i,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'long_accuracy': long_acc,
            'short_accuracy': short_acc,
            'long_signals': int((ml_pred == 1).sum()),
            'short_signals': int((ml_pred == 0).sum()),
            'n_samples': len(y_test),
            'date_range': f'{ts_test.min()} to {ts_test.max()}',
        })

        all_predictions.append(pd.DataFrame({
            'timestamp': ts_test,
            'y_true': y_test,
            'y_pred': meta_pred,
            'ml_pred': ml_pred,
            'ensemble_std': ensemble_std,
            'window': i,
        }))

        print(f"  Window {i+1}/{len(windows)}: acc={accuracy:.3f} prec={precision:.3f} recall={recall:.3f} "
              f"L_acc={long_acc:.1f}%({(ml_pred==1).sum()}) S_acc={short_acc:.1f}%({(ml_pred==0).sum()})")

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    summary = {}
    for m in metrics:
        values = [r[m] for r in results]
        summary[f'{m}_mean'] = np.mean(values)
        summary[f'{m}_std'] = np.std(values)

    long_accs = [r['long_accuracy'] for r in results if r['long_signals'] > 0]
    short_accs = [r['short_accuracy'] for r in results if r['short_signals'] > 0]
    summary['long_accuracy_mean'] = np.mean(long_accs) if long_accs else 0
    summary['short_accuracy_mean'] = np.mean(short_accs) if short_accs else 0

    summary['n_windows'] = len(results)
    summary['total_samples'] = sum(r['n_samples'] for r in results)

    train_time = time.time() - t0
    print(f"\n  Walk-forward done in {train_time:.1f}s")
    print(f"  Accuracy: {summary['accuracy_mean']*100:.1f}% (+/- {summary['accuracy_std']*100:.1f}%)")
    print(f"  LONG accuracy: {summary['long_accuracy_mean']:.1f}%")
    print(f"  SHORT accuracy: {summary['short_accuracy_mean']:.1f}%")

    return summary, predictions_df, results, train_time


def train_final_models(X, y, scale_pos_weight=5.5):
    print("\nTraining final models on all training data...")
    t0 = time.time()

    pipeline = FeaturePipeline(use_robust_scaler=True)
    X_scaled = pipeline.fit_transform(X)

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    xgb.fit(X_scaled, y)

    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42, verbosity=-1
    )
    lgbm.fit(X_scaled, y)

    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(
            iterations=300, depth=8, learning_rate=0.05,
            class_weights={0: 1.0, 1: scale_pos_weight},
            random_seed=42, verbose=0
        )
        cb.fit(X_scaled, y)
        base_models = [xgb, lgbm, cb]
    except ImportError:
        base_models = [xgb, lgbm]

    train_proba = np.column_stack([m.predict_proba(X_scaled)[:, 1] for m in base_models])

    from scipy.optimize import minimize
    weights = np.array([1.0 / len(base_models)] * len(base_models))

    def neg_log_loss(w):
        w = np.abs(w) / np.sum(np.abs(w))
        meta_proba = train_proba @ w
        meta_proba = np.clip(meta_proba, 1e-7, 1 - 1e-7)
        ll = y * np.log(meta_proba) + (1 - y) * np.log(1 - meta_proba)
        return -np.mean(ll)

    opt = minimize(neg_log_loss, weights, method='Nelder-Mead',
                  options={'maxiter': 1000, 'xatol': 1e-6})
    weights = np.abs(opt.x) / np.sum(np.abs(opt.x))

    print(f"  Optimized weights: [{', '.join(f'{w:.3f}' for w in weights)}]")

    train_time = time.time() - t0
    print(f"  Final models trained in {train_time:.1f}s")

    return xgb, lgbm, base_models[2] if len(base_models) > 2 else None, weights, pipeline


def test_holdout(xgb, lgbm, catboost, weights, pipeline, X_june, y_june, ts_june):
    print("\n" + "=" * 60)
    print("  HOLDOUT TEST: June 2026")
    print("=" * 60)

    X_june_scaled = pipeline.transform(X_june)

    base_preds = [xgb.predict_proba(X_june_scaled)[:, 1], lgbm.predict_proba(X_june_scaled)[:, 1]]
    if catboost is not None:
        base_preds.append(catboost.predict_proba(X_june_scaled)[:, 1])
    base_preds = np.column_stack(base_preds)

    meta_pred = base_preds @ weights
    ml_pred = (meta_pred > 0.5).astype(int)

    xgb_proba = xgb.predict_proba(X_june_scaled)[:, 1]
    lgbm_proba = lgbm.predict_proba(X_june_scaled)[:, 1]
    catboost_proba = catboost.predict_proba(X_june_scaled)[:, 1] if catboost is not None else lgbm_proba
    ensemble_std = np.std([xgb_proba, lgbm_proba, catboost_proba], axis=0)

    base_acc = accuracy_score(y_june, ml_pred)
    print(f"\n  Base accuracy: {base_acc*100:.1f}%")
    print(f"  Weights: [{', '.join(f'{w:.3f}' for w in weights)}]")

    print(f"\n  Meta proba stats: mean={meta_pred.mean():.3f} min={meta_pred.min():.3f} max={meta_pred.max():.3f}")
    print(f"  ml_pred=0 (SHORT): {(ml_pred==0).sum()}, ml_pred=1 (LONG): {(ml_pred==1).sum()}")

    june_timestamps = pd.to_datetime(ts_june)
    hours = june_timestamps.hour
    dow = june_timestamps.weekday

    sessions = [(7, 9), (13, 16), (17, 20)]
    session_mask = np.zeros(len(hours), dtype=bool)
    for sh, eh in sessions:
        session_mask |= ((hours >= sh) & (hours < eh) & (dow < 5))

    print(f"\n  Session bars: {session_mask.sum()} / {len(session_mask)}")

    print(f"\n  {'T':>4s} {'Dir':>12s} {'Signals':>8s} {'Acc%':>6s} {'Wins':>5s} {'Loss':>5s} {'PnL':>8s}")
    print("  " + "-" * 55)

    results = []
    for threshold in [60, 65, 70, 75, 78, 80, 82, 85]:
        thresh = threshold / 100
        long_mask = session_mask & (meta_pred >= thresh) & (ml_pred == 1)
        short_mask = session_mask & ((1 - meta_pred) >= thresh) & (ml_pred == 0)

        for label, mask, expected_dir in [("LONG", long_mask, 1), ("SHORT", short_mask, 0)]:
            gen = mask.sum()
            if gen > 0:
                labels = y_june[mask]
                correct = (labels == expected_dir).sum()
                acc = correct / gen * 100

                if expected_dir == 1:
                    wins = correct
                    losses = gen - correct
                else:
                    wins = correct
                    losses = gen - correct

                avg_tp = 16.5
                avg_sl = 6.4
                pnl = wins * avg_tp - losses * avg_sl
            else:
                acc = 0
                wins = 0
                losses = 0
                pnl = 0

            results.append({
                'threshold': threshold,
                'direction': label,
                'signals': int(gen),
                'accuracy': round(acc, 1),
                'wins': wins,
                'losses': losses,
                'pnl_pips': round(pnl, 1),
            })

            if gen > 0:
                print(f"  {threshold:4d} {label:>12s} {gen:8d} {acc:6.1f}% {wins:5d} {losses:5d} {pnl:+8.1f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('data/eurusd/v7_holdout_results.csv', index=False)
    print(f"\n  Saved to data/eurusd/v7_holdout_results.csv")

    return results_df


def save_models(xgb, lgbm, catboost, weights, pipeline, wf_summary, holdout_results, train_time, feat_cols, X, y, ts):
    print("\nSaving models...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost.joblib'))
    joblib.dump(lgbm, os.path.join(MODELS_DIR, 'lightgbm.joblib'))
    if catboost is not None:
        joblib.dump(catboost, os.path.join(MODELS_DIR, 'catboost.joblib'))
    joblib.dump(weights, os.path.join(MODELS_DIR, 'ensemble_weights.joblib'))
    pipeline.save(os.path.join(MODELS_DIR, 'feature_pipeline.joblib'))

    summary = {
        'description': 'V7: Phase 1 fixes - class imbalance, 94 features, confluence, spread/vol gates',
        'train_method': 'walk_forward_12_windows_v7',
        'train_period': f'{pd.Timestamp(ts.min())} to {pd.Timestamp(ts.max())}',
        'n_features': len(feat_cols),
        'new_features': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'minute_of_day',
                         'is_london_open', 'is_ny_afternoon', 'is_any_session',
                         'london_progress', 'overlap_progress', 'ny_progress',
                         'hour_of_day', 'is_weekend'],
        'has_h4_regime': True,
        'class_weights': True,
        'scale_pos_weight': round((y == 0).sum() / (y == 1).sum(), 2),
        'june_excluded': True,
        'normalization_leak_fixed': True,
        'catboost_added': catboost is not None,
        'meta_learner': 'weighted_average_nelder_mead',
        'ensemble_weights': [round(w, 4) for w in weights],
        'scaler': 'RobustScaler',
        'embargo_bars': 48,
        'confluence_min': 2,
        'ml_gate': 0.75,
        'n_windows': wf_summary['n_windows'],
        'walk_forward_accuracy_mean': round(wf_summary['accuracy_mean'], 4),
        'walk_forward_accuracy_std': round(wf_summary['accuracy_std'], 4),
        'long_accuracy_mean': round(wf_summary.get('long_accuracy_mean', 0), 4),
        'short_accuracy_mean': round(wf_summary.get('short_accuracy_mean', 0), 4),
        'total_train_bars': int(len(X)),
        'long_count': int((y == 1).sum()),
        'short_count': int((y == 0).sum()),
        'train_time_seconds': round(train_time, 1),
    }

    with open(os.path.join(MODELS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved: {MODELS_DIR}/xgboost.joblib")
    print(f"  Saved: {MODELS_DIR}/lightgbm.joblib")
    if catboost is not None:
        print(f"  Saved: {MODELS_DIR}/catboost.joblib")
    print(f"  Saved: {MODELS_DIR}/ensemble_weights.joblib")
    print(f"  Saved: {MODELS_DIR}/feature_pipeline.joblib")
    print(f"  Saved: {MODELS_DIR}/summary.json")


def main():
    print("=" * 60)
    print("  V7 TRAINING")
    print("  Phase 1: class imbalance fix + 94 features + confluence")
    print("=" * 60)

    m5, raw_features = load_data()
    labeled = generate_labels(m5, raw_features)
    filtered = apply_filters(labeled, raw_features)
    X, y, ts, X_june, y_june, ts_june, feat_cols, scale_pos_weight = prepare_training_data(raw_features, filtered)

    wf_summary, predictions_df, wf_results, train_time = train_walk_forward(
        X, y, ts, n_windows=12, embargo_bars=48, scale_pos_weight=scale_pos_weight
    )

    xgb, lgbm, catboost, weights, pipeline = train_final_models(X, y, scale_pos_weight)

    holdout_results = test_holdout(xgb, lgbm, catboost, weights, pipeline, X_june, y_june, ts_june)

    save_models(xgb, lgbm, catboost, weights, pipeline, wf_summary, holdout_results, train_time, feat_cols, X, y, ts)

    print("\n" + "=" * 60)
    print("  V7 TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

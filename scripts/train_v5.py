import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureEngineer
from src.labels import TripleBarrierLabeler
from src.rule_filters import RuleFilters
from src.feature_pipeline import FeaturePipeline
from src.signal_engine import SignalEngine
from src.confidence_scorer import ConfidenceScorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

MODELS_DIR = 'models/v5_phase1'


def load_raw_data():
    print("Loading raw data...")
    m5 = pd.read_parquet('data/eurusd/raw/m5_ohlcv.parquet')
    m15 = pd.read_parquet('data/eurusd/raw/m15_ohlcv.parquet')
    h1 = pd.read_parquet('data/eurusd/raw/h1_ohlcv.parquet')
    h4 = pd.read_parquet('data/eurusd/raw/h4_ohlcv.parquet')

    print(f"  M5: {len(m5):,} bars | {m5.time.min()} to {m5.time.max()}")
    print(f"  M15: {len(m15):,} bars | {m15.time.min()} to {m15.time.max()}")
    print(f"  H1: {len(h1):,} bars | {h1.time.min()} to {h1.time.max()}")
    print(f"  H4: {len(h4):,} bars | {h4.time.min()} to {h4.time.max()}")

    return m5, m15, h1, h4


def generate_features(m5, m15, h1, h4):
    print("\nGenerating features...")
    t0 = time.time()

    fe = FeatureEngineer()
    raw_features = fe.compute_all(m5, m15, h1, h4)
    raw_features['time'] = m5['time'].values

    print(f"  Raw features: {raw_features.shape}")

    feat_time = time.time() - t0
    print(f"  Feature generation: {feat_time:.1f}s")

    return raw_features


def generate_labels(m5, raw_features):
    print("\nGenerating labels...")
    t0 = time.time()

    labeler = TripleBarrierLabeler(
        tp_mult=4.5, sl_mult=1.5, horizon=20,
        min_tp_pips=9, max_tp_pips=45,
        min_sl_pips=3, max_sl_pips=15
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
    print(f"  LONG: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
    print(f"  SHORT: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")

    june_mask_valid = june_mask & (filtered['label'] != -1)
    X_june = raw_features.loc[june_mask_valid, feat_cols].values.astype(float)
    y_june = filtered.loc[june_mask_valid, 'label'].values.astype(int)
    ts_june = timestamps[june_mask_valid].values

    print(f"  June holdout: {len(X_june):,} bars")

    return X, y, ts, X_june, y_june, ts_june, feat_cols


def compute_class_weights(y):
    n_long = (y == 1).sum()
    n_short = (y == 0).sum()
    total = len(y)
    weight_long = total / (2 * n_long)
    weight_short = total / (2 * n_short)
    print(f"  Class weights: LONG={weight_long:.2f}, SHORT={weight_short:.2f}")
    return weight_long, weight_short


def train_walk_forward(X, y, ts, n_windows=12, embargo_bars=48):
    print(f"\nWalk-forward training ({n_windows} windows, embargo={embargo_bars})...")
    t0 = time.time()

    timestamps = pd.to_datetime(ts)
    min_date = timestamps.min()
    max_date = timestamps.max()

    train_months = 6
    test_months = 1
    train_days = train_months * 30
    test_days = test_months * 30

    weight_long, weight_short = compute_class_weights(y)

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

        sw_train = np.where(y_train == 1, weight_long, weight_short)

        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        xgb = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=weight_long/weight_short,
            eval_metric='logloss', random_state=42, verbosity=0
        )
        xgb.fit(X_train_scaled, y_train, sample_weight=sw_train)

        lgbm = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True,
            random_state=42, verbosity=-1
        )
        lgbm.fit(X_train_scaled, y_train, sample_weight=sw_train)

        try:
            from catboost import CatBoostClassifier
            catboost = CatBoostClassifier(
                iterations=300, depth=8, learning_rate=0.05,
                auto_class_weights='Balanced',
                random_seed=42, verbose=0
            )
            catboost.fit(X_train_scaled, y_train)
            base_models = [xgb, lgbm, catboost]
        except ImportError:
            base_models = [xgb, lgbm]

        meta_features = np.column_stack([m.predict_proba(X_train_scaled)[:, 1] for m in base_models])
        meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        meta_model.fit(meta_features, y_train)

        test_base = np.column_stack([m.predict_proba(X_test_scaled)[:, 1] for m in base_models])
        meta_pred = meta_model.predict_proba(test_base)[:, 1]
        ml_pred = (meta_pred > 0.5).astype(int)

        xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
        lgbm_proba = lgbm.predict_proba(X_test_scaled)[:, 1]
        catboost_proba = base_models[2].predict_proba(X_test_scaled)[:, 1] if len(base_models) > 2 else lgbm_proba
        ensemble_std = np.std([xgb_proba, lgbm_proba, catboost_proba], axis=0)

        accuracy = accuracy_score(y_test, ml_pred)
        precision = precision_score(y_test, ml_pred, zero_division=0)
        recall = recall_score(y_test, ml_pred, zero_division=0)
        f1 = f1_score(y_test, ml_pred, zero_division=0)

        results.append({
            'window': i,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': len(y_test),
            'date_range': f'{ts_test.min()} to {ts_test.max()}',
        })

        all_predictions.append(pd.DataFrame({
            'timestamp': ts_test,
            'y_true': y_test,
            'y_pred': meta_pred,
            'ensemble_std': ensemble_std,
            'window': i,
        }))

        print(f"  Window {i+1}/{len(windows)}: acc={accuracy:.3f} prec={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    summary = {}
    for m in metrics:
        values = [r[m] for r in results]
        summary[f'{m}_mean'] = np.mean(values)
        summary[f'{m}_std'] = np.std(values)

    summary['n_windows'] = len(results)
    summary['total_samples'] = sum(r['n_samples'] for r in results)

    train_time = time.time() - t0
    print(f"\n  Walk-forward done in {train_time:.1f}s")
    print(f"  Accuracy: {summary['accuracy_mean']*100:.1f}% (+/- {summary['accuracy_std']*100:.1f}%)")

    return summary, predictions_df, results, train_time


def train_final_models(X, y, weight_long, weight_short):
    print("\nTraining final models on all training data...")
    t0 = time.time()

    pipeline = FeaturePipeline(use_robust_scaler=True)
    X_scaled = pipeline.fit_transform(X)

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    sw = np.where(y == 1, weight_long, weight_short)

    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=weight_long/weight_short,
        eval_metric='logloss', random_state=42, verbosity=0
    )
    xgb.fit(X_scaled, y, sample_weight=sw)

    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True,
        random_state=42, verbosity=-1
    )
    lgbm.fit(X_scaled, y, sample_weight=sw)

    try:
        from catboost import CatBoostClassifier
        catboost = CatBoostClassifier(
            iterations=300, depth=8, learning_rate=0.05,
            auto_class_weights='Balanced',
            random_seed=42, verbose=0
        )
        catboost.fit(X_scaled, y)
        base_models = [xgb, lgbm, catboost]
    except ImportError:
        base_models = [xgb, lgbm]

    meta_features = np.column_stack([m.predict_proba(X_scaled)[:, 1] for m in base_models])
    meta_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    meta_model.fit(meta_features, y)

    train_time = time.time() - t0
    print(f"  Final models trained in {train_time:.1f}s")

    return xgb, lgbm, base_models[2] if len(base_models) > 2 else None, meta_model, pipeline


def test_holdout(xgb, lgbm, catboost, meta_model, pipeline, X_june, y_june, ts_june):
    print("\n" + "=" * 60)
    print("  HOLDOUT TEST: June 2026")
    print("=" * 60)

    X_june_scaled = pipeline.transform(X_june)

    base_preds = [xgb.predict_proba(X_june_scaled)[:, 1], lgbm.predict_proba(X_june_scaled)[:, 1]]
    if catboost is not None:
        base_preds.append(catboost.predict_proba(X_june_scaled)[:, 1])
    base_preds = np.column_stack(base_preds)

    meta_pred = meta_model.predict_proba(base_preds)[:, 1]
    ml_pred = (meta_pred > 0.5).astype(int)

    xgb_proba = xgb.predict_proba(X_june_scaled)[:, 1]
    lgbm_proba = lgbm.predict_proba(X_june_scaled)[:, 1]
    catboost_proba = catboost.predict_proba(X_june_scaled)[:, 1] if catboost is not None else lgbm_proba
    ensemble_std = np.std([xgb_proba, lgbm_proba, catboost_proba], axis=0)

    base_acc = accuracy_score(y_june, ml_pred)
    print(f"\n  Base accuracy: {base_acc*100:.1f}%")

    june_timestamps = pd.to_datetime(ts_june)
    hours = june_timestamps.hour
    dow = june_timestamps.weekday
    session_mask = (hours >= 13) & (hours < 16) & (dow < 5)

    print(f"\n  {'T':>4s} {'Dir':>12s} {'Signals':>8s} {'Acc%':>6s} {'Agreement':>10s}")
    print("  " + "-" * 50)

    results = []
    for threshold in [60, 70, 75, 78, 80, 82, 85]:
        for agreement_thresh in [0.0, 0.2, 0.3]:
            mask = session_mask & (meta_pred >= threshold/100) & (ensemble_std <= agreement_thresh) if agreement_thresh > 0 else session_mask & (meta_pred >= threshold/100)

            for label, mask_dir in [("LONG+SHORT", mask), ("LONG-only", mask & (ml_pred == 1)), ("SHORT-only", mask & (ml_pred == 0))]:
                gen = mask_dir.sum()
                if gen > 0:
                    labels = y_june[mask_dir]
                    dirs = np.where(ml_pred[mask_dir] == 1, 'LONG', 'SHORT')
                    correct = ((dirs == 'LONG') & (labels == 1)).sum() + ((dirs == 'SHORT') & (labels == 0)).sum()
                    acc = correct / gen * 100
                else:
                    acc = 0
                results.append({
                    'threshold': threshold,
                    'agreement': agreement_thresh,
                    'direction': label,
                    'signals': int(gen),
                    'accuracy': round(acc, 1)
                })
                if label in ['LONG-only', 'SHORT-only'] and gen > 0 and agreement_thresh in [0.0, 0.3]:
                    print(f"  {threshold:4d} {label:>12s} {gen:8d} {acc:6.1f}% {agreement_thresh:10.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('data/eurusd/v5_holdout_results.csv', index=False)
    print(f"\n  Saved to data/eurusd/v5_holdout_results.csv")

    return results_df


def save_models(xgb, lgbm, catboost, meta_model, pipeline, wf_summary, holdout_results, train_time, feat_cols, X, y, ts):
    print("\nSaving models...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost.joblib'))
    joblib.dump(lgbm, os.path.join(MODELS_DIR, 'lightgbm.joblib'))
    if catboost is not None:
        joblib.dump(catboost, os.path.join(MODELS_DIR, 'catboost.joblib'))
    joblib.dump(meta_model, os.path.join(MODELS_DIR, 'meta.joblib'))
    pipeline.save(os.path.join(MODELS_DIR, 'feature_pipeline.joblib'))

    summary = {
        'description': 'Phase 1 fixes: normalization leak fixed, CatBoost added, LogisticRegression meta, embargo=48, RobustScaler',
        'train_method': 'walk_forward_12_windows_phase1',
        'train_period': f'{pd.Timestamp(ts.min())} to {pd.Timestamp(ts.max())}',
        'n_features': len(feat_cols),
        'has_h4_regime': True,
        'class_imbalance_fixed': True,
        'june_excluded': True,
        'normalization_leak_fixed': True,
        'catboost_added': catboost is not None,
        'meta_learner': 'LogisticRegression',
        'scaler': 'RobustScaler',
        'embargo_bars': 48,
        'n_windows': wf_summary['n_windows'],
        'walk_forward_accuracy_mean': round(wf_summary['accuracy_mean'], 4),
        'walk_forward_accuracy_std': round(wf_summary['accuracy_std'], 4),
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
    print(f"  Saved: {MODELS_DIR}/meta.joblib")
    print(f"  Saved: {MODELS_DIR}/feature_pipeline.joblib")
    print(f"  Saved: {MODELS_DIR}/summary.json")


def main():
    print("=" * 60)
    print("  V5 PHASE 1 TRAINING")
    print("  Normalization fix + CatBoost + LogisticRegression + embargo=48")
    print("=" * 60)

    m5, m15, h1, h4 = load_raw_data()
    raw_features = generate_features(m5, m15, h1, h4)
    labeled = generate_labels(m5, raw_features)
    filtered = apply_filters(labeled, raw_features)
    X, y, ts, X_june, y_june, ts_june, feat_cols = prepare_training_data(raw_features, filtered)

    weight_long, weight_short = compute_class_weights(y)

    wf_summary, predictions_df, wf_results, train_time = train_walk_forward(X, y, ts, n_windows=12, embargo_bars=48)

    xgb, lgbm, catboost, meta_model, pipeline = train_final_models(X, y, weight_long, weight_short)

    holdout_results = test_holdout(xgb, lgbm, catboost, meta_model, pipeline, X_june, y_june, ts_june)

    save_models(xgb, lgbm, catboost, meta_model, pipeline, wf_summary, holdout_results, train_time, feat_cols, X, y, ts)

    print("\n" + "=" * 60)
    print("  V5 PHASE 1 TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

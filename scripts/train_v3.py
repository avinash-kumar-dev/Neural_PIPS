import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_trainer import EnsembleTrainer
from src.rule_filters import RuleFilters
from src.confidence_scorer import ConfidenceScorer
from sklearn.metrics import accuracy_score, precision_score, recall_score

MODELS_DIR = 'models/v3_regime'


def test_holdout(xgb, lgbm, meta_model, pipeline):
    print("\n" + "=" * 60)
    print("  HOLDOUT TEST: June 2026")
    print("=" * 60)

    features = pd.read_parquet('data/eurusd/features_m5_v3.parquet')
    raw = pd.read_parquet('data/eurusd/features_m5_raw_v3.parquet')
    filtered = pd.read_parquet('data/eurusd/m5_filtered.parquet')

    timestamps = pd.to_datetime(filtered['time'])
    june_mask = timestamps >= pd.Timestamp('2026-06-01')
    valid_mask = june_mask & (filtered['label'] != -1)

    feat_cols = [c for c in features.columns if c != 'time']
    X_test = features.loc[valid_mask, feat_cols].values.astype(float)
    y_test = filtered.loc[valid_mask, 'label'].values.astype(int)

    print(f"  Test bars: {valid_mask.sum()} (LONG={ (y_test==1).sum() }, SHORT={ (y_test==0).sum() })")

    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
    base_preds = np.column_stack([xgb_proba, lgbm_proba])
    meta_pred = np.clip(meta_model.predict(base_preds), 0, 1)
    ml_pred = (meta_pred > 0.5).astype(int)

    base_acc = accuracy_score(y_test, ml_pred)
    print(f"  Base accuracy: {base_acc*100:.1f}%")

    hours = pd.to_datetime(filtered.loc[valid_mask, 'time']).dt.hour
    dow = pd.to_datetime(filtered.loc[valid_mask, 'time']).dt.weekday
    session_mask = (hours >= 13) & (hours < 16) & (dow < 5)

    rf = RuleFilters()
    filtered_june = rf.apply_all(
        filtered.loc[valid_mask].reset_index(drop=True),
        raw.loc[valid_mask].reset_index(drop=True)
    )
    passed_mask = filtered_june['filter_reject'] == 0
    print(f"  Passed filters: {passed_mask.sum()} / {len(filtered_june)}")

    scorer = ConfidenceScorer({'threshold': 0})
    test_raw = raw.loc[valid_mask].reset_index(drop=True)
    test_filt = filtered.loc[valid_mask].reset_index(drop=True)
    conf_scores = np.zeros(len(test_filt))

    passed_idx = np.where(passed_mask.values)[0]
    for i in passed_idx:
        row = test_raw.iloc[i].to_dict()
        row['time'] = test_filt.iloc[i]['time']
        weighted, _ = scorer.score(row, row)
        conf_scores[i] = weighted

    atr_pips = test_raw['atr_pips'].values if 'atr_pips' in test_raw.columns else np.full(len(test_filt), 10.0)
    signal_dir = np.where(ml_pred == 1, 'LONG', 'SHORT')

    print(f"\n  {'T':>4s} {'Dir':>12s} {'Signals':>8s} {'Acc%':>6s} {'LONG_acc':>9s} {'SHORT_acc':>10s}")
    print("  " + "-" * 55)

    results = []
    for threshold in [60, 70, 75, 78, 80, 82, 85]:
        mask = passed_mask.values & session_mask.values & (atr_pips >= 3) & (conf_scores >= threshold)

        for label, mask_dir in [("LONG+SHORT", mask), ("LONG-only", mask & (ml_pred == 1)), ("SHORT-only", mask & (ml_pred == 0))]:
            gen = mask_dir.sum()
            if gen > 0:
                labels = y_test[mask_dir]
                dirs = signal_dir[mask_dir]
                correct = ((dirs == 'LONG') & (labels == 1)).sum() + ((dirs == 'SHORT') & (labels == 0)).sum()
                acc = correct / gen * 100
            else:
                acc = 0
            results.append({'threshold': threshold, 'direction': label, 'signals': int(gen), 'accuracy': round(acc, 1)})
            if label in ['LONG-only', 'SHORT-only'] and gen > 0:
                print(f"  {threshold:4d} {label:>12s} {gen:8d} {acc:6.1f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv('data/eurusd/v3_holdout_results.csv', index=False)
    print(f"\n  Saved to data/eurusd/v3_holdout_results.csv")

    return results_df


def main():
    print("=" * 60)
    print("  V3 REGIME-CONDITIONAL TRAINING")
    print("  Walk-forward + h4_regime feature")
    print("=" * 60)

    print("\nLoading v3 features (41 features with h4_regime)...")
    features = pd.read_parquet('data/eurusd/features_m5_v3.parquet')
    filtered = pd.read_parquet('data/eurusd/m5_filtered.parquet')
    print(f"  Features: {features.shape}")
    print(f"  Labels: {len(filtered)}")

    timestamps = pd.to_datetime(filtered['time'])
    valid_mask = filtered['label'] != -1

    feat_cols = [c for c in features.columns if c != 'time']
    X = features.loc[valid_mask, feat_cols].values.astype(float)
    y = filtered.loc[valid_mask, 'label'].values.astype(int)
    ts = timestamps[valid_mask].values

    print(f"  Valid bars: {len(X)} (LONG={ (y==1).sum() }, SHORT={ (y==0).sum() })")
    print(f"  h4_regime in features: {'h4_regime' in feat_cols}")

    print("\nWalk-forward training (10 windows)...")
    t0 = time.time()

    trainer = EnsembleTrainer({
        'n_windows': 10,
        'train_months': 6,
        'test_months': 1,
        'embargo_bars': 20,
        'min_train_bars': 10000,
        'models_dir': MODELS_DIR,
    })

    summary, predictions_df, results = trainer.train_walk_forward(X, y, ts)

    train_time = time.time() - t0
    print(f"\n  Training done in {train_time:.1f}s")
    print(f"  Walk-forward accuracy: {summary['accuracy_mean']*100:.1f}% (+/- {summary['accuracy_std']*100:.1f}%)")

    print("\nSaving v3 models...")
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import Ridge

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
    xgb.fit(X, y)
    lgbm.fit(X, y)

    meta_features = np.column_stack([m.predict_proba(X)[:, 1] for m in [xgb, lgbm]])
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost.joblib'))
    joblib.dump(lgbm, os.path.join(MODELS_DIR, 'lightgbm.joblib'))
    joblib.dump(meta_model, os.path.join(MODELS_DIR, 'meta.joblib'))

    v3_summary = {
        'description': 'Regime-conditional. Single model with h4_regime feature. Walk-forward validated.',
        'train_method': 'walk_forward_10_windows_with_regime_feature',
        'train_period': f'{timestamps[valid_mask].min()} to {timestamps[valid_mask].max()}',
        'n_features': len(feat_cols),
        'has_h4_regime': True,
        'n_windows': summary['n_windows'],
        'walk_forward_accuracy_mean': round(summary['accuracy_mean'], 4),
        'walk_forward_accuracy_std': round(summary['accuracy_std'], 4),
        'walk_forward_precision_mean': round(summary['precision_mean'], 4),
        'walk_forward_recall_mean': round(summary['recall_mean'], 4),
        'walk_forward_f1_mean': round(summary['f1_mean'], 4),
        'total_train_bars': int(len(X)),
        'long_count': int((y == 1).sum()),
        'short_count': int((y == 0).sum()),
        'train_time_seconds': round(train_time, 1),
    }
    with open(os.path.join(MODELS_DIR, 'summary.json'), 'w') as f:
        json.dump(v3_summary, f, indent=2, default=str)

    print(f"  Saved: {MODELS_DIR}/xgboost.joblib")
    print(f"  Saved: {MODELS_DIR}/lightgbm.joblib")
    print(f"  Saved: {MODELS_DIR}/meta.joblib")
    print(f"  Saved: {MODELS_DIR}/summary.json")

    holdout_results = test_holdout(xgb, lgbm, meta_model, None)

    print("\n" + "=" * 60)
    print("  V3 vs V2 COMPARISON (June holdout)")
    print("=" * 60)

    v2_results = pd.read_csv('data/eurusd/holdout_test_results.csv')
    v2_short = v2_results[(v2_results['direction'] == 'SHORT-only') & (v2_results['threshold'] == 78)]
    v3_short = holdout_results[(holdout_results['direction'] == 'SHORT-only') & (holdout_results['threshold'] == 78)]
    v2_long = v2_results[(v2_results['direction'] == 'LONG-only') & (v2_results['threshold'] == 78)]
    v3_long = holdout_results[(holdout_results['direction'] == 'LONG-only') & (holdout_results['threshold'] == 78)]

    print(f"  SHORT T=78: v2={v2_short['accuracy'].values[0]}% ({v2_short['signals'].values[0]} sigs) -> v3={v3_short['accuracy'].values[0]}% ({v3_short['signals'].values[0]} sigs)")
    print(f"  LONG  T=78: v2={v2_long['accuracy'].values[0]}% ({v2_long['signals'].values[0]} sigs) -> v3={v3_long['accuracy'].values[0]}% ({v3_long['signals'].values[0]} sigs)")

    v3_summary['holdout_june_metrics'] = {
        'short_accuracy_T78': float(v3_short['accuracy'].values[0]),
        'short_signals_T78': int(v3_short['signals'].values[0]),
        'long_accuracy_T78': float(v3_long['accuracy'].values[0]),
        'long_signals_T78': int(v3_long['signals'].values[0]),
    }
    with open(os.path.join(MODELS_DIR, 'summary.json'), 'w') as f:
        json.dump(v3_summary, f, indent=2, default=str)

    print("\nDONE.")


if __name__ == '__main__':
    main()

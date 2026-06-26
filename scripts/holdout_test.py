import pandas as pd
import numpy as np
import joblib
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rule_filters import RuleFilters
from src.confidence_scorer import ConfidenceScorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score

MODELS_DIR = 'models'


def main():
    print("=" * 60)
    print("  PROPER HOLDOUT TEST")
    print("  Train: before June 2026 | Test: June 2026 only")
    print("=" * 60)

    print("\nLoading data...")
    features = pd.read_parquet('data/eurusd/features_m5.parquet')
    raw_features = pd.read_parquet('data/eurusd/features_m5_raw.parquet')
    filtered = pd.read_parquet('data/eurusd/m5_filtered.parquet')
    print(f"  Total bars: {len(features)}")

    timestamps = pd.to_datetime(filtered['time'])
    june_cutoff = pd.Timestamp('2026-06-01')

    train_mask = timestamps < june_cutoff
    test_mask = timestamps >= june_cutoff

    print(f"  Train bars: {train_mask.sum()} ({timestamps[train_mask].min().date()} to {timestamps[train_mask].max().date()})")
    print(f"  Test bars:  {test_mask.sum()} ({timestamps[test_mask].min().date()} to {timestamps[test_mask].max().date()})")

    feat_cols = [c for c in features.columns if c != 'time']

    train_valid = train_mask & (filtered['label'] != -1)
    test_valid = test_mask & (filtered['label'] != -1)

    X_train = features.loc[train_valid, feat_cols].values.astype(float)
    X_test = features.loc[test_valid, feat_cols].values.astype(float)
    y_train = filtered.loc[train_valid, 'label'].values.astype(int)
    y_test = filtered.loc[test_valid, 'label'].values.astype(int)

    print(f"  Train bars: {train_valid.sum()} (LONG={ (y_train==1).sum() }, SHORT={ (y_train==0).sum() })")
    print(f"  Test bars:  {test_valid.sum()} (LONG={ (y_test==1).sum() }, SHORT={ (y_test==0).sum() })")

    print("\nTraining models on pre-June 2026 data...")
    t0 = time.time()

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

    meta_features_train = np.column_stack([m.predict_proba(X_train)[:, 1] for m in base_models])
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_features_train, y_train)

    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s")

    print("\nPredicting on June 2026 holdout set...")
    t0 = time.time()

    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
    base_preds = np.column_stack([xgb_proba, lgbm_proba])
    meta_pred = np.clip(meta_model.predict(base_preds), 0, 1)
    ml_pred = (meta_pred > 0.5).astype(int)

    pred_time = time.time() - t0
    print(f"  Prediction done in {pred_time:.1f}s")
    print(f"  ML predicts LONG: {(ml_pred == 1).sum()}, SHORT: {(ml_pred == 0).sum()}")

    base_accuracy = accuracy_score(y_test, ml_pred)
    print(f"\n  Base accuracy (no filtering): {base_accuracy*100:.1f}%")

    print("\nApplying rule filters on test set...")
    rf = RuleFilters()
    filtered_test = rf.apply_all(
        filtered.loc[test_valid].reset_index(drop=True),
        raw_features.loc[test_valid].reset_index(drop=True)
    )
    passed_mask = filtered_test['filter_reject'] == 0
    print(f"  Passed filters: {passed_mask.sum()} / {len(filtered_test)}")

    print("\nComputing confidence scores on test set...")
    scorer = ConfidenceScorer({'threshold': 0})
    test_raw = raw_features.loc[test_valid].reset_index(drop=True)
    test_filtered = filtered.loc[test_valid].reset_index(drop=True)
    conf_scores = np.zeros(len(test_filtered))

    passed_indices = np.where(passed_mask.values)[0]
    for i in passed_indices:
        row = test_raw.iloc[i].to_dict()
        row['time'] = test_filtered.iloc[i]['time']
        weighted, _ = scorer.score(row, row)
        conf_scores[i] = weighted

    print(f"  Confidence range: {conf_scores[conf_scores > 0].min():.1f} - {conf_scores.max():.1f}")

    hours = pd.to_datetime(test_filtered['time']).dt.hour
    dow = pd.to_datetime(test_filtered['time']).dt.weekday
    session_mask = (hours >= 13) & (hours < 16) & (dow < 5)

    atr_pips = test_raw['atr_pips'].values if 'atr_pips' in test_raw.columns else np.full(len(test_filtered), 10.0)

    signal_dir = np.where(ml_pred == 1, 'LONG', 'SHORT')

    print("\n" + "=" * 60)
    print("  THRESHOLD COMPARISON ON HOLDOUT SET")
    print("=" * 60)

    results = []
    for threshold in [60, 65, 70, 75, 78, 80, 82, 85]:
        mask = passed_mask.values & session_mask.values & (atr_pips >= 3) & (conf_scores >= threshold)
        mask_long = mask & (ml_pred == 1)
        mask_short = mask & (ml_pred == 0)

        for label, mask_dir in [("LONG+SHORT", mask), ("LONG-only", mask_long), ("SHORT-only", mask_short)]:
            gen_count = mask_dir.sum()
            if gen_count > 0:
                labels = y_test[mask_dir]
                dirs = signal_dir[mask_dir]
                correct = ((dirs == 'LONG') & (labels == 1)).sum() + ((dirs == 'SHORT') & (labels == 0)).sum()
                accuracy = correct / gen_count * 100
                avg_tp = (atr_pips[mask_dir] * 4.5).mean()
                avg_sl = (atr_pips[mask_dir] * 1.5).mean()
            else:
                accuracy = 0
                avg_tp = 0
                avg_sl = 0

            results.append({
                'threshold': threshold,
                'direction': label,
                'signals': int(gen_count),
                'accuracy': round(accuracy, 1),
                'avg_tp': round(avg_tp, 1),
                'avg_sl': round(avg_sl, 1),
            })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv('data/eurusd/holdout_test_results.csv', index=False)
    print(f"\nSaved to data/eurusd/holdout_test_results.csv")

    print("\n" + "=" * 60)
    print("  COMPARISON: OLD (inflated) vs NEW (holdout)")
    print("=" * 60)

    old_results = pd.read_csv('data/eurusd/threshold_comparison.csv')
    print("\nOLD results (trained + tested on same data — INFLATED):")
    print(old_results[['threshold', 'signals', 'accuracy', 'signals_per_day']].to_string(index=False))

    new_long = results_df[(results_df['direction'] == 'LONG-only') & (results_df['threshold'].isin([78, 80]))]
    print("\nNEW results (holdout June 2026 — HONEST):")
    print(new_long.to_string(index=False))

    print("\n" + "=" * 60)
    print("  SAVING HOLDOUT MODELS")
    print("=" * 60)

    joblib.dump(xgb, os.path.join(MODELS_DIR, 'holdout_xgboost.joblib'))
    joblib.dump(lgbm, os.path.join(MODELS_DIR, 'holdout_lightgbm.joblib'))
    joblib.dump(meta_model, os.path.join(MODELS_DIR, 'holdout_meta.joblib'))

    summary = {
        'train_period': f'{timestamps[train_mask].min()} to {timestamps[train_mask].max()}',
        'test_period': f'{timestamps[test_mask].min()} to {timestamps[test_mask].max()}',
        'train_bars': int(train_mask.sum()),
        'test_bars': int(test_mask.sum()),
        'base_accuracy': round(base_accuracy * 100, 1),
        'train_time_seconds': round(train_time, 1),
    }
    import json
    with open(os.path.join(MODELS_DIR, 'holdout_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved: holdout_xgboost.joblib, holdout_lightgbm.joblib, holdout_meta.joblib")
    print(f"  Saved: holdout_summary.json")
    print("\nDONE.")


if __name__ == '__main__':
    main()

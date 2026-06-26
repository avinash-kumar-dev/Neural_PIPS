import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rule_filters import RuleFilters
from src.confidence_scorer import ConfidenceScorer


def main():
    print("Loading data...")
    features = pd.read_parquet('data/eurusd/features_m5.parquet')
    raw_features = pd.read_parquet('data/eurusd/features_m5_raw.parquet')
    filtered = pd.read_parquet('data/eurusd/m5_filtered.parquet')

    print(f"Total bars: {len(features)}")

    print("Applying rule filters...")
    rf = RuleFilters()
    filtered_all = rf.apply_all(filtered, raw_features)
    passed_mask = filtered_all['filter_reject'] == 0
    print(f"  Passed all filters: {passed_mask.sum()} / {len(filtered_all)}")
    print(f"  Rejection reasons: {filtered_all[passed_mask == False]['filter_reason'].value_counts().to_dict()}")

    print("Loading models...")
    import joblib
    xgb_model = joblib.load('models/ensemble_xgboost.joblib')
    lgbm_model = joblib.load('models/ensemble_lightgbm.joblib')
    meta_model = joblib.load('models/ensemble_meta.joblib')

    print("Batch ML prediction (all 95K bars)...")
    t0 = time.time()
    feat_cols = [c for c in features.columns if c != 'time']
    X = features[feat_cols].values.astype(float)

    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(X)[:, 1]

    base_preds = np.column_stack([xgb_proba, lgbm_proba])
    meta_pred = np.clip(meta_model.predict(base_preds), 0, 1)

    ml_pred = (meta_pred > 0.5).astype(int)
    ml_confidence = meta_pred
    print(f"ML predictions done in {time.time()-t0:.1f}s")
    print(f"  ML predicts LONG (1): {(ml_pred == 1).sum()} / SHORT (0): {(ml_pred == 0).sum()}")

    hours = pd.to_datetime(filtered['time']).dt.hour
    dow = pd.to_datetime(filtered['time']).dt.weekday
    session_mask = (hours >= 13) & (hours < 16) & (dow < 5)

    signal_dir = np.where(ml_pred == 1, 'LONG', 'SHORT')
    atr_pips_full = raw_features['atr_pips'].values if 'atr_pips' in raw_features.columns else np.full(len(features), 10.0)

    print("Batch confidence scoring (passed-filter bars)...")
    t0 = time.time()
    passed_idx = passed_mask.values
    passed_raw = raw_features.loc[passed_idx].reset_index(drop=True)
    passed_filtered = filtered.loc[passed_idx].reset_index(drop=True)
    scorer = ConfidenceScorer({'threshold': 0})
    conf_scores = np.zeros(len(features))
    for i, idx in enumerate(passed_raw.index):
        row = passed_raw.loc[idx].to_dict()
        row['time'] = passed_filtered.loc[idx, 'time']
        weighted, _ = scorer.score(row, row)
        conf_scores[np.where(passed_idx)[0][i]] = weighted
    print(f"Confidence scores done in {time.time()-t0:.1f}s")
    print(f"  Range: {conf_scores[conf_scores > 0].min():.1f} - {conf_scores.max():.1f}")
    print(f"  Mean: {conf_scores[conf_scores > 0].mean():.1f}")

    results = []
    for threshold in [60, 65, 70, 75, 80, 85]:
        t0 = time.time()
        mask = passed_mask.values & session_mask.values & (atr_pips_full >= 3) & (conf_scores >= threshold)
        mask_long = mask & (meta_pred > 0.5)
        mask_short = mask & (meta_pred <= 0.5)
        mask_both = mask_long | mask_short
        gen_count = mask_both.sum()

        if gen_count > 0:
            labels = filtered.loc[mask_both, 'label'].values
            dirs = signal_dir[mask_both]

            correct = ((dirs == 'LONG') & (labels == 1)).sum() + ((dirs == 'SHORT') & (labels == 0)).sum()
            accuracy = correct / gen_count * 100
            long_count = (dirs == 'LONG').sum()
            short_count = (dirs == 'SHORT').sum()
            avg_tp = (raw_features.loc[mask_both, 'atr_pips'].values * 4.5).mean() if 'atr_pips' in raw_features.columns else 0
            avg_sl = (raw_features.loc[mask_both, 'atr_pips'].values * 1.5).mean() if 'atr_pips' in raw_features.columns else 0
        else:
            correct = 0
            accuracy = 0
            long_count = 0
            short_count = 0
            avg_tp = 0
            avg_sl = 0

        total_days = len(features) / 1152
        signals_per_day = gen_count / total_days if total_days > 0 else 0

        results.append({
            'threshold': threshold,
            'total_bars': len(features),
            'passed_filters': int(passed_mask.sum()),
            'signals': int(gen_count),
            'long': int(long_count),
            'short': int(short_count),
            'signals_per_day': round(signals_per_day, 1),
            'correct': int(correct),
            'accuracy': round(accuracy, 1),
            'avg_tp_pips': round(avg_tp, 1),
            'avg_sl_pips': round(avg_sl, 1),
        })
        print(f"  T={threshold}: {gen_count} signals ({long_count}L/{short_count}S), {accuracy:.1f}% acc, {signals_per_day:.1f}/day ({time.time()-t0:.1f}s)")

    results_df = pd.DataFrame(results)
    results_df.to_csv('data/eurusd/threshold_comparison.csv', index=False)
    print(f"\nSaved to data/eurusd/threshold_comparison.csv")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()

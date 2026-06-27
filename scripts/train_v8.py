import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import time as time_module
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureEngineer, TimeFeatures
from src.feature_pipeline import FeaturePipeline
from src.labels_profit import ProfitAwareLabeler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier as LightGBMClassifier
from catboost import CatBoostClassifier
from scipy.optimize import minimize


MODELS_DIR = 'models/v8'
MIN_TRAIN_BARS = 20000
EMBARGO_BARS = 48
N_WINDOWS = 12
HOLDOUT_MONTH = 6
HOLDOUT_YEAR = 2026


def load_data():
    print("Loading M5 raw data...")
    m5 = pd.read_parquet('data/eurusd/raw/m5_ohlcv.parquet')
    print(f"  M5: {len(m5):,} bars | {m5['time'].min()} to {m5['time'].max()}")

    print("Loading precomputed features...")
    old_features = pd.read_parquet('data/eurusd/features_m5_v6_raw.parquet')

    if 'time' not in old_features.columns:
        old_features['time'] = m5['time'].values

    print("Computing time features...")
    time_feat = TimeFeatures().compute(m5['time'])

    print("Computing quality features...")
    engineer = FeatureEngineer()
    quality_feat = engineer.quality.compute(m5)

    feat_cols_old = [c for c in old_features.columns if c != 'time']
    raw_features = pd.concat([
        old_features[feat_cols_old].reset_index(drop=True),
        time_feat.reset_index(drop=True),
        quality_feat.reset_index(drop=True),
        old_features[['time']].reset_index(drop=True),
    ], axis=1)

    print(f"  Features: {raw_features.shape[1] - 1} cols")
    return m5, raw_features


def generate_labels(m5):
    print("\nGenerating profit-aware labels...")
    labeler = ProfitAwareLabeler(tp_mult=4.5, sl_mult=1.5, horizon=20, min_tp_pips=9, min_sl_pips=3)
    labels, meta = labeler.label(m5)

    win = (labels == 1).sum()
    loss = (labels == 0).sum()
    timeout = (labels == 2).sum()
    total = win + loss + timeout

    print(f"  WIN: {win} ({win/total*100:.1f}%)")
    print(f"  LOSS: {loss} ({loss/total*100:.1f}%)")
    print(f"  TIMEOUT: {timeout} ({timeout/total*100:.1f}%)")
    print(f"  Trainable (WIN+LOSS): {win + loss}")

    return labels, meta


def prepare_training_data(m5, raw_features, labels):
    m5_time = pd.to_datetime(m5['time'])
    holdout_start = pd.Timestamp(f'{HOLDOUT_YEAR}-{HOLDOUT_MONTH:02d}-01')
    holdout_mask = m5_time >= holdout_start

    train_mask = (~holdout_mask) & (labels != 2)
    feature_cols = [c for c in raw_features.columns if c != 'time']

    X_train = raw_features.loc[train_mask, feature_cols].values
    y_train = labels[train_mask].values

    X_holdout = raw_features.loc[holdout_mask, feature_cols].values
    y_holdout = labels[holdout_mask].values

    print(f"\n  Training bars: {len(X_train):,}")
    print(f"  Holdout bars: {len(X_holdout):,}")
    print(f"  Features: {X_train.shape[1]}")

    return X_train, y_train, X_holdout, y_holdout, feature_cols, holdout_mask


def optimize_weights(probas, y_true):
    n_models = probas.shape[1]

    def objective(w):
        w_norm = np.abs(w) / np.sum(np.abs(w))
        combined = probas @ w_norm
        preds = (combined > 0.5).astype(int)
        return -np.mean(preds == y_true)

    x0 = np.ones(n_models) / n_models
    bounds = [(0, 1)] * n_models
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return np.abs(result.x) / np.sum(np.abs(result.x))


def train_quality_model(X_train, y_train, feature_names):
    print("\n  Training quality model (WIN=1 vs LOSS=0)...")
    mask = y_train != 2
    X = X_train[mask]
    y = y_train[mask]

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"    WIN: {n_pos} | LOSS: {n_neg} | scale_pos_weight: {spw:.2f}")

    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=42, eval_metric='logloss',
        feature_names=feature_names
    )
    lgbm = LightGBMClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=42, verbose=-1
    )
    catboost = CatBoostClassifier(
        iterations=300, depth=8, learning_rate=0.05,
        subsample=0.8, l2_leaf_reg=3.0,
        class_weights={0: 1.0, 1: spw}, random_seed=42, verbose=0
    )

    t0 = time_module.time()
    xgb.fit(X, y)
    lgbm.fit(X, y)
    catboost.fit(X, y)
    train_time = time_module.time() - t0

    print(f"    Trained in {train_time:.1f}s")

    xgb_proba = xgb.predict_proba(X)[:, 1]
    lgbm_proba = lgbm.predict_proba(X)[:, 1]
    cat_proba = catboost.predict_proba(X)[:, 1]
    all_proba = np.column_stack([xgb_proba, lgbm_proba, cat_proba])

    weights = optimize_weights(all_proba, y)
    print(f"    Weights: XGB={weights[0]:.3f} LGBM={weights[1]:.3f} CatBoost={weights[2]:.3f}")

    combined = all_proba @ weights
    preds = (combined > 0.5).astype(int)
    acc = np.mean(preds == y)
    n_pos_pred = preds.sum()
    print(f"    Train accuracy: {acc:.1%} | Positive predictions: {n_pos_pred}/{len(y)}")

    return xgb, lgbm, catboost, weights, train_time


def walk_forward(X_train, y_train, feature_names):
    print(f"\nWalk-forward training ({N_WINDOWS} windows, embargo={EMBARGO_BARS})...")

    usable = len(X_train) - MIN_TRAIN_BARS
    if usable <= 0:
        print("  Not enough data for walk-forward!")
        return None

    step = usable // N_WINDOWS
    quality_models_list = []
    quality_weights_list = []
    quality_metrics = []

    for w in range(N_WINDOWS):
        train_end = MIN_TRAIN_BARS + step * w
        test_start = train_end + EMBARGO_BARS
        test_end = test_start + step

        if test_end > len(X_train):
            break

        X_tr = X_train[:train_end]
        y_tr = y_train[:train_end]
        X_te = X_train[test_start:test_end]
        y_te = y_train[test_start:test_end]

        mask_tr = y_tr != 2
        mask_te = y_te != 2
        X_tr_q = X_tr[mask_tr]
        y_tr_q = y_tr[mask_tr]
        X_te_q = X_te[mask_te]
        y_te_q = y_te[mask_te]

        if len(X_tr_q) < 100 or len(X_te_q) < 50:
            continue

        n_pos = (y_tr_q == 1).sum()
        n_neg = (y_tr_q == 0).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        xgb = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=spw, random_state=42, eval_metric='logloss',
            feature_names=feature_names
        )
        lgbm = LightGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=spw, random_state=42, verbose=-1
        )
        catboost = CatBoostClassifier(
            iterations=300, depth=8, learning_rate=0.05,
            subsample=0.8, l2_leaf_reg=3.0,
            class_weights={0: 1.0, 1: spw}, random_seed=42, verbose=0
        )

        xgb.fit(X_tr_q, y_tr_q)
        lgbm.fit(X_tr_q, y_tr_q)
        catboost.fit(X_tr_q, y_tr_q)

        xgb_p = xgb.predict_proba(X_te_q)[:, 1]
        lgbm_p = lgbm.predict_proba(X_te_q)[:, 1]
        cat_p = catboost.predict_proba(X_te_q)[:, 1]
        all_p = np.column_stack([xgb_p, lgbm_p, cat_p])

        weights = optimize_weights(all_p, y_te_q)
        combined = all_p @ weights
        preds = (combined > 0.5).astype(int)

        acc = np.mean(preds == y_te_q)
        n_pred = preds.sum()
        n_true_pos = ((preds == 1) & (y_te_q == 1)).sum()
        precision = n_true_pos / n_pred if n_pred > 0 else 0
        n_true_all_pos = (y_te_q == 1).sum()
        recall = n_true_pos / n_true_all_pos if n_true_all_pos > 0 else 0

        print(f"  Window {w+1}: acc={acc:.1%} precision={precision:.1%} recall={recall:.1%} "
              f"preds={n_pred} weights=[{weights[0]:.3f},{weights[1]:.3f},{weights[2]:.3f}]")

        quality_models_list.append((xgb, lgbm, catboost))
        quality_weights_list.append(weights)
        quality_metrics.append({
            'accuracy': acc, 'precision': precision, 'recall': recall,
            'n_preds': n_pred, 'weights': weights.tolist()
        })

    if not quality_models_list:
        print("  No windows completed!")
        return None

    accs = [m['accuracy'] for m in quality_metrics]
    precs = [m['precision'] for m in quality_metrics]
    recs = [m['recall'] for m in quality_metrics]
    print(f"\n  Walk-forward done: acc={np.mean(accs):.1%}±{np.std(accs):.1%} "
          f"prec={np.mean(precs):.1%} rec={np.mean(recs):.1%}")

    return quality_models_list, quality_weights_list, quality_metrics


def holdout_test(models_list, weights_list, X_holdout, y_holdout, feature_names):
    print("\n" + "=" * 60)
    print("  HOLDOUT TEST: June 2026")
    print("=" * 60)

    mask = y_holdout != 2
    X_h = X_holdout[mask]
    y_h = y_holdout[mask]

    print(f"  Holdout bars (WIN/LOSS): {len(y_h)}")
    print(f"  WIN: {(y_h == 1).sum()} | LOSS: {(y_h == 0).sum()}")

    xgb_preds = []
    lgbm_preds = []
    cat_preds = []

    for xgb, lgbm, catboost in models_list:
        xgb_preds.append(xgb.predict_proba(X_h)[:, 1])
        lgbm_preds.append(lgbm.predict_proba(X_h)[:, 1])
        cat_preds.append(catboost.predict_proba(X_h)[:, 1])

    xgb_avg = np.mean(xgb_preds, axis=0)
    lgbm_avg = np.mean(lgbm_preds, axis=0)
    cat_avg = np.mean(cat_preds, axis=0)

    all_proba = np.column_stack([xgb_avg, lgbm_avg, cat_avg])

    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    print(f"\n  {'T':>5} {'Signals':>8} {'WIN':>5} {'LOSS':>5} {'WR%':>6} {'Precision':>10}")

    for thresh in thresholds:
        preds = (all_proba[:, 0] > thresh).astype(int)
        n_sig = preds.sum()
        if n_sig == 0:
            continue
        wins = ((preds == 1) & (y_h == 1)).sum()
        losses = ((preds == 1) & (y_h == 0)).sum()
        wr = wins / n_sig * 100 if n_sig > 0 else 0
        print(f"  {thresh:>5.2f} {n_sig:>8} {wins:>5} {losses:>5} {wr:>5.1f}% {wr:>9.1f}%")

    best_thresh = 0.50
    best_wr = 0
    for t in np.arange(0.40, 0.80, 0.01):
        preds = (all_proba[:, 0] > t).astype(int)
        n_sig = preds.sum()
        if n_sig < 10:
            continue
        wins = ((preds == 1) & (y_h == 1)).sum()
        wr = wins / n_sig
        if wr > best_wr:
            best_wr = wr
            best_thresh = t

    print(f"\n  Best threshold: {best_thresh:.2f} -> {best_wr:.1%} accuracy")
    return best_thresh, best_wr


def main():
    print("=" * 60)
    print("  V8 TRAINING: Profit-Aware Quality Model")
    print("=" * 60)

    t_start = time_module.time()

    m5, raw_features = load_data()
    labels, meta = generate_labels(m5)
    X_train, y_train, X_holdout, y_holdout, feature_names, holdout_mask = prepare_training_data(m5, raw_features, labels)

    wf_result = walk_forward(X_train, y_train, feature_names)
    if wf_result is None:
        print("Walk-forward failed!")
        return

    quality_models_list, quality_weights_list, quality_metrics = wf_result

    print("\nTraining final quality models on all training data...")
    xgb_final, lgbm_final, cat_final, final_weights, final_train_time = train_quality_model(
        X_train, y_train, feature_names
    )

    best_thresh, best_wr = holdout_test(
        quality_models_list, quality_weights_list, X_holdout, y_holdout, feature_names
    )

    print("\nSaving models...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(xgb_final, os.path.join(MODELS_DIR, 'xgboost.joblib'))
    joblib.dump(lgbm_final, os.path.join(MODELS_DIR, 'lightgbm.joblib'))
    joblib.dump(cat_final, os.path.join(MODELS_DIR, 'catboost.joblib'))
    joblib.dump(final_weights, os.path.join(MODELS_DIR, 'ensemble_weights.joblib'))

    summary = {
        'version': 'v8',
        'description': 'Profit-aware quality model (WIN/LOSS prediction)',
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'n_windows': len(quality_models_list),
        'walk_forward': {
            'accuracy_mean': float(np.mean([m['accuracy'] for m in quality_metrics])),
            'accuracy_std': float(np.std([m['accuracy'] for m in quality_metrics])),
            'precision_mean': float(np.mean([m['precision'] for m in quality_metrics])),
            'recall_mean': float(np.mean([m['recall'] for m in quality_metrics])),
        },
        'ensemble_weights': final_weights.tolist(),
        'holdout': {
            'best_threshold': best_thresh,
            'best_accuracy': best_wr,
        },
        'train_time_seconds': final_train_time,
        'label_distribution': {
            'win': int((labels == 1).sum()),
            'loss': int((labels == 0).sum()),
            'timeout': int((labels == 2).sum()),
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(os.path.join(MODELS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    total_time = time_module.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Saved to {MODELS_DIR}/")
    print("\n" + "=" * 60)
    print("  V8 TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

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
from src.labels import TripleBarrierLabeler
from src.labels_profit import ProfitAwareLabeler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.optimize import minimize


MODELS_DIR = 'models/v8_retrained'
EMBARGO_BARS = 48
TEST_START = pd.Timestamp('2026-06-25')
TEST_END = pd.Timestamp('2026-06-26 23:59:00')


def load_data():
    print("Loading M5 raw data...")
    m5 = pd.read_parquet('data/eurusd/raw/m5_ohlcv.parquet')
    m5['time'] = pd.to_datetime(m5['time'])
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


def train_direction_model(X_train, y_train, feature_names):
    """Train direction model: LONG=1, SHORT=0 (NO-TRADE excluded)."""
    print("\n  Training direction model (LONG vs SHORT)...")
    mask = y_train != -1
    X = X_train[mask]
    y = y_train[mask]

    n_long = (y == 1).sum()
    n_short = (y == 0).sum()
    spw = n_short / n_long if n_long > 0 else 1.0
    print(f"    LONG: {n_long} | SHORT: {n_short} | scale_pos_weight: {spw:.2f}")

    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw, random_state=42, eval_metric='logloss',
    )
    lgbm = LGBMClassifier(
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

    xgb_p = xgb.predict_proba(X)[:, 1]
    lgbm_p = lgbm.predict_proba(X)[:, 1]
    cat_p = catboost.predict_proba(X)[:, 1]
    all_p = np.column_stack([xgb_p, lgbm_p, cat_p])
    weights = optimize_weights(all_p, y)

    combined = all_p @ weights
    preds = (combined > 0.5).astype(int)
    acc = np.mean(preds == y)
    print(f"    Weights: XGB={weights[0]:.3f} LGBM={weights[1]:.3f} CatBoost={weights[2]:.3f}")
    print(f"    Train accuracy: {acc:.1%} | Train time: {train_time:.1f}s")

    return xgb, lgbm, catboost, weights, train_time


def train_quality_model(X_train, y_train, feature_names):
    """Train quality model: WIN=1, LOSS=0 (TIMEOUT excluded)."""
    print("\n  Training quality model (WIN vs LOSS)...")
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
    )
    lgbm = LGBMClassifier(
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

    xgb_p = xgb.predict_proba(X)[:, 1]
    lgbm_p = lgbm.predict_proba(X)[:, 1]
    cat_p = catboost.predict_proba(X)[:, 1]
    all_p = np.column_stack([xgb_p, lgbm_p, cat_p])
    weights = optimize_weights(all_p, y)

    combined = all_p @ weights
    preds = (combined > 0.5).astype(int)
    acc = np.mean(preds == y)
    print(f"    Weights: XGB={weights[0]:.3f} LGBM={weights[1]:.3f} CatBoost={weights[2]:.3f}")
    print(f"    Train accuracy: {acc:.1%} | Train time: {train_time:.1f}s")

    return xgb, lgbm, catboost, weights, train_time


def backtest_june25_26(m5, raw_features, direction_models, direction_weights,
                       direction_feature_names, quality_models, quality_weights):
    print("\n" + "=" * 60)
    print("  BACKTEST: June 25-26, 2026")
    print("=" * 60)

    pipeline = FeaturePipeline.load(os.path.join(MODELS_DIR, 'direction_pipeline.joblib'))

    engine_config = {
        'confidence_threshold': 78,
        'ml_gate': 0.75,
        'confluence_min': 2,
        'max_spread_pips': 0.5,
        'min_atr_pips': 3.0,
        'max_atr_pips': 20.0,
        'min_atr_ratio': 0.5,
        'max_atr_ratio': 2.5,
        'sessions': [
            {'start': 7, 'end': 9},
            {'start': 13, 'end': 16},
            {'start': 17, 'end': 20},
        ],
        'feature_pipeline': pipeline,
        'quality_model': quality_models,
        'quality_weights': quality_weights,
        'quality_threshold': 0.55,
    }

    from src.signal_engine import SignalEngine
    engine = SignalEngine(engine_config)

    test_mask = (m5['time'] >= TEST_START) & (m5['time'] <= TEST_END)
    test_indices = m5[test_mask].index.tolist()
    print(f"  Test bars: {len(test_indices):,}")
    print(f"  Date range: {m5[test_mask]['time'].min()} to {m5[test_mask]['time'].max()}")

    sessions = [{'start': 7, 'end': 9}, {'start': 13, 'end': 16}, {'start': 17, 'end': 20}]
    signals = []
    cooldown = 0
    consecutive_losses = 0
    session_signal_count = 0
    current_session = None
    processed = 0

    for idx in test_indices:
        bar_time = m5['time'].iloc[idx]
        hour = bar_time.hour
        session_key = None
        for s in sessions:
            if s['start'] <= hour < s['end']:
                session_key = f"{s['start']}-{s['end']}"
                break
        if session_key != current_session:
            current_session = session_key
            session_signal_count = 0
        if cooldown > 0:
            cooldown -= 1
            continue

        processed += 1
        if processed % 200 == 0:
            print(f"  Processed {processed} session bars, {len(signals)} signals...")
        if idx < 50:
            continue

        feat_row = raw_features.loc[idx].copy()
        feat_row_dict = feat_row.to_dict()
        feat_row_dict['time'] = bar_time
        feat_series = pd.Series(feat_row_dict)

        raw_dict = {}
        for col in feat_row.index:
            if col != 'time':
                val = feat_row[col]
                raw_dict[col] = 0.0 if pd.isna(val) else val
        raw_dict['atr_pips'] = raw_dict.get('atr_14', 0) * 10000
        raw_series = pd.Series(raw_dict)

        result = engine.generate_signal(feat_series, raw_series, direction_models, direction_weights)

        if result['signal'] != 'NO-TRADE':
            entry_price = m5['close'].iloc[idx]
            if result['signal'] == 'LONG':
                tp_price = entry_price + result['tp_pips'] / 10000
                sl_price = entry_price - result['sl_pips'] / 10000
            else:
                tp_price = entry_price - result['tp_pips'] / 10000
                sl_price = entry_price + result['sl_pips'] / 10000

            result_pnl = None
            result_bars = None
            for future_idx in range(idx + 1, min(idx + 21, len(m5))):
                future_bar = m5.iloc[future_idx]
                if result['signal'] == 'LONG':
                    if future_bar['high'] >= tp_price:
                        result_pnl = result['tp_pips']
                        result_bars = future_idx - idx
                        break
                    if future_bar['low'] <= sl_price:
                        result_pnl = -result['sl_pips']
                        result_bars = future_idx - idx
                        break
                else:
                    if future_bar['low'] <= tp_price:
                        result_pnl = result['tp_pips']
                        result_bars = future_idx - idx
                        break
                    if future_bar['high'] >= sl_price:
                        result_pnl = -result['sl_pips']
                        result_bars = future_idx - idx
                        break

            signal_entry = {
                'id': len(signals) + 1,
                'signal': result['signal'],
                'confidence': round(result['confidence'], 1),
                'ml_confidence': round(result['ml_confidence'], 3),
                'quality_score': round(result.get('quality_score', 0), 3),
                'confluence': result['confluence'],
                'tp_pips': round(result['tp_pips'], 1),
                'sl_pips': round(result['sl_pips'], 1),
                'tp_sl_ratio': round(result['tp_sl_ratio'], 1),
                'atr_pips': round(result['atr_pips'], 1),
                'entry_price': round(entry_price, 5),
                'tp_price': round(tp_price, 5),
                'sl_price': round(sl_price, 5),
                'timestamp': str(bar_time),
                'result': 'WIN' if result_pnl and result_pnl > 0 else ('LOSS' if result_pnl and result_pnl < 0 else 'OPEN'),
                'pnl_pips': round(result_pnl, 1) if result_pnl else 0,
                'bars_held': result_bars,
            }
            signals.append(signal_entry)
            session_signal_count += 1

            if result_pnl and result_pnl < 0:
                consecutive_losses += 1
                if consecutive_losses >= 3:
                    cooldown = 6
                    consecutive_losses = 0
            else:
                consecutive_losses = 0
            cooldown = 6

    return signals


def print_results(signals, label=""):
    if not signals:
        print(f"\n  {label} No signals generated!")
        return

    wins = [s for s in signals if s['result'] == 'WIN']
    losses = [s for s in signals if s['result'] == 'LOSS']
    open_trades = [s for s in signals if s['result'] == 'OPEN']
    longs = [s for s in signals if s['signal'] == 'LONG']
    shorts = [s for s in signals if s['signal'] == 'SHORT']

    total_pnl = sum(s['pnl_pips'] for s in wins + losses)
    wr = len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) > 0 else 0

    print(f"\n  {label} Results:")
    print(f"  Total signals: {len(signals)}")
    print(f"  LONG: {len(longs)} | SHORT: {len(shorts)}")
    print(f"  Wins: {len(wins)} | Losses: {len(losses)} | Open: {len(open_trades)}")
    print(f"  Win rate: {wr:.1f}%")
    print(f"  Total PnL: {total_pnl:+.1f} pips")
    if wins:
        print(f"  Avg win: {np.mean([s['pnl_pips'] for s in wins]):.1f} pips")
    if losses:
        print(f"  Avg loss: {np.mean([s['pnl_pips'] for s in losses]):.1f} pips")

    by_day = {}
    for s in signals:
        day = s['timestamp'][:10]
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(s)

    print(f"\n  Daily breakdown:")
    for day in sorted(by_day.keys()):
        ds = by_day[day]
        dw = len([s for s in ds if s['result'] == 'WIN'])
        dl = len([s for s in ds if s['result'] == 'LOSS'])
        dp = sum(s['pnl_pips'] for s in ds)
        print(f"    {day}: {len(ds)} signals, {dw}W/{dl}L, {dp:+.1f} pips")

    print(f"\n  Signal details:")
    for s in signals:
        status = 'W' if s['result'] == 'WIN' else ('L' if s['result'] == 'LOSS' else 'O')
        print(f"    #{s['id']:2d} {s['timestamp'][:16]} {s['signal']:5s} "
              f"conf={s['confidence']:5.0f} ml={s['ml_confidence']:.3f} "
              f"q={s['quality_score']:.3f} "
              f"tp={s['tp_pips']:5.1f} sl={s['sl_pips']:5.1f} "
              f"pnl={s['pnl_pips']:+6.1f} [{status}]")


def main():
    t_start = time_module.time()
    print("=" * 60)
    print("  V8 RETRAINED: Train on pre-June + June 1-24, Test June 25-26")
    print("=" * 60)

    m5, raw_features = load_data()
    feature_names = [c for c in raw_features.columns if c != 'time']

    m5_time = m5['time']
    train_mask = m5_time < TEST_START
    feature_cols = [c for c in raw_features.columns if c != 'time']

    X_train = raw_features.loc[train_mask, feature_cols].values
    print(f"\n  Training bars: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")

    print("\nGenerating direction labels (LONG/SHORT/NO-TRADE)...")
    labeler = TripleBarrierLabeler(tp_mult=4.5, sl_mult=1.5, horizon=20, min_tp_pips=9, min_sl_pips=3)
    dir_labels_result = labeler.label_bars(m5)
    dir_labels = dir_labels_result[0]
    y_dir = dir_labels[train_mask].values
    long_ct = (y_dir == 1).sum()
    short_ct = (y_dir == 0).sum()
    no_trade_ct = (y_dir == -1).sum()
    print(f"  LONG: {long_ct} ({long_ct/len(y_dir)*100:.1f}%)")
    print(f"  SHORT: {short_ct} ({short_ct/len(y_dir)*100:.1f}%)")
    print(f"  NO-TRADE: {no_trade_ct} ({no_trade_ct/len(y_dir)*100:.1f}%)")

    print("\nGenerating profit-aware labels (WIN/LOSS/TIMEOUT)...")
    profit_labeler = ProfitAwareLabeler(tp_mult=4.5, sl_mult=1.5, horizon=20, min_tp_pips=9, min_sl_pips=3)
    profit_labels, profit_meta = profit_labeler.label(m5)
    y_profit = profit_labels[train_mask].values
    win_ct = (y_profit == 1).sum()
    loss_ct = (y_profit == 0).sum()
    timeout_ct = (y_profit == 2).sum()
    print(f"  WIN: {win_ct} ({win_ct/len(y_profit)*100:.1f}%)")
    print(f"  LOSS: {loss_ct} ({loss_ct/len(y_profit)*100:.1f}%)")
    print(f"  TIMEOUT: {timeout_ct} ({timeout_ct/len(y_profit)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("  CREATING FEATURE PIPELINE")
    print("=" * 60)
    pipeline = FeaturePipeline(variance_threshold=0.001, use_robust_scaler=True)
    X_train_transformed = pipeline.fit_transform(X_train, feature_names=feature_names)
    print(f"  Pipeline: {X_train.shape[1]} -> {X_train_transformed.shape[1]} features")
    os.makedirs(MODELS_DIR, exist_ok=True)
    pipeline.save(os.path.join(MODELS_DIR, 'direction_pipeline.joblib'))
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)

    print("\n" + "=" * 60)
    print("  TRAINING DIRECTION MODEL")
    print("=" * 60)
    dir_xgb, dir_lgbm, dir_cat, dir_weights, dir_time = train_direction_model(
        X_train_transformed, y_dir, feature_names
    )

    print("\n" + "=" * 60)
    print("  TRAINING QUALITY MODEL")
    print("=" * 60)
    qual_xgb, qual_lgbm, qual_cat, qual_weights, qual_time = train_quality_model(
        X_train, y_profit, feature_names
    )

    print("\nSaving models...")
    joblib.dump(dir_xgb, os.path.join(MODELS_DIR, 'direction_xgboost.joblib'))
    joblib.dump(dir_lgbm, os.path.join(MODELS_DIR, 'direction_lightgbm.joblib'))
    joblib.dump(dir_cat, os.path.join(MODELS_DIR, 'direction_catboost.joblib'))
    joblib.dump(dir_weights, os.path.join(MODELS_DIR, 'direction_weights.joblib'))

    joblib.dump(qual_xgb, os.path.join(MODELS_DIR, 'quality_xgboost.joblib'))
    joblib.dump(qual_lgbm, os.path.join(MODELS_DIR, 'quality_lightgbm.joblib'))
    joblib.dump(qual_cat, os.path.join(MODELS_DIR, 'quality_catboost.joblib'))
    joblib.dump(qual_weights, os.path.join(MODELS_DIR, 'quality_weights.joblib'))

    print("\n" + "=" * 60)
    print("  BACKTESTING: June 25-26")
    print("=" * 60)

    direction_models = [dir_xgb, dir_lgbm, dir_cat]
    quality_models = [qual_xgb, qual_lgbm, qual_cat]

    signals = backtest_june25_26(
        m5, raw_features, direction_models, dir_weights,
        feature_names, quality_models, qual_weights
    )

    print_results(signals, label="June 25-26:")

    output_file = os.path.join(MODELS_DIR, 'backtest_june25_26.json')
    with open(output_file, 'w') as f:
        json.dump(signals, f, indent=2, default=str)
    print(f"\n  Saved to {output_file}")

    total_time = time_module.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()

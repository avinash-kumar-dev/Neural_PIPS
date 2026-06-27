import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureEngineer, TimeFeatures
from src.feature_pipeline import FeaturePipeline
from src.signal_engine import SignalEngine


MODELS_DIR = 'models/v7'


def load_models():
    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib'))
    lgbm = joblib.load(os.path.join(MODELS_DIR, 'lightgbm.joblib'))
    pipeline = FeaturePipeline.load(os.path.join(MODELS_DIR, 'feature_pipeline.joblib'))

    catboost_path = os.path.join(MODELS_DIR, 'catboost.joblib')
    models = [xgb, lgbm]
    if os.path.exists(catboost_path):
        catboost = joblib.load(catboost_path)
        models.append(catboost)

    weights_path = os.path.join(MODELS_DIR, 'ensemble_weights.joblib')
    if os.path.exists(weights_path):
        weights = joblib.load(weights_path)
    else:
        weights = np.array([1.0 / len(models)] * len(models))

    return models, weights, pipeline


def backtest_june(models, weights, pipeline):
    print("=" * 60)
    print("  V7 BACKTEST: June 2026 (precomputed features)")
    print("=" * 60)

    engine = SignalEngine({
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
    })

    print("\nLoading M5 raw data...")
    m5 = pd.read_parquet('data/eurusd/raw/m5_ohlcv.parquet')
    m5['time'] = pd.to_datetime(m5['time'])

    print("Loading precomputed features...")
    old_features = pd.read_parquet('data/eurusd/features_m5_v6_raw.parquet')

    if 'time' not in old_features.columns:
        old_features['time'] = m5['time'].values

    print("Computing time features...")
    time_feat = TimeFeatures().compute(m5['time'])

    feat_cols_old = [c for c in old_features.columns if c != 'time']
    raw_features = pd.concat([
        old_features[feat_cols_old].reset_index(drop=True),
        time_feat.reset_index(drop=True),
        old_features[['time']].reset_index(drop=True),
    ], axis=1)

    print(f"  M5: {len(m5):,} bars | Features: {raw_features.shape[1] - 1} cols")

    june_start = pd.Timestamp('2026-06-01')
    june_end = pd.Timestamp('2026-06-30')
    june_mask = (m5['time'] >= june_start) & (m5['time'] <= june_end)
    june_indices = m5[june_mask].index.tolist()

    print(f"  June bars: {len(june_indices):,}")

    sessions = [{'start': 7, 'end': 9}, {'start': 13, 'end': 16}, {'start': 17, 'end': 20}]

    signals = []
    cooldown = 0
    consecutive_losses = 0
    max_consecutive_losses = 3
    session_signal_count = 0
    current_session = None

    processed = 0
    for idx in june_indices:
        bar_time = m5['time'].iloc[idx]

        session_key = None
        hour = bar_time.hour
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
        if processed % 500 == 0:
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

        atr_14 = raw_dict.get('atr_14', 0)
        if pd.isna(atr_14):
            atr_14 = 0
        raw_dict['atr_pips'] = atr_14 * 10000

        raw_series = pd.Series(raw_dict)

        result = engine.generate_signal(feat_series, raw_series, models, weights)

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
                if consecutive_losses >= max_consecutive_losses:
                    cooldown = 6
                    consecutive_losses = 0
            else:
                consecutive_losses = 0

            cooldown = 6

    return signals


def print_results(signals):
    if not signals:
        print("\n  No signals generated!")
        return

    wins = [s for s in signals if s['result'] == 'WIN']
    losses = [s for s in signals if s['result'] == 'LOSS']
    open_trades = [s for s in signals if s['result'] == 'OPEN']
    longs = [s for s in signals if s['signal'] == 'LONG']
    shorts = [s for s in signals if s['signal'] == 'SHORT']

    total_pnl = sum(s['pnl_pips'] for s in wins + losses)
    win_pnl = sum(s['pnl_pips'] for s in wins)
    loss_pnl = sum(s['pnl_pips'] for s in losses)

    wr = len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) > 0 else 0

    print(f"\n  Results:")
    print(f"  Total signals: {len(signals)}")
    print(f"  LONG: {len(longs)} | SHORT: {len(shorts)}")
    print(f"  Wins: {len(wins)} | Losses: {len(losses)} | Open: {len(open_trades)}")
    print(f"  Win rate: {wr:.1f}%")
    print(f"  Total PnL: {total_pnl:+.1f} pips")
    print(f"  Win PnL: {win_pnl:+.1f} | Loss PnL: {loss_pnl:+.1f}")

    if wins:
        avg_win = np.mean([s['pnl_pips'] for s in wins])
        print(f"  Avg win: {avg_win:.1f} pips")
    if losses:
        avg_loss = np.mean([s['pnl_pips'] for s in losses])
        print(f"  Avg loss: {avg_loss:.1f} pips")

    by_day = {}
    for s in signals:
        day = s['timestamp'][:10]
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(s)

    print(f"\n  Daily breakdown:")
    for day in sorted(by_day.keys()):
        day_signals = by_day[day]
        day_wins = len([s for s in day_signals if s['result'] == 'WIN'])
        day_losses = len([s for s in day_signals if s['result'] == 'LOSS'])
        day_pnl = sum(s['pnl_pips'] for s in day_signals)
        print(f"    {day}: {len(day_signals)} signals, {day_wins}W/{day_losses}L, {day_pnl:+.1f} pips")

    print(f"\n  Signal details:")
    for s in signals:
        status = 'W' if s['result'] == 'WIN' else ('L' if s['result'] == 'LOSS' else 'O')
        print(f"    #{s['id']:2d} {s['timestamp'][:16]} {s['signal']:5s} "
              f"conf={s['confidence']:5.0f} ml={s['ml_confidence']:.3f} "
              f"conf={s['confluence']} tp={s['tp_pips']:5.1f} sl={s['sl_pips']:5.1f} "
              f"pnl={s['pnl_pips']:+6.1f} [{status}]")


def main():
    print("Loading models...")
    models, weights, pipeline = load_models()
    print(f"  Models: {len(models)} | Weights: [{', '.join(f'{w:.3f}' for w in weights)}]")
    print(f"  Pipeline features: {len(pipeline.feature_names_out_)}")

    signals = backtest_june(models, weights, pipeline)

    print_results(signals)

    output_file = 'data/eurusd/v7_june_backtest.json'
    with open(output_file, 'w') as f:
        json.dump(signals, f, indent=2, default=str)
    print(f"\n  Saved to {output_file}")


if __name__ == '__main__':
    main()

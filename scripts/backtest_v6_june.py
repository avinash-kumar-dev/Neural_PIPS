import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureEngineer
from src.labels import TripleBarrierLabeler
from src.signal_engine import SignalEngine
from src.confidence_scorer import ConfidenceScorer

MODELS_DIR = 'models/v6'


def load_models():
    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib'))
    lgbm = joblib.load(os.path.join(MODELS_DIR, 'lightgbm.joblib'))
    cb = joblib.load(os.path.join(MODELS_DIR, 'catboost.joblib'))
    weights = joblib.load(os.path.join(MODELS_DIR, 'ensemble_weights.joblib'))
    return [xgb, lgbm, cb], weights


def backtest_june():
    print("=" * 70)
    print("  BACKTEST: V6 Model on June 2026")
    print("  Phase 2 features (81) + weighted average + CatBoost")
    print("=" * 70)

    models, weights = load_models()
    print(f"  Models loaded: XGB + LGBM + CatBoost (weights: {[f'{w:.3f}' for w in weights]})")

    feature_pipeline = joblib.load(os.path.join(MODELS_DIR, 'feature_pipeline.joblib'))

    m5 = pd.read_parquet('data/eurusd/raw/m5_ohlcv.parquet')
    m15 = pd.read_parquet('data/eurusd/raw/m15_ohlcv.parquet')
    h1 = pd.read_parquet('data/eurusd/raw/h1_ohlcv.parquet')
    h4 = pd.read_parquet('data/eurusd/raw/h4_ohlcv.parquet')

    fe = FeatureEngineer()
    raw_features = fe.compute_all(m5, m15, h1, h4)
    raw_features['time'] = m5['time'].values

    engine = SignalEngine({
        'confidence_threshold': 78,
        'session_start': 13,
        'session_end': 16,
        'feature_pipeline': feature_pipeline,
    })
    scorer = ConfidenceScorer({'threshold': 78})

    june_start = pd.Timestamp('2026-06-01')
    june_end = pd.Timestamp('2026-06-30')
    session_mask = (m5['time'] >= june_start) & (m5['time'] < june_end)
    session_bars = m5[session_mask]
    session_indices = session_bars.index.tolist()

    print(f"\n  Period: June 2026")
    print(f"  Session bars (13:00-16:00 UTC): {len(session_indices)}")

    signals = []
    check_num = 0

    for idx in session_indices:
        check_num += 1
        feat_row = raw_features.loc[idx].to_dict()
        feat_row['time'] = m5.loc[idx, 'time']
        feat_series = pd.Series(feat_row)
        raw_series = pd.Series(feat_row)

        engine_result = engine.generate_signal(feat_series, raw_series, models, weights)

        ts = m5.loc[idx, 'time']
        h4_regime = feat_row.get('h4_regime', 0)
        h4_str = 'BULL' if h4_regime > 0 else ('BEAR' if h4_regime < 0 else 'NEUT')

        if engine_result['signal'] != 'NO-TRADE':
            entry_price = m5.loc[idx, 'close']
            tp_pips = engine_result['tp_pips']
            sl_pips = engine_result['sl_pips']

            if engine_result['signal'] == 'LONG':
                tp_price = entry_price + tp_pips / 10000
                sl_price = entry_price - sl_pips / 10000
            else:
                tp_price = entry_price - tp_pips / 10000
                sl_price = entry_price + sl_pips / 10000

            future_bars = m5[m5.index > idx].head(20)
            outcome = 'OPEN'
            pnl = 0
            for _, bar in future_bars.iterrows():
                if engine_result['signal'] == 'LONG':
                    if bar['high'] >= tp_price:
                        outcome = 'WIN'
                        pnl = tp_pips
                        break
                    if bar['low'] <= sl_price:
                        outcome = 'LOSS'
                        pnl = -sl_pips
                        break
                else:
                    if bar['low'] <= tp_price:
                        outcome = 'WIN'
                        pnl = tp_pips
                        break
                    if bar['high'] >= sl_price:
                        outcome = 'LOSS'
                        pnl = -sl_pips
                        break

            signals.append({
                'date': ts.strftime('%Y-%m-%d'),
                'time': ts.strftime('%H:%M'),
                'signal': engine_result['signal'],
                'confidence': engine_result['confidence'],
                'h4': h4_str,
                'entry': entry_price,
                'tp': tp_price,
                'sl': sl_price,
                'tp_pips': tp_pips,
                'sl_pips': sl_pips,
                'outcome': outcome,
                'pnl': pnl,
            })

    print(f"\n  Total signals: {len(signals)}")
    if not signals:
        print("  No signals generated.")
        return

    df = pd.DataFrame(signals)
    closed = df[df['outcome'].isin(['WIN', 'LOSS'])]
    open_trades = df[df['outcome'] == 'OPEN']
    wins = df[df['outcome'] == 'WIN']
    losses = df[df['outcome'] == 'LOSS']

    print(f"\n  Wins: {len(wins)} | Losses: {len(losses)} | Open: {len(open_trades)}")
    print(f"  Win rate: {len(wins) / len(closed) * 100:.1f}%" if len(closed) > 0 else "  No closed trades")
    print(f"  Total PnL: {df['pnl'].sum():+.1f} pips")

    if len(wins) > 0:
        print(f"  Avg win: +{wins['pnl'].mean():.1f} pips")
    if len(losses) > 0:
        print(f"  Avg loss: {losses['pnl'].mean():.1f} pips")

    daily = df.groupby('date').agg(
        signals=('signal', 'count'),
        wins=('outcome', lambda x: (x == 'WIN').sum()),
        losses=('outcome', lambda x: (x == 'LOSS').sum()),
        pnl=('pnl', 'sum')
    ).reset_index()

    print(f"\n  {'Date':>10s} {'Signals':>8s} {'Wins':>5s} {'Losses':>7s} {'PnL':>8s}")
    print("  " + "-" * 45)
    for _, row in daily.iterrows():
        print(f"  {row['date']:>10s} {row['signals']:8d} {row['wins']:5d} {row['losses']:7d} {row['pnl']:+8.1f}")

    print(f"\n  Signal breakdown:")
    longs = df[df['signal'] == 'LONG']
    shorts = df[df['signal'] == 'SHORT']
    print(f"    LONG: {len(longs)} signals, {(longs['outcome']=='WIN').sum()}W / {(longs['outcome']=='LOSS').sum()}L")
    print(f"    SHORT: {len(shorts)} signals, {(shorts['outcome']=='WIN').sum()}W / {(shorts['outcome']=='LOSS').sum()}L")


if __name__ == '__main__':
    backtest_june()

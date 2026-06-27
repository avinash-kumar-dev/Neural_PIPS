import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureEngineer
from src.labels import TripleBarrierLabeler
from src.signal_engine import SignalEngine
from src.confidence_scorer import ConfidenceScorer
from src.rule_filters import RuleFilters

MODELS_DIR = 'models/v7'


def load_models():
    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib'))
    lgbm = joblib.load(os.path.join(MODELS_DIR, 'lightgbm.joblib'))
    weights = joblib.load(os.path.join(MODELS_DIR, 'ensemble_weights.joblib'))
    models = [xgb, lgbm]
    cb_path = os.path.join(MODELS_DIR, 'catboost.joblib')
    if os.path.exists(cb_path):
        cb = joblib.load(cb_path)
        models.append(cb)
    return models, weights


def backtest_today():
    print("=" * 70)
    print("  BACKTEST: V4 Model on Today's Session (27 Jun 2026)")
    print("  V6: Phase 2 features (81) + H4 Hard Gate + Weighted Average Meta")
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

    labeler = TripleBarrierLabeler()
    labeled = labeler.label_dataframe(m5)

    rf = RuleFilters()
    filtered = rf.apply_all(labeled, raw_features)

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
        'feature_pipeline': feature_pipeline,
    })
    scorer = ConfidenceScorer({'threshold': 78})

    today = pd.Timestamp('2026-06-26')
    session_start = today.replace(hour=13, minute=0, second=0)
    session_end = today.replace(hour=16, minute=0, second=0)

    session_mask = (m5['time'] >= session_start) & (m5['time'] < session_end)
    session_indices = m5[session_mask].index.tolist()

    print(f"\n  Session: {session_start} to {session_end}")
    print(f"  M5 bars in session: {len(session_indices)}")

    print(f"\n  {'Check':>5s} {'Time':>8s} {'Signal':>8s} {'Conf':>5s} {'H4':>5s} {'Entry':>9s} {'TP':>9s} {'SL':>9s} {'Outcome':>10s} {'PnL':>6s}")
    print("  " + "-" * 85)

    signals = []
    check_num = 0

    for idx in session_indices:
        check_num += 1
        feat_row = raw_features.loc[idx].to_dict()
        feat_row['time'] = m5.loc[idx, 'time']
        feat_series = pd.Series(feat_row)
        raw_series = pd.Series(feat_row)

        engine_result = engine.generate_signal(feat_series, raw_series, models, weights)

        ts = m5.loc[idx, 'time'].strftime('%H:%M')
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
                'check': check_num,
                'time': ts,
                'signal': engine_result['signal'],
                'confidence': engine_result['confidence'],
                'h4': h4_str,
                'entry': entry_price,
                'tp': tp_price,
                'sl': sl_price,
                'outcome': outcome,
                'pnl': pnl,
            })

            outcome_str = f"{outcome}" if outcome != 'OPEN' else 'OPEN'
            print(f"  {check_num:5d} {ts:>8s} {engine_result['signal']:>8s} {engine_result['confidence']:5.1f} {h4_str:>5s} {entry_price:9.5f} {tp_price:9.5f} {sl_price:9.5f} {outcome_str:>10s} {pnl:+6.1f}")
        else:
            if check_num <= 5 or check_num % 10 == 0:
                print(f"  {check_num:5d} {ts:>8s} {'--':>8s} {'--':>5s} {h4_str:>5s} {'--':>9s} {'--':>9s} {'--':>9s} {engine_result['reason']:>10s} {'--':>6s}")

    print("\n" + "=" * 70)
    print("  SESSION SUMMARY")
    print("=" * 70)
    print(f"  Total checks: {check_num}")
    print(f"  Signals generated: {len(signals)}")

    if signals:
        wins = [s for s in signals if s['outcome'] == 'WIN']
        losses = [s for s in signals if s['outcome'] == 'LOSS']
        open_trades = [s for s in signals if s['outcome'] == 'OPEN']
        total_pnl = sum(s['pnl'] for s in signals)

        print(f"\n  Wins: {len(wins)} | Losses: {len(losses)} | Open: {len(open_trades)}")
        print(f"  Total PnL: {total_pnl:+.1f} pips")

        if wins or losses:
            closed = wins + losses
            win_rate = len(wins) / len(closed) * 100 if closed else 0
            print(f"  Win rate: {win_rate:.1f}%")

        print(f"\n  {'Check':>5s} {'Time':>8s} {'Signal':>8s} {'Conf':>5s} {'Entry':>9s} {'TP':>9s} {'SL':>9s} {'Outcome':>10s} {'PnL':>6s}")
        print("  " + "-" * 75)
        for s in signals:
            print(f"  {s['check']:5d} {s['time']:>8s} {s['signal']:>8s} {s['confidence']:5.1f} {s['entry']:9.5f} {s['tp']:9.5f} {s['sl']:9.5f} {s['outcome']:>10s} {s['pnl']:+6.1f}")
    else:
        print("\n  No signals generated.")

    print("\n" + "=" * 70)
    print("  COMPARISON: V3 (no H4 gate) vs V4 (with H4 gate)")
    print("=" * 70)
    print(f"  V3 (T=70): 10 SHORT signals, 7 losses, -102.4 pips")
    print(f"  V4 (T=78): {len(signals)} signals, H4 gate active")
    if signals:
        print(f"  V4 PnL: {total_pnl:+.1f} pips")
    print("=" * 70)


if __name__ == '__main__':
    backtest_today()

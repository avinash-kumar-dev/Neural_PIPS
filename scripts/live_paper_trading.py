import sys
import os
import json
import time
import signal
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.feature_pipeline import FeaturePipeline
from src.rule_filters import RuleFilters
from src.confidence_scorer import ConfidenceScorer
from src.signal_engine import SignalEngine

SIGNALS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'eurusd', 'live_signals.json')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'v3_regime')

running = True


def handle_signal(signum, frame):
    global running
    running = False
    print("\nGracefully shutting down...")


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def load_signals():
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_signals(signals):
    os.makedirs(os.path.dirname(SIGNALS_FILE), exist_ok=True)
    with open(SIGNALS_FILE, 'w') as f:
        json.dump(signals, f, indent=2, default=str)


def load_models():
    import joblib
    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib'))
    lgbm = joblib.load(os.path.join(MODELS_DIR, 'lightgbm.joblib'))
    meta = joblib.load(os.path.join(MODELS_DIR, 'meta.joblib'))
    pipeline = FeaturePipeline.load(os.path.join(MODELS_DIR, 'feature_pipeline.joblib'))
    return [xgb, lgbm], meta, pipeline


def check_signal_outcome(signal_entry, m5_df):
    if signal_entry.get('result') != 'pending':
        return signal_entry

    if 'tp_price' not in signal_entry or 'sl_price' not in signal_entry:
        entry_price = signal_entry.get('entry_price', 0)
        direction = signal_entry.get('signal', 'LONG')
        tp_pips = signal_entry['tp_pips']
        sl_pips = signal_entry['sl_pips']
        if direction == 'LONG':
            signal_entry['tp_price'] = round(entry_price + tp_pips / 10000, 5)
            signal_entry['sl_price'] = round(entry_price - sl_pips / 10000, 5)
        else:
            signal_entry['tp_price'] = round(entry_price - tp_pips / 10000, 5)
            signal_entry['sl_price'] = round(entry_price + sl_pips / 10000, 5)

    signal_time = pd.to_datetime(signal_entry['timestamp'])
    entry_price = signal_entry.get('entry_price', 0)
    tp_pips = signal_entry['tp_pips']
    sl_pips = signal_entry['sl_pips']

    future_bars = m5_df[pd.to_datetime(m5_df['time']) > signal_time].head(20)

    for _, bar in future_bars.iterrows():
        high_pips = (bar['high'] - entry_price) * 10000
        low_pips = (entry_price - bar['low']) * 10000

        if high_pips >= tp_pips:
            signal_entry['result'] = 'WIN'
            signal_entry['pnl_pips'] = tp_pips
            signal_entry['closed_at'] = str(bar['time'])
            return signal_entry

        if low_pips >= sl_pips:
            signal_entry['result'] = 'LOSS'
            signal_entry['pnl_pips'] = -sl_pips
            signal_entry['closed_at'] = str(bar['time'])
            return signal_entry

    return signal_entry


def run_check(fetcher, engineer, pipeline, models, meta_model, scorer, engine):
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour

    if not (13 <= hour < 16):
        return None, 'outside_session'

    try:
        m5_df = fetcher.fetch_ohlcv('M5', 200)
        m15_df = fetcher.fetch_ohlcv('M15', 100)
        h1_df = fetcher.fetch_ohlcv('H1', 100)
        h4_df = fetcher.fetch_ohlcv('H4', 50)
    except Exception as e:
        return None, f'fetch_error: {e}'

    from src.live_feature_engineering import LiveFeatureEngineer
    live_eng = LiveFeatureEngineer(engineer, pipeline)

    try:
        feat_row, raw_row = live_eng.compute_for_latest_bar(m5_df, m15_df, h1_df, h4_df)
    except Exception as e:
        return None, f'feature_error: {e}'

    feat_row['time'] = m5_df['time'].iloc[-1]

    ml_pred, ml_conf = engine._predict(feat_row, models, meta_model)
    direction = 'LONG' if ml_pred == 1 else 'SHORT'

    confidence, scores = scorer.score(feat_row, raw_row)
    if confidence < engine.confidence_threshold:
        return None, f'low_confidence_{confidence:.0f}'

    atr_pips = raw_row.get('atr_pips', 10)
    if atr_pips < engine.min_sl_pips:
        return None, 'atr_too_low'

    tp_pips = min(max(atr_pips * 4.5, engine.min_tp_pips), engine.max_tp_pips)
    sl_pips = min(max(atr_pips * 1.5, engine.min_sl_pips), engine.max_sl_pips)

    entry_price = m5_df['close'].iloc[-1]

    if direction == 'LONG':
        tp_price = round(entry_price + tp_pips / 10000, 5)
        sl_price = round(entry_price - sl_pips / 10000, 5)
    else:
        tp_price = round(entry_price - tp_pips / 10000, 5)
        sl_price = round(entry_price + sl_pips / 10000, 5)

    signal = {
        'id': len(load_signals()) + 1,
        'signal': direction,
        'confidence': round(confidence, 1),
        'ml_confidence': round(ml_conf, 3),
        'tp_pips': round(tp_pips, 1),
        'sl_pips': round(sl_pips, 1),
        'tp_sl_ratio': round(tp_pips / sl_pips, 1),
        'atr_pips': round(atr_pips, 1),
        'entry_price': round(entry_price, 5),
        'tp_price': tp_price,
        'sl_price': sl_price,
        'timestamp': str(m5_df['time'].iloc[-1]),
        'scores': {k: round(v, 1) for k, v in scores.items()},
        'result': 'pending',
        'pnl_pips': 0,
        'closed_at': None,
    }
    return signal, 'signal_generated'


def main():
    print("=" * 60)
    print("  EUR/USD Live Paper Trading")
    print("  Session: 13:00-16:00 UTC (18:00-21:00 PKT)")
    print("  Threshold: 78 | TP:SL = 3:1 | V3 regime-conditional")
    print("=" * 60)

    print("\nLoading models...")
    models, meta_model, pipeline = load_models()
    print(f"  XGBoost: {type(models[0]).__name__}")
    print(f"  LightGBM: {type(models[1]).__name__}")
    print(f"  Meta: {type(meta_model).__name__}")
    print(f"  Pipeline: {len(pipeline.feature_names_out_)} features")

    engineer = FeatureEngineer()
    scorer = ConfidenceScorer({'threshold': 78})
    engine = SignalEngine({'confidence_threshold': 78, 'session_start': 13, 'session_end': 16})
    rule_filters = RuleFilters()

    print("\nConnecting to MT5...")
    fetcher = DataFetcher(host='localhost', port=8001)
    try:
        info = fetcher.connect()
        print(f"  Connected! Account: {info.login}, Balance: {info.balance}")
    except Exception as e:
        print(f"  Connection failed: {e}")
        print("  Make sure MT5 Docker is running and RPyC server is started.")
        return

    signals = load_signals()
    print(f"\nLoaded {len(signals)} existing signals")

    now_utc = datetime.now(timezone.utc)
    print(f"\nCurrent time: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"              {(now_utc + timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S')} PKT")

    if not (13 <= now_utc.hour < 16):
        next_session = now_utc.replace(hour=13, minute=0, second=0, microsecond=0)
        if now_utc.hour >= 16:
            next_session += timedelta(days=1)
        wait_seconds = (next_session - now_utc).total_seconds()
        print(f"\nOutside session. Next session starts at {next_session.strftime('%H:%M')} UTC")
        print(f"Waiting {wait_seconds/3600:.1f} hours...")

        while running and datetime.now(timezone.utc) < next_session:
            time.sleep(30)
        if not running:
            fetcher.disconnect()
            return

    print("\nStarting live signal monitoring...")
    print("-" * 60)

    check_count = 0
    signal_count = 0

    while running:
        now_utc = datetime.now(timezone.utc)

        if not (13 <= now_utc.hour < 16):
            print(f"\n[{now_utc.strftime('%H:%M:%S')}] Session ended. Stopping.")
            break

        check_count += 1
        signal_obj, reason = run_check(fetcher, engineer, pipeline, models, meta_model, scorer, engine)

        ts = now_utc.strftime('%H:%M:%S')
        if signal_obj:
            signal_count += 1
            signals.append(signal_obj)
            save_signals(signals)

            print(f"\n[{ts}] === SIGNAL #{signal_obj['id']} ===")
            print(f"  Direction:  {signal_obj['signal']}")
            print(f"  Confidence: {signal_obj['confidence']}/100")
            print(f"  Entry:      {signal_obj['entry_price']}")
            print(f"  TP:         {signal_obj['tp_pips']} pips ({signal_obj['tp_price']})")
            print(f"  SL:         {signal_obj['sl_pips']} pips ({signal_obj['sl_price']})")
            print(f"  Ratio:      1:{signal_obj['tp_sl_ratio']}")
            print(f"  ATR:        {signal_obj['atr_pips']} pips")
            print(f"  Scores:     {signal_obj['scores']}")
        else:
            print(f"[{ts}] Check #{check_count}: {reason} | Signals today: {signal_count}", end='\r')

        for i in range(300):
            if not running:
                break
            time.sleep(1)

    fetcher.disconnect()

    print(f"\n\n{'=' * 60}")
    print(f"  Session Summary")
    print(f"{'=' * 60}")
    print(f"  Total checks:  {check_count}")
    print(f"  Signals today: {signal_count}")
    print(f"  Total signals: {len(signals)}")
    print(f"  Results saved to: {SIGNALS_FILE}")

    pending = [s for s in signals if s['result'] == 'pending']
    wins = [s for s in signals if s['result'] == 'WIN']
    losses = [s for s in signals if s['result'] == 'LOSS']
    longs = [s for s in signals if s['signal'] == 'LONG']
    shorts = [s for s in signals if s['signal'] == 'SHORT']
    print(f"  Pending: {len(pending)} | Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"  LONG: {len(longs)} | SHORT: {len(shorts)}")
    if wins or losses:
        total_pnl = sum(s['pnl_pips'] for s in wins + losses)
        print(f"  Total PnL: {total_pnl:+.1f} pips")


if __name__ == '__main__':
    main()

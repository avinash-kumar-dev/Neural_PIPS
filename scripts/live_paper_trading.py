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
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'v7')

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


SESSIONS = [
    {'start': 7, 'end': 9, 'name': 'London Open'},
    {'start': 13, 'end': 16, 'name': 'Overlap'},
    {'start': 17, 'end': 20, 'name': 'NY Afternoon'},
]


def in_session(hour):
    for s in SESSIONS:
        if s['start'] <= hour < s['end']:
            return True
    return False


def get_session_name(hour):
    for s in SESSIONS:
        if s['start'] <= hour < s['end']:
            return s['name']
    return None


def run_check(fetcher, engineer, pipeline, models, weights, scorer, engine):
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour

    if not in_session(hour):
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

    feat_series = pd.Series(feat_row)
    raw_series = pd.Series(raw_row)

    engine_result = engine.generate_signal(feat_series, raw_series, models, weights)

    if engine_result['signal'] == 'NO-TRADE':
        return None, engine_result['reason']

    entry_price = m5_df['close'].iloc[-1]
    if engine_result['signal'] == 'LONG':
        tp_price = round(entry_price + engine_result['tp_pips'] / 10000, 5)
        sl_price = round(entry_price - engine_result['sl_pips'] / 10000, 5)
    else:
        tp_price = round(entry_price - engine_result['tp_pips'] / 10000, 5)
        sl_price = round(entry_price + engine_result['sl_pips'] / 10000, 5)

    signal = {
        'id': len(load_signals()) + 1,
        'signal': engine_result['signal'],
        'confidence': round(engine_result['confidence'], 1),
        'ml_confidence': round(engine_result['ml_confidence'], 3),
        'tp_pips': round(engine_result['tp_pips'], 1),
        'sl_pips': round(engine_result['sl_pips'], 1),
        'tp_sl_ratio': round(engine_result['tp_sl_ratio'], 1),
        'atr_pips': round(engine_result['atr_pips'], 1),
        'entry_price': round(entry_price, 5),
        'tp_price': tp_price,
        'sl_price': sl_price,
        'timestamp': str(m5_df['time'].iloc[-1]),
        'scores': {k: round(v, 1) for k, v in engine_result['scores'].items()},
        'result': 'pending',
        'pnl_pips': 0,
        'closed_at': None,
    }
    return signal, 'signal_generated'


def main():
    print("=" * 60)
    print("  EUR/USD Live Paper Trading")
    print("  Sessions: 07-09, 13-16, 17-20 UTC")
    print("  Threshold: 78 | TP:SL = 3:1 | V7 (XGB+LGBM+CatBoost)")
    print("=" * 60)

    print("\nLoading models...")
    models, weights, pipeline = load_models()
    print(f"  XGBoost: {type(models[0]).__name__}")
    print(f"  LightGBM: {type(models[1]).__name__}")
    if len(models) > 2:
        print(f"  CatBoost: {type(models[2]).__name__}")
    print(f"  Ensemble weights: [{', '.join(f'{w:.3f}' for w in weights)}]")
    print(f"  Pipeline: {len(pipeline.feature_names_out_)} features")

    engineer = FeatureEngineer()
    scorer = ConfidenceScorer({'threshold': 78})
    engine = SignalEngine({
        'confidence_threshold': 78,
        'ml_gate': 0.75,
        'confluence_min': 2,
        'max_spread_pips': 0.5,
        'min_atr_pips': 3.0,
        'max_atr_pips': 20.0,
        'min_atr_ratio': 0.5,
        'max_atr_ratio': 2.5,
        'sessions': SESSIONS,
        'feature_pipeline': pipeline,
    })
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

    if not in_session(now_utc.hour):
        next_session = None
        for s in SESSIONS:
            if now_utc.hour < s['start']:
                candidate = now_utc.replace(hour=s['start'], minute=0, second=0, microsecond=0)
                if candidate > now_utc:
                    next_session = candidate
                    break
            elif now_utc.hour >= s['end'] and now_utc.hour < 17:
                candidate = now_utc.replace(hour=17, minute=0, second=0, microsecond=0)
                if candidate > now_utc:
                    next_session = candidate
                    break
        if next_session is None:
            tomorrow = now_utc + timedelta(days=1)
            next_session = tomorrow.replace(hour=7, minute=0, second=0, microsecond=0)
        wait_seconds = (next_session - now_utc).total_seconds()
        print(f"\nOutside session. Next session: {next_session.strftime('%H:%M')} UTC")
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

        if not in_session(now_utc.hour):
            session_name = get_session_name(now_utc.hour)
            if session_name is None:
                print(f"\n[{now_utc.strftime('%H:%M:%S')}] All sessions ended. Stopping.")
                break
            time.sleep(30)
            continue

        check_count += 1
        signal_obj, reason = run_check(fetcher, engineer, pipeline, models, weights, scorer, engine)

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

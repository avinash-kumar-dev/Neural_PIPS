# PHASE 1: ARCHITECTURE DESIGN — XAU/USD Scalping Signal Engine

*Comprehensive system design. Every component specified for implementation.*

---

## 1. System Overview

### 1.1 What We're Building

A real-time XAU/USD scalping signal engine that:
- Analyzes 4 timeframes simultaneously (H1, M15, M5, M1)
- Uses ternary classification (LONG / SHORT / NO-TRADE)
- Requires extreme selectivity: only 2-15% of candles produce actionable signals
- Targets >90% directional accuracy on selected signals
- Runs inference in <100ms per signal check
- Connects to MetaTrader 5 for data and execution

### 1.2 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MT5 TERMINAL (Windows)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ XAUUSD   │  │ DXY      │  │ US10Y    │  │ Tick Data Feed   │   │
│  │ M1/M5/   │  │ (if      │  │ (if      │  │ copy_ticks_from  │   │
│  │ M15/H1   │  │ offered) │  │ offered) │  │                  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│       │              │              │                  │             │
│       └──────────────┴──────────────┴──────────────────┘             │
│                           │                                          │
│                    MT5 Python API                                    │
│                    (socket connection)                               │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PYTHON SIGNAL ENGINE (local)                      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    DATA PIPELINE                              │   │
│  │  MT5 Fetch → DataFrame → Parquet Storage → Feature Compute   │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │              MULTI-TIMEFRAME CONFIRMATION                     │   │
│  │  H1 (Trend) → M15 (Structure) → M5 (Model) → M1 (Entry)    │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │                    MODEL INFERENCE                            │   │
│  │  XGBoost M1 (calibrated) + XGBoost M5 (calibrated)          │   │
│  │  Agreement Gate + Confidence Threshold                        │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│  ┌──────────────────────────▼───────────────────────────────────┐   │
│  │                    SIGNAL GATE PIPELINE                       │   │
│  │  Session → News → Volatility → Spread → H1 → M15 → Model    │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│                      FINAL SIGNAL                                    │
│               LONG / SHORT / NO-TRADE                                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                 EXECUTION / NOTIFICATION                      │   │
│  │  MT5 OrderSend (paper/live) + Telegram/Email Alert           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                 MONITORING & HEALTH                           │   │
│  │  Heartbeat Protocol + Spread Tracker + Accuracy Monitor      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Event-Driven Architecture

The system uses an event-driven loop (inspired by PyEventBT and production MT5 bots). Every M1 candle close triggers the pipeline:

```
M1 Candle Close Event
    │
    ├── 1. Fetch latest OHLCV for M1, M5, M15, H1
    ├── 2. Fetch latest DXY data (if available)
    ├── 3. Fetch latest tick data for CVD
    ├── 4. Compute all features (Tier 1 + Tier 2)
    ├── 5. Run H1 trend filter
    ├── 6. Run M15 structure filter
    ├── 7. Run M5 model prediction
    ├── 8. Run M1 model prediction
    ├── 9. Check agreement gate
    ├── 10. Run macro gates (session, news, volatility, spread)
    ├── 11. Emit signal (LONG/SHORT/NO-TRADE)
    └── 12. Log + notify
```

---

## 2. Data Pipeline

### 2.1 MT5 Connection Layer

```python
# Pseudocode — exact API calls documented

import MetaTrader5 as mt5

# Connection
mt5.initialize()
mt5.login(account_id, password, server)

# Heartbeat (every 5 seconds)
def heartbeat():
    ping = mt5.account_info()
    if ping is None:
        # Connection dead → halt all signal generation
        # Send emergency alert
        raise ConnectionError("MT5 heartbeat failed")
    return True
```

### 2.2 Data Fetching (per candle close)

```python
# M1 OHLCV — last 100 candles
m1_data = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M1, 0, 100)

# M5 OHLCV — last 50 candles
m5_data = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 50)

# M15 OHLCV — last 30 candles
m15_data = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 30)

# H1 OHLCV — last 50 candles
h1_data = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 50)

# DXY (if broker offers symbol)
dxy_data = mt5.copy_rates_from_pos("DXY", mt5.TIMEFRAME_M1, 0, 100)

# US10Y (if broker offers symbol)
us10y_data = mt5.copy_rates_from_pos("US10Y", mt5.TIMEFRAME_M1, 0, 100)

# Tick data for CVD (last 500 ticks)
ticks = mt5.copy_ticks_from("XAUUSD", last_candle_time, 500, mt5.COPY_TICKS_ALL)
```

### 2.3 Historical Data Storage

**Format**: Parquet files, partitioned by timeframe and date.

```
data/
├── xauusd/
│   ├── M1/
│   │   ├── 2024-01.parquet
│   │   ├── 2024-02.parquet
│   │   └── ...
│   ├── M5/
│   ├── M15/
│   └── H1/
├── dxy/
│   └── M1/
├── us10y/
│   └── daily/
├── ticks/
│   └── 2024-01/
│       ├── day_01.parquet
│       └── ...
└── features/
    ├── m1_features/
    ├── m5_features/
    └── ...
```

### 2.4 Data Pipeline Schedule

| Task | Frequency | Latency Budget |
|------|-----------|----------------|
| Historical data pull (backfill) | Daily at 00:00 UTC | N/A (background) |
| M1 OHLCV fetch | Every M1 candle close | <100ms |
| M5/M15/H1 OHLCV fetch | Every M1 candle close | <50ms (cached) |
| DXY fetch | Every M1 candle close | <100ms |
| Tick data fetch | Every M1 candle close | <200ms |
| Feature computation | After data fetch | <100ms |
| Model inference | After features | <50ms |
| **Total pipeline** | **Per M1 candle** | **<500ms** |

---

## 3. Feature Engineering Module

### 3.1 Feature Computation Functions

Each function takes a DataFrame of OHLCV data and returns a Series/DataFrame of features.

#### Tier 1 Features (Highest Alpha)

```python
# FEATURE 1: DXY Lag Returns
# Input: DXY DataFrame (M1)
# Output: 3 features
def compute_dxy_lag_features(dxy_df):
    returns = dxy_df['close'].pct_change()
    return {
        'dxy_return_lag1': returns.shift(1),      # t-1 candle return
        'dxy_return_lag2': returns.shift(2),      # t-2 candle return
        'dxy_return_lag3': returns.shift(3),      # t-3 candle return
    }

# FEATURE 2: DXY Rate of Change (3-period momentum)
def compute_dxy_roc(dxy_df, period=3):
    return (dxy_df['close'] - dxy_df['close'].shift(period)) / dxy_df['close'].shift(period)

# FEATURE 3: US10Y Real Yield Change
# Input: DGS10 + T10YIE from FRED (daily, interpolated to M1)
def compute_real_yield_change(us10y_df, breakeven_df):
    real_yield = us10y_df['DGS10'] - breakeven_df['T10YIE']
    return real_yield.diff()  # Change from previous candle

# FEATURE 4: CVD Divergence Signal
# Input: Tick data (MT5 copy_ticks_from)
def compute_cvd_features(ticks_df, price_df):
    # Classify each tick as buy or sell
    ticks_df['direction'] = np.where(
        ticks_df['last'] >= ticks_df['last'].shift(1), 1, -1
    )
    ticks_df['cvd_delta'] = ticks_df['volume'] * ticks_df['direction']

    # Cumulate per M1 bar
    cvd_per_bar = ticks_df.groupby(pd.Grouper(freq='1min'))['cvd_delta'].sum()

    # Divergence: price making new high but CVD not
    price_high_5 = price_df['high'].rolling(5).max()
    cvd_high_5 = cvd_per_bar.rolling(5).max()

    divergence = np.where(
        (price_df['high'] >= price_high_5.shift(1)) &
        (cvd_per_bar < cvd_high_5.shift(1) * 0.8),  # CVD failing to confirm
        -1,  # Bearish divergence
        np.where(
            (price_df['low'] <= price_df['low'].rolling(5).min().shift(1)) &
            (cvd_per_bar > cvd_per_bar.rolling(5).min().shift(1) * 1.2),
            1,   # Bullish divergence
            0    # No divergence
        )
    )

    return {
        'cvd_delta': cvd_per_bar,
        'cvd_divergence': divergence,
        'cvd_roc_5': cvd_per_bar.pct_change(5),
    }

# FEATURE 5: FVG Detection and Distance
def compute_fvg_features(df, lookback=50):
    fvgs = []
    for i in range(2, min(lookback, len(df)-1)):
        # Bullish FVG: Low[i] > High[i-2] AND Low[i] > High[i+1] if exists
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
            gap_midpoint = (df['low'].iloc[i] + df['high'].iloc[i-2]) / 2
            fvgs.append({
                'type': 'bullish',
                'midpoint': gap_midpoint,
                'size': gap_size,
                'index': i
            })
        # Bearish FVG
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
            gap_midpoint = (df['high'].iloc[i] + df['low'].iloc[i-2]) / 2
            fvgs.append({
                'type': 'bearish',
                'midpoint': gap_midpoint,
                'size': gap_size,
                'index': i
            })

    # Distance from current price to nearest unfilled FVG
    current_price = df['close'].iloc[-1]
    unfilled = [f for f in fvgs if f['index'] < len(df) - 1]  # Not yet filled

    if unfilled:
        distances = [abs(current_price - f['midpoint']) for f in unfilled]
        nearest_idx = np.argmin(distances)
        return {
            'fvg_nearest_distance': distances[nearest_idx],
            'fvg_nearest_type': 1 if unfilled[nearest_idx]['type'] == 'bullish' else -1,
            'fvg_nearest_size': unfilled[nearest_idx]['size'],
            'fvg_count_unfilled': len(unfilled),
        }
    else:
        return {
            'fvg_nearest_distance': 999,  # No FVG
            'fvg_nearest_type': 0,
            'fvg_nearest_size': 0,
            'fvg_count_unfilled': 0,
        }

# FEATURE 6: VWAP Distance (Normalized)
def compute_vwap_features(df, session_start_idx):
    # Session VWAP anchored at London open (or specified session)
    session_data = df.iloc[session_start_idx:]
    typical_price = (session_data['high'] + session_data['low'] + session_data['close']) / 3
    volume = session_data['tick_volume']

    vwap = (typical_price * volume).cumsum() / volume.cumsum()

    # Normalized distance
    atr = compute_atr(df, period=14)
    distance = (df['close'] - vwap) / atr

    return {
        'vwap_distance': distance.iloc[-1],
        'vwap_direction': np.sign(distance.iloc[-1]),  # +1 above, -1 below
    }

# FEATURE 7: Session Cyclical Encoding
def compute_session_features(timestamp_utc):
    minute_of_day = timestamp_utc.hour * 60 + timestamp_utc.minute
    return {
        'session_sin': np.sin(2 * np.pi * minute_of_day / 1440),
        'session_cos': np.cos(2 * np.pi * minute_of_day / 1440),
        'is_london_ny_overlap': 1 if 13 <= timestamp_utc.hour < 17 else 0,
        'is_london_session': 1 if 6 <= timestamp_utc.hour < 17 else 0,
        'is_ny_session': 1 if 13 <= timestamp_utc.hour < 21 else 0,
        'is_asian_session': 1 if (timestamp_utc.hour >= 22 or timestamp_utc.hour < 6) else 0,
    }
```

#### Tier 2 Features (Supporting)

```python
# ATR Ratio (Volatility Filter)
def compute_atr_ratio(df, period=14, lookback=20):
    atr = compute_atr(df, period)
    atr_mean = atr.rolling(lookback).mean()
    return atr / atr_mean  # >2.5 = spike, <0.5 = dead

# EMA Slopes (M5, M15, H1)
def compute_ema_slopes(df, periods=[8, 21, 50]):
    slopes = {}
    for p in periods:
        ema = df['close'].ewm(span=p).mean()
        slopes[f'ema_{p}_slope'] = ema.pct_change(3)  # 3-candle slope
        slopes[f'ema_{p}_above'] = (df['close'] > ema).astype(int)
    return slopes

# RSI
def compute_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Stochastic RSI
def compute_stoch_rsi(df, rsi_period=14, stoch_period=14, k=3, d=3):
    rsi = compute_rsi(df, rsi_period)
    stoch_rsi = (rsi - rsi.rolling(stoch_period).min()) / \
                (rsi.rolling(stoch_period).max() - rsi.rolling(stoch_period).min())
    k_line = stoch_rsi.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return {'stoch_rsi_k': k_line, 'stoch_rsi_d': d_line}

# Bollinger Band Position
def compute_bb_position(df, period=20, std=2):
    sma = df['close'].rolling(period).mean()
    std_dev = df['close'].rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return (df['close'] - lower) / (upper - lower)  # 0=lower, 1=upper

# Price Change Lags (Autocorrelation)
def compute_price_lags(df):
    returns = df['close'].pct_change()
    return {
        'return_lag1': returns.shift(1),
        'return_lag2': returns.shift(2),
        'return_lag3': returns.shift(3),
        'return_lag5': returns.shift(5),
    }
```

### 3.2 Feature Vector (Final Input to Model)

```python
# M1 Feature Vector (32 features)
M1_FEATURES = [
    # Tier 1 — 13 features
    'dxy_return_lag1', 'dxy_return_lag2', 'dxy_return_lag3',
    'dxy_roc_3',
    'real_yield_change',
    'cvd_delta', 'cvd_divergence', 'cvd_roc_5',
    'fvg_nearest_distance', 'fvg_nearest_type', 'fvg_nearest_size',
    'vwap_distance',
    'session_sin', 'session_cos',

    # Tier 2 — 19 features
    'atr_ratio',
    'ema_8_slope', 'ema_21_slope', 'ema_50_slope',
    'rsi_14',
    'stoch_rsi_k', 'stoch_rsi_d',
    'bb_position',
    'return_lag1', 'return_lag2', 'return_lag3', 'return_lag5',
    'volume_ratio_20',
    'candle_body_ratio',  # body / total range
    'is_london_ny_overlap',
    'spread_current',
]

# M5 Feature Vector (same 32 features, computed from M5 data)
M5_FEATURES = M1_FEATURES  # Same features, different timeframe
```

---

## 4. Multi-Timeframe Confirmation System

### 4.1 H1 Trend Filter (Directional Bias)

```python
class H1TrendFilter:
    """
    Determines the allowed trading direction based on H1 structure.
    Returns: 'LONG_ONLY', 'SHORT_ONLY', 'BOTH' (ranging)
    """
    def evaluate(self, h1_data):
        ema_50 = h1_data['close'].ewm(span=50).mean()
        ema_200 = h1_data['close'].ewm(span=200).mean()

        current_price = h1_data['close'].iloc[-1]

        # Higher highs and higher lows on H1
        swing_highs = self._find_swing_highs(h1_data, lookback=20)
        swing_lows = self._find_swing_lows(h1_data, lookback=20)

        uptrend = (
            current_price > ema_50.iloc[-1] and
            ema_50.iloc[-1] > ema_200.iloc[-1] and
            len(swing_highs) >= 2 and swing_highs[-1] > swing_highs[-2] and
            len(swing_lows) >= 2 and swing_lows[-1] > swing_lows[-2]
        )

        downtrend = (
            current_price < ema_50.iloc[-1] and
            ema_50.iloc[-1] < ema_200.iloc[-1] and
            len(swing_highs) >= 2 and swing_highs[-1] < swing_highs[-2] and
            len(swing_lows) >= 2 and swing_lows[-1] < swing_lows[-2]
        )

        if uptrend:
            return 'LONG_ONLY'
        elif downtrend:
            return 'SHORT_ONLY'
        else:
            return 'BOTH'  # Ranging — both directions allowed
```

### 4.2 M15 Structure Filter (Pullback Detection)

```python
class M15StructureFilter:
    """
    Checks if price is in a valid pullback zone on M15.
    Returns: 'VALID_LONG', 'VALID_SHORT', 'INVALID'
    """
    def evaluate(self, m15_data, h1_bias):
        vwap = self._compute_session_vwap(m15_data)
        current_price = m15_data['close'].iloc[-1]

        # Find M15 order blocks and FVGs
        order_blocks = self._find_order_blocks(m15_data, lookback=30)
        fvgs = self._find_fvgs(m15_data, lookback=30)

        # LONG setup: price pulling back to bullish OB or bullish FVG
        if h1_bias in ('LONG_ONLY', 'BOTH'):
            near_bullish_ob = any(
                abs(current_price - ob['bottom']) < 2.0  # Within 2 pips
                for ob in order_blocks if ob['type'] == 'bullish'
            )
            near_bullish_fvg = any(
                abs(current_price - fvg['midpoint']) < 2.0
                for fvg in fvgs if fvg['type'] == 'bullish'
            )
            above_vwap = current_price > vwap

            if (near_bullish_ob or near_bullish_fvg) and above_vwap:
                return 'VALID_LONG'

        # SHORT setup: price rallying to bearish OB or bearish FVG
        if h1_bias in ('SHORT_ONLY', 'BOTH'):
            near_bearish_ob = any(
                abs(current_price - ob['top']) < 2.0
                for ob in order_blocks if ob['type'] == 'bearish'
            )
            near_bearish_fvg = any(
                abs(current_price - fvg['midpoint']) < 2.0
                for fvg in fvgs if fvg['type'] == 'bearish'
            )
            below_vwap = current_price < vwap

            if (near_bearish_ob or near_bearish_fvg) and below_vwap:
                return 'VALID_SHORT'

        return 'INVALID'
```

### 4.3 Model Agreement Gate

```python
class AgreementGate:
    """
    Both M5 and M1 models must agree on direction AND confidence.
    """
    def evaluate(self, m5_prediction, m1_prediction):
        m5_direction = m5_prediction['direction']  # LONG/SHORT/NO_TRADE
        m1_direction = m1_prediction['direction']
        m5_confidence = m5_prediction['confidence']
        m1_confidence = m1_prediction['confidence']

        # Both must predict the same direction
        if m5_direction != m1_direction:
            return 'NO_TRADE', 0.0

        if m5_direction == 'NO_TRADE' or m1_direction == 'NO_TRADE':
            return 'NO_TRADE', 0.0

        # Confidence thresholds
        M5_THRESHOLD = 0.75  # M5 must be >75% confident
        M1_THRESHOLD = 0.85  # M1 must be >85% confident

        if m5_confidence < M5_THRESHOLD or m1_confidence < M1_THRESHOLD:
            return 'NO_TRADE', 0.0

        # Combined confidence = geometric mean
        combined_confidence = np.sqrt(m5_confidence * m1_confidence)

        return m5_direction, combined_confidence
```

---

## 5. Model Architecture

### 5.1 XGBoost Configuration

```python
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

# Model 1: M1 XGBoost
m1_model_params = {
    'objective': 'multi:softprob',
    'num_class': 3,  # LONG=0, SHORT=1, NO_TRADE=2
    'n_estimators': 500,
    'max_depth': 4,           # Shallow — prevent overfitting
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,    # Regularization
    'gamma': 0.1,             # Regularization
    'reg_alpha': 0.1,         # L1 regularization
    'reg_lambda': 1.0,        # L2 regularization
    'scale_pos_weight': 1.0,  # Adjusted during training based on class distribution
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 50,
    'random_state': 42,
}

# Model 2: M5 XGBoost (same architecture, different training data)
m5_model_params = m1_model_params.copy()
```

### 5.2 Label Generation (Triple-Barrier Method)

```python
class TripleBarrierLabeler:
    """
    López de Prado's Triple-Barrier Method.

    For each candle i, look forward up to `horizon` candles:
    - LONG: price hits +TP before -SL
    - SHORT: price hits -SL before +TP (price fell first)
    - NO_TRADE: neither barrier hit within horizon
    """
    def __init__(self, tp_pips=10, sl_pips=6, horizon=10):
        self.tp = tp_pips * 0.01  # Convert pips to price
        self.sl = sl_pips * 0.01
        self.horizon = horizon

    def label(self, df):
        labels = []
        for i in range(len(df) - self.horizon):
            entry_price = df['close'].iloc[i]
            label = 2  # Default: NO_TRADE

            for j in range(1, self.horizon + 1):
                future_high = df['high'].iloc[i + j]
                future_low = df['low'].iloc[i + j]

                hit_tp = future_high >= entry_price + self.tp
                hit_sl = future_low <= entry_price - self.sl

                if hit_tp and not hit_sl:
                    label = 0  # LONG
                    break
                elif hit_sl and not hit_tp:
                    label = 1  # SHORT
                    break
                elif hit_tp and hit_sl:
                    # Both hit — use first barrier touched
                    # Check which was hit first within the candle
                    if (future_high - entry_price) / self.tp < \
                       (entry_price - future_low) / self.sl:
                        label = 0  # TP closer
                    else:
                        label = 1  # SL closer
                    break

            labels.append(label)

        # Pad end with NO_TRADE
        labels.extend([2] * self.horizon)
        return np.array(labels)
```

### 5.3 Probability Calibration

```python
class ModelCalibrator:
    """
    XGBoost's predict_proba() is NOT well-calibrated.
    Use Isotonic Regression on a held-out calibration set.

    CRITICAL: Calibration set must be:
    1. Separate from training data
    2. Separate from test data
    3. Recent (most recent 20% of training window before test window)
    """
    def __init__(self):
        self.calibrators = {}  # One per class

    def fit(self, model, X_cal, y_cal):
        """Fit isotonic regression on calibration set probabilities."""
        raw_probs = model.predict_proba(X_cal)

        for class_idx in range(3):
            binary_labels = (y_cal == class_idx).astype(int)
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(raw_probs[:, class_idx], binary_labels)
            self.calibrators[class_idx] = iso

    def calibrate(self, raw_probs):
        """Apply calibration to raw probabilities."""
        calibrated = np.zeros_like(raw_probs)
        for class_idx in range(3):
            calibrated[:, class_idx] = self.calibrators[class_idx].predict(
                raw_probs[:, class_idx]
            )
        # Renormalize to sum to 1
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        return calibrated
```

### 5.4 Training Pipeline

```python
class ModelTrainer:
    """
    Walk-Forward training with purging and embargo.

    Data split for each fold:
    [Training Data] --purge(5 candles)-- [Embargo(10 candles)] -- [Test Data]
    """
    def __init__(self, purge_gap=5, embargo_gap=10):
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap

    def walk_forward_train(self, X, y, timestamps,
                           initial_train_pct=0.4,
                           test_pct=0.1,
                           step_pct=0.05):
        """
        Expanding window walk-forward CV.

        Returns:
            List of (model, calibrator, test_predictions, test_timestamps)
        """
        n = len(X)
        results = []

        initial_train_end = int(n * initial_train_pct)
        test_size = int(n * test_pct)
        step_size = int(n * step_pct)

        for start in range(initial_train_end, n - test_size, step_size):
            # Training window: everything before start, with purge
            train_end = start - self.purge_gap
            X_train = X[:train_end]
            y_train = y[:train_end]

            # Test window
            test_start = start + self.embargo_gap
            test_end = min(test_start + test_size, n)
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]
            timestamps_test = timestamps[test_start:test_end]

            # Calibration set: last 20% of training window
            cal_size = int(len(X_train) * 0.2)
            X_cal = X_train[-cal_size:]
            y_cal = y_train[-cal_size:]
            X_train_main = X_train[:-cal_size]
            y_train_main = y_train[:-cal_size]

            # Train XGBoost
            model = xgb.XGBClassifier(**m1_model_params)
            model.fit(
                X_train_main, y_train_main,
                eval_set=[(X_cal, y_cal)],
                verbose=False
            )

            # Calibrate
            calibrator = ModelCalibrator()
            calibrator.fit(model, X_cal, y_cal)

            # Predict on test
            raw_probs = model.predict_proba(X_test)
            calibrated_probs = calibrator.calibrate(raw_probs)

            results.append({
                'model': model,
                'calibrator': calibrator,
                'predictions': calibrated_probs,
                'labels': y_test,
                'timestamps': timestamps_test,
            })

        return results
```

---

## 6. Signal Gate Pipeline (Inference)

### 6.1 Complete Signal Gate

```python
class SignalGate:
    """
    The final decision gate. Runs all filters in sequence.
    Any filter that fails → NO_TRADE.
    """
    def __init__(self):
        self.h1_filter = H1TrendFilter()
        self.m15_filter = M15StructureFilter()
        self.agreement_gate = AgreementGate()
        self.news_filter = NewsFilter()
        self.volatility_filter = VolatilityFilter()

    def evaluate(self, market_data, m5_prediction, m1_prediction):
        """
        Returns: (signal, confidence, rejection_reason)
        """
        signal = 'NO_TRADE'
        confidence = 0.0
        reason = None

        # Gate 1: Session filter
        current_hour = market_data['timestamp'].hour
        if not (8 <= current_hour < 17):  # Only London/NY
            reason = 'OUTSIDE_SESSION'
            return signal, confidence, reason

        # Gate 2: News filter
        if self.news_filter.is_news_within_15_min(market_data['timestamp']):
            reason = 'NEWS_EVENT'
            return signal, confidence, reason

        # Gate 3: Volatility filter
        atr_ratio = self.volatility_filter.compute_atr_ratio(market_data['m1_data'])
        if atr_ratio < 0.5 or atr_ratio > 2.5:
            reason = 'VOLATILITY_OUT_OF_RANGE'
            return signal, confidence, reason

        # Gate 4: Spread filter
        current_spread = market_data['spread']
        if current_spread > 0.30:  # 0.30 pips max
            reason = 'SPREAD_TOO_HIGH'
            return signal, confidence, reason

        # Gate 5: H1 trend filter
        h1_bias = self.h1_filter.evaluate(market_data['h1_data'])

        # Gate 6: M15 structure filter
        m15_status = self.m15_filter.evaluate(market_data['m15_data'], h1_bias)

        # Gate 7: Model agreement gate
        raw_signal, raw_confidence = self.agreement_gate.evaluate(
            m5_prediction, m1_prediction
        )

        if raw_signal == 'NO_TRADE':
            reason = 'MODEL_NO_AGREEMENT'
            return signal, confidence, reason

        # Gate 8: Direction must match H1 bias
        if h1_bias == 'LONG_ONLY' and raw_signal == 'SHORT':
            reason = 'AGAINST_H1_TREND'
            return signal, confidence, reason
        if h1_bias == 'SHORT_ONLY' and raw_signal == 'LONG':
            reason = 'AGAINST_H1_TREND'
            return signal, confidence, reason

        # Gate 9: M15 structure must be valid
        if raw_signal == 'LONG' and m15_status != 'VALID_LONG':
            reason = 'M15_STRUCTURE_INVALID'
            return signal, confidence, reason
        if raw_signal == 'SHORT' and m15_status != 'VALID_SHORT':
            reason = 'M15_STRUCTURE_INVALID'
            return signal, confidence, reason

        # All gates passed
        signal = raw_signal
        confidence = raw_confidence
        reason = None

        return signal, confidence, reason
```

### 6.2 News Filter

```python
class NewsFilter:
    """
    Checks if a high-impact USD event is within ±15 minutes.
    Uses ForexFactory JSON feed (free, no auth needed).
    """
    def __init__(self):
        self.calendar_cache = None
        self.cache_time = None

    def fetch_calendar(self):
        """Fetch current week's events. Cache for 1 hour."""
        import requests
        from datetime import datetime

        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        response = requests.get(url)
        self.calendar_cache = response.json()
        self.cache_time = datetime.now()

    def is_news_within_15_min(self, timestamp):
        """Returns True if high-impact USD event within ±15 min."""
        if self.calendar_cache is None or \
           (datetime.now() - self.cache_time).seconds > 3600:
            self.fetch_calendar()

        for event in self.calendar_cache:
            if event.get('impact') != 'High':
                continue
            if event.get('currency') not in ('USD', 'XAU'):
                continue

            event_time = datetime.fromisoformat(
                event['date'].replace('Z', '+00:00')
            )
            diff_minutes = abs((timestamp - event_time).total_seconds() / 60)

            if diff_minutes <= 15:
                return True

        return False
```

### 6.3 Inference Loop

```python
class SignalEngine:
    """
    Main event loop. Runs every M1 candle close.
    """
    def __init__(self):
        self.signal_gate = SignalGate()
        self.m1_model = None  # Loaded from disk
        self.m5_model = None
        self.m1_calibrator = None
        self.m5_calibrator = None
        self.feature_computer = FeatureComputer()
        self.signal_log = []

    def run(self):
        """Main loop — call every M1 candle close."""
        import time

        while True:
            try:
                # Wait for next M1 candle close
                self._wait_for_candle_close()

                # 1. Fetch data
                market_data = self._fetch_all_data()

                # 2. Compute M1 features
                m1_features = self.feature_computer.compute_m1(market_data['m1_data'])
                m1_features_array = self._prepare_features(m1_features)

                # 3. Compute M5 features
                m5_features = self.feature_computer.compute_m5(market_data['m5_data'])
                m5_features_array = self._prepare_features(m5_features)

                # 4. Model inference
                m1_raw_probs = self.m1_model.predict_proba(m1_features_array)
                m1_probs = self.m1_calibrator.calibrate(m1_raw_probs)

                m5_raw_probs = self.m5_model.predict_proba(m5_features_array)
                m5_probs = self.m5_calibrator.calibrate(m5_raw_probs)

                m1_prediction = self._decode_prediction(m1_probs)
                m5_prediction = self._decode_prediction(m5_probs)

                # 5. Signal gate
                signal, confidence, reason = self.signal_gate.evaluate(
                    market_data, m5_prediction, m1_prediction
                )

                # 6. Log and notify
                self._log_signal(signal, confidence, reason, market_data)

                if signal != 'NO_TRADE':
                    self._send_notification(signal, confidence, market_data)

            except Exception as e:
                self._handle_error(e)

    def _decode_prediction(self, calibrated_probs):
        """Convert calibrated probabilities to direction + confidence."""
        class_names = ['LONG', 'SHORT', 'NO_TRADE']
        predicted_class = np.argmax(calibrated_probs)
        confidence = calibrated_probs[0, predicted_class]

        return {
            'direction': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'LONG': calibrated_probs[0, 0],
                'SHORT': calibrated_probs[0, 1],
                'NO_TRADE': calibrated_probs[0, 2],
            }
        }

    def _wait_for_candle_close(self):
        """Sleep until the next M1 candle closes."""
        import time
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        seconds_until_next_minute = 60 - now.second
        time.sleep(seconds_until_next_minute + 1)  # +1s buffer
```

---

## 7. Risk Management

### 7.1 Position Sizing (Modified Kelly)

```python
class RiskManager:
    """
    Dynamic position sizing based on account equity and signal confidence.
    """
    def __init__(self, account_balance, risk_per_trade_pct=0.5):
        self.initial_balance = account_balance
        self.risk_per_trade = risk_per_trade_pct / 100

    def calculate_lot_size(self, signal, confidence, sl_pips=6):
        """
        lot_size = (account_equity * risk_per_trade) / (sl_pips * pip_value)
        """
        account_equity = self._get_account_equity()
        risk_amount = account_equity * self.risk_per_trade

        # Confidence scaling: reduce size for lower confidence
        confidence_multiplier = min(1.0, confidence / 0.90)

        # Pip value for 0.01 lot = $0.10 per pip
        pip_value = 0.10  # Per 0.01 lot
        sl_pip_value = sl_pips * pip_value

        lot_size = (risk_amount * confidence_multiplier) / sl_pip_value

        # Round to broker's minimum lot step (usually 0.01)
        lot_size = round(lot_size, 2)

        # Cap at maximum
        max_lot = account_equity * 0.02 / sl_pip_value  # 2% max
        lot_size = min(lot_size, max_lot)

        return max(0.01, lot_size)

    def _get_account_equity(self):
        account = mt5.account_info()
        return account.equity if account else self.initial_balance
```

### 7.2 Daily Loss Limit

```python
class DailyLossLimiter:
    """
    Hard stop: if daily loss exceeds 3% of starting equity, halt all trading.
    """
    def __init__(self, daily_loss_limit_pct=3.0):
        self.daily_limit = daily_loss_limit_pct / 100
        self.day_start_equity = None
        self.day_start_date = None

    def check(self):
        today = datetime.utcnow().date()
        account = mt5.account_info()

        if self.day_start_date != today:
            self.day_start_date = today
            self.day_start_equity = account.equity

        current_equity = account.equity
        daily_pnl_pct = (current_equity - self.day_start_equity) / self.day_start_equity

        if daily_pnl_pct < -self.daily_limit:
            return False  # HALT — daily loss limit hit
        return True
```

---

## 8. Monitoring & Health

### 8.1 Metrics to Track

```python
class MetricsTracker:
    """Track live performance metrics."""
    def __init__(self):
        self.signals = []
        self.trades = []

    def log_signal(self, signal, confidence, reason, timestamp):
        self.signals.append({
            'timestamp': timestamp,
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
        })

    def log_trade(self, signal, entry_price, exit_price, pnl, timestamp):
        self.trades.append({
            'timestamp': timestamp,
            'signal': signal,
            'entry': entry_price,
            'exit': exit_price,
            'pnl': pnl,
        })

    def compute_live_accuracy(self, lookback_hours=24):
        """Compute accuracy over last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        recent_signals = [s for s in self.signals if s['timestamp'] > cutoff]
        # Match with trades and compute precision
        # Returns: (precision, total_signals, total_trades)
        pass

    def check_accuracy_drift(self, target_precision=0.90, window_days=7):
        """Alert if live accuracy drops below target."""
        live_accuracy = self.compute_live_accuracy(lookback_hours=24*window_days)
        if live_accuracy < target_precision * 0.90:  # 10% degradation
            return True  # Model may need retraining
        return False
```

### 8.2 Spread Tracker

```python
class SpreadTracker:
    """
    Track real-time spread. If spread consistently >0.40 pips,
    suppress signals or alert user.
    """
    def __init__(self, max_spread=0.40):
        self.spreads = []
        self.max_spread = max_spread

    def update(self, spread):
        self.spreads.append({
            'timestamp': datetime.utcnow(),
            'spread': spread,
        })

    def is_acceptable(self):
        if not self.spreads:
            return True
        recent = [s['spread'] for s in self.spreads[-20:]]  # Last 20 checks
        avg_spread = np.mean(recent)
        return avg_spread < self.max_spread
```

---

## 9. Deployment Architecture

### 9.1 Option A: Local Deployment (Recommended)

```
┌─────────────────────────────────────────────┐
│              YOUR WINDOWS PC                 │
│                                              │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │  MT5 Terminal     │  │  Python Engine    │ │
│  │  (XAUUSD chart)   │◄─┤  (signal engine)  │ │
│  │                    │  │                    │ │
│  │  - Data feed       │  │  - Feature comp    │ │
│  │  - Order execution │  │  - Model inference │ │
│  │  - Account mgmt    │  │  - Signal gate     │ │
│  └──────────────────┘  │  - Risk management  │ │
│                          └──────────┬─────────┘ │
│                                     │           │
│                          ┌──────────▼─────────┐ │
│                          │  Notifications      │ │
│                          │  - Telegram bot     │ │
│                          │  - Email alerts     │ │
│                          │  - Desktop popup    │ │
│                          └────────────────────┘ │
└─────────────────────────────────────────────┘
```

**Pros**: Zero latency, no cloud cost, full control
**Cons**: Requires PC running 24/5, no remote access

### 9.2 Option B: VPS Deployment

```
┌──────────────────────────────────────────────────┐
│              VPS (Windows, 2vCPU, 4GB RAM)       │
│                                                   │
│  MT5 Terminal + Python Engine                     │
│  (same as local, but on remote server)           │
│                                                   │
│  Remote access: RDP / SSH tunnel                  │
│  Notifications: Telegram (always online)          │
└──────────────────────┬────────────────────────────┘
                       │
                       │ Internet
                       │
┌──────────────────────▼────────────────────────────┐
│              YOUR LOCAL PC                         │
│  - Telegram app (receive signals)                 │
│  - Optional: dashboard web app                    │
└───────────────────────────────────────────────────┘
```

**Pros**: Runs 24/5, low latency, ~$10-20/month
**Cons**: Windows VPS required for MT5, monthly cost

### 9.3 File Structure

```
xauusd-scalping-engine/
├── src/
│   ├── __init__.py
│   ├── config.py                 # All constants, thresholds, params
│   ├── mt5_connection.py         # MT5 connection, heartbeat, data fetch
│   ├── data_pipeline.py          # Historical data pull, parquet storage
│   ├── feature_engineering.py    # All feature computation functions
│   ├── labels.py                 # Triple-barrier labeling
│   ├── model_trainer.py          # Walk-forward training, calibration
│   ├── model_inference.py        # Load model, predict, calibrate
│   ├── multi_timeframe.py        # H1, M15 filters, agreement gate
│   ├── signal_gate.py            # Complete signal gate pipeline
│   ├── risk_manager.py           # Position sizing, daily loss limit
│   ├── news_filter.py            # Economic calendar integration
│   ├── execution.py              # MT5 order send, position management
│   ├── notifications.py          # Telegram, email, desktop alerts
│   ├── metrics.py                # Live accuracy, spread tracking
│   └── main.py                   # Main event loop
├── models/
│   ├── m1_xgboost.joblib         # Trained M1 model
│   ├── m5_xgboost.joblib         # Trained M5 model
│   ├── m1_calibrator.joblib      # M1 isotonic calibrator
│   └── m5_calibrator.joblib      # M5 isotonic calibrator
├── data/
│   ├── xauusd/                   # Parquet files by timeframe
│   ├── dxy/                      # DXY data
│   ├── us10y/                    # Yield data
│   └── ticks/                    # Tick data for CVD
├── config/
│   ├── config.yaml               # Main configuration
│   └── params.yaml               # Model hyperparameters
├── tests/
│   ├── test_features.py
│   ├── test_labels.py
│   ├── test_signal_gate.py
│   └── test_model.py
├── scripts/
│   ├── pull_historical_data.py   # One-time data pull
│   ├── train_models.py           # Model training
│   ├── backtest.py               # Full backtest
│   └── live_trading.py           # Live inference loop
├── requirements.txt
├── .env                          # MT5 credentials (never commit)
└── README.md
```

---

## 10. Configuration

### 10.1 config.yaml

```yaml
# Trading Parameters
symbol: "XAUUSD"
pip_value: 0.01  # 1 pip = 0.01 price change

# Labeling
labeling:
  tp_pips: 10          # Take profit in pips
  sl_pips: 6           # Stop loss in pips
  horizon_m1: 10       # Look-forward candles on M1
  horizon_m5: 6        # Look-forward candles on M5

# Model
model:
  n_estimators: 500
  max_depth: 4
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.7

# Confidence Thresholds
thresholds:
  m1_confidence: 0.85  # M1 must be >85% confident
  m5_confidence: 0.75  # M5 must be >75% confident

# Session Filter
sessions:
  allowed_start_utc: 8   # London open
  allowed_end_utc: 17    # NY close
  blocked_hours_utc: []  # Additional blocked hours

# Volatility Filter
volatility:
  min_atr_ratio: 0.5
  max_atr_ratio: 2.5
  atr_period: 14

# Spread Filter
spread:
  max_spread_pips: 0.30

# News Filter
news:
  block_minutes_before: 15
  block_minutes_after: 15
  impacts: ["High"]
  currencies: ["USD", "XAU"]

# Risk Management
risk:
  risk_per_trade_pct: 0.5
  max_daily_loss_pct: 3.0
  max_lot_size: 1.0

# MT5 Connection
mt5:
  heartbeat_interval_sec: 5
  max_heartbeat_failures: 3
  reconnect_delay_sec: 10
```

---

## 11. Training Workflow

### Step 1: Pull Historical Data (one-time)
```bash
python scripts/pull_historical_data.py --symbol XAUUSD --years 3
```

### Step 2: Pull DXY + Yield Data (one-time)
```bash
python scripts/pull_macro_data.py --start 2023-01-01
```

### Step 3: Compute Features
```bash
python scripts/compute_features.py --timeframes M1 M5 M15 H1
```

### Step 4: Generate Labels
```bash
python scripts/generate_labels.py --tp 10 --sl 6 --horizon 10
```

### Step 5: Train Models
```bash
python scripts/train_models.py --model xgboost --timeframe M1
python scripts/train_models.py --model xgboost --timeframe M5
```

### Step 6: Validate (Walk-Forward)
```bash
python scripts/validate.py --method walk_forward --purge 5 --embargo 10
```

### Step 7: Backtest
```bash
python scripts/backtest.py --spread 0.20 --slippage 1.0 --delay 150ms
```

### Step 8: Go Live
```bash
python scripts/live_trading.py --mode paper  # Paper trading first
python scripts/live_trading.py --mode live   # After 2 weeks paper
```

---

## 12. Expected Performance Summary

| Metric | Target | Notes |
|--------|--------|-------|
| Signal frequency | 2-8 per hour | During London/NY overlap only |
| Daily signals | 15-40 | Highly selective |
| Long/Short precision | >0.90 | On selected signals only |
| Combined recall | >0.15 | Signal on 15%+ of eligible candles |
| Sharpe Ratio | >1.5 | After realistic costs |
| Max Drawdown | <15% | Per walk-forward test fold |
| Train-Test Gap | <5% | Overfitting guardrail |
| Inference latency | <100ms | Feature compute + model predict |
| Total pipeline latency | <500ms | Per M1 candle close |

---

*This architecture is designed for implementation. Every component has specified inputs, outputs, and logic. Ready to begin coding in Phase 2.*

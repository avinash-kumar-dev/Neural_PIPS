# PHASE 1: ARCHITECTURE DESIGN — EUR/USD 3:1 TP:SL Scalping Signal Engine

*Comprehensive system design for EUR/USD signal generation with minimum 3:1 TP:SL ratio.*

---

## 1. System Overview

### 1.1 What We're Building

A real-time EUR/USD scalping signal **generator** (no execution) that:
- Analyzes 4 timeframes simultaneously (H4, H1, M15, M5)
- Generates ternary signals: LONG / SHORT / NO-TRADE
- Enforces strict minimum 3:1 TP:SL ratio (SL=5 pips, TP=15 pips)
- Trades only during London-NY overlap (13:00-16:00 UTC)
- Uses ensemble ML (XGBoost + LightGBM + CatBoost) with rule-based filters
- Sends alerts via Telegram/Email — **no MT5 order execution**

### 1.2 Core Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Symbol | EUR/USD only | Single dominant driver (DXY), tightest spread |
| TP | 15 pips | Fits within single M15 candle during overlap |
| SL | 5 pips | Tight stop, 2% spread impact of TP |
| Net R:R | 2.94:1 | After 0.2 pip spread round-trip |
| Break-even WR | 25% | Mathematical floor |
| Target WR | >35% | +0.40R expectancy per trade |
| Session | 13:00-16:00 UTC only | 20-30+ ATR/h, best liquidity |
| Max signals/day | 1-3 | Extreme selectivity for 3:1 |
| Confidence threshold | >= 70/100 | 8-factor weighted score |
| H4 filter | HARD (never trade against) | Reduces signals 50-65%, improves quality |

### 1.3 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MT5 TERMINAL (Windows)                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌────────────────────┐  │
│  │  EURUSD   │  │   DXY     │  │  US10Y    │  │  Economic Calendar │  │
│  │  M5/M15/  │  │  (yfinance│  │  (FRED    │  │  (ForexFactory     │  │
│  │  H1/H4    │  │  fallback)│  │  API)     │  │   free JSON)       │  │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └────────┬───────────┘  │
│        │               │               │                  │              │
│        └───────────────┴───────────────┴──────────────────┘              │
│                              │                                           │
│                       MT5 Python API                                    │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     PYTHON SIGNAL ENGINE (local)                         │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  1. DATA LAYER                                                    │  │
│  │  MT5 Fetch -> DataFrame -> Parquet Storage -> PIT Alignment       │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                               │                                          │
│  ┌───────────────────────────▼───────────────────────────────────────┐  │
│  │  2. FEATURE LAYER                                                 │  │
│  │  Alpha158-style Factors -> datasieve Pipeline                     │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                               │                                          │
│  ┌───────────────────────────▼───────────────────────────────────────┐  │
│  │  3. RULE-BASED FILTERS                                            │  │
│  │  Session -> Spread -> Volatility -> News -> H4 -> H1 -> M15 -> ADX│  │
│  │  (Eliminates ~90% of bars)                                        │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                               │                                          │
│  ┌───────────────────────────▼───────────────────────────────────────┐  │
│  │  4. ML LAYER                                                      │  │
│  │  XGBoost + LightGBM + CatBoost -> Meta-Learner -> Confidence Gate │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                               │                                          │
│  ┌───────────────────────────▼───────────────────────────────────────┐  │
│  │  5. SIGNAL LAYER                                                  │  │
│  │  8-Factor Score -> Threshold Gate -> LONG/SHORT/NO-TRADE          │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                               │                                          │
│  ┌───────────────────────────▼───────────────────────────────────────┐  │
│  │  6. NOTIFICATION LAYER                                            │  │
│  │  Telegram Bot + Email Alert + Signal Log (CSV)                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  7. BACKTEST LAYER                                                │  │
│  │  VectorBT (speed) + Backtrader (realism)                          │  │
│  │  Walk-forward (30 windows) + Purged CV + Turbulence Risk Mgmt     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  8. MONITORING LAYER                                              │  │
│  │  Signal accuracy + Spread monitor + Daily P&L + Retraining trigger│  │
│  └───────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Event-Driven Flow

Every M5 candle close triggers the pipeline (M5 is the entry timeframe for 3:1):

```
M5 Candle Close Event
    |
    +-- 1. Check session gate (is it 13:00-16:00 UTC?)
    +-- 2. Fetch latest OHLCV: M5, M15, H1, H4
    +-- 3. Fetch DXY data (yfinance or MT5)
    +-- 4. Fetch US10Y data (FRED API, daily interpolated)
    +-- 5. Fetch economic calendar (ForexFactory, cache 1hr)
    +-- 6. Compute features (~50 factors, Alpha158-style)
    +-- 7. Run datasieve Pipeline (scaling, threshold)
    +-- 8. Run rule-based filters (session -> spread -> vol -> news -> H4 -> H1 -> M15 -> ADX)
    +-- 9. If rules pass: run ML ensemble (XGB + LGB + CatBoost)
    +-- 10. Meta-learner combines predictions
    +-- 11. 8-factor confidence scoring
    +-- 12. Threshold gate (confidence >= 70)
    +-- 13. Emit signal (LONG/SHORT/NO-TRADE) + TP/SL levels
    +-- 14. Send Telegram/Email notification
    +-- 15. Log to CSV + update metrics tracker
```

---

## 2. Data Pipeline

### 2.1 Data Sources

| Source | Data | Frequency | Cost | Fallback |
|--------|------|-----------|------|----------|
| MT5 | EUR/USD OHLCV (M5/M15/H1/H4) | Real-time | Free (broker) | -- |
| MT5 | EUR/USD spread, tick volume | Real-time | Free (broker) | -- |
| yfinance | DXY (DX-Y.NYB) | Daily | Free | MT5 DXY symbol |
| FRED API | DGS10 (10yr Treasury), T10YIE (breakeven) | Daily | Free (unlimited) | -- |
| ForexFactory | Economic calendar (USD/EUR high-impact) | Cached 1hr | Free JSON | -- |
| NewsAPI | News sentiment (optional) | 100 req/day | Free tier | Skip |

### 2.2 MT5 Connection Layer

```python
import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

class MT5Connection:
    def __init__(self):
        self.connected = False

    def connect(self):
        if not mt5.initialize():
            raise ConnectionError(f"MT5 init failed: {mt5.last_error()}")
        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')
        if not mt5.login(login, password, server):
            raise ConnectionError(f"MT5 login failed: {mt5.last_error()}")
        self.connected = True
        return True

    def heartbeat(self):
        """Call every 5 seconds. Returns False if connection dead."""
        info = mt5.account_info()
        if info is None:
            self.connected = False
            return False
        return True

    def shutdown(self):
        mt5.shutdown()
        self.connected = False
```

### 2.3 Data Fetching

```python
class DataFetcher:
    SYMBOL = "EURUSD"
    TIMEFRAMES = {
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
    }

    def fetch_ohlcv(self, timeframe, count):
        rates = mt5.copy_rates_from_pos(
            self.SYMBOL, self.TIMEFRAMES[timeframe], 0, count
        )
        if rates is None or len(rates) == 0:
            raise ValueError(f"Failed to fetch {timeframe}: {mt5.last_error()}")
        return pd.DataFrame(rates)

    def fetch_all(self):
        return {
            'M5': self.fetch_ohlcv('M5', 100),
            'M15': self.fetch_ohlcv('M15', 50),
            'H1': self.fetch_ohlcv('H1', 100),
            'H4': self.fetch_ohlcv('H4', 50),
        }

    def fetch_spread(self):
        tick = mt5.symbol_info_tick(self.SYMBOL)
        if tick is None:
            return None
        symbol_info = mt5.symbol_info(self.SYMBOL)
        spread_points = tick.ask - tick.bid
        spread_pips = spread_points / symbol_info.point * 0.1
        return spread_pips

    def fetch_dxy(self):
        import yfinance as yf
        dxy = yf.download("DX-Y.NYB", period="60d", interval="1d", progress=False)
        return dxy

    def fetch_us10y(self):
        import requests
        api_key = os.getenv('FRED_API_KEY')
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={api_key}&file_type=json&sort_order=desc&limit=60"
        resp = requests.get(url)
        data = resp.json()['observations']
        df = pd.DataFrame(data)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df
```

### 2.4 Historical Data Storage

```
data/
+-- eurusd/
|   +-- M5/
|   |   +-- 2024-01.parquet
|   |   +-- 2024-02.parquet
|   |   +-- ... (5 years = 60 files per TF)
|   +-- M15/
|   +-- H1/
|   +-- H4/
+-- dxy/
|   +-- daily/
|       +-- 2024-2026.parquet
+-- us10y/
|   +-- daily/
|       +-- 2024-2026.parquet
+-- features/
|   +-- m5_features/
|   +-- m15_features/
+-- labels/
|   +-- triple_barrier_3to1/
+-- signals/
    +-- 2026-06.csv
    +-- ...
```

### 2.5 Parquet Storage Helper

```python
import pandas as pd
import os
from datetime import datetime

class ParquetStorage:
    BASE_DIR = "data"

    def save(self, df, asset, timeframe, date=None):
        if date is None:
            date = datetime.utcnow()
        dir_path = os.path.join(self.BASE_DIR, asset, timeframe)
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{date.strftime('%Y-%m')}.parquet"
        filepath = os.path.join(dir_path, filename)
        if os.path.exists(filepath):
            existing = pd.read_parquet(filepath)
            df = pd.concat([existing, df]).drop_duplicates(
                subset=['time'], keep='last'
            ).sort_values('time').reset_index(drop=True)
        df.to_parquet(filepath, index=False)

    def load(self, asset, timeframe, start_date, end_date):
        all_data = []
        dir_path = os.path.join(self.BASE_DIR, asset, timeframe)
        if not os.path.exists(dir_path):
            return pd.DataFrame()
        for f in sorted(os.listdir(dir_path)):
            if f.endswith('.parquet'):
                month_str = f.replace('.parquet', '')
                month_date = datetime.strptime(month_str, '%Y-%m')
                if start_date <= month_date <= end_date:
                    df = pd.read_parquet(os.path.join(dir_path, f))
                    all_data.append(df)
        if all_data:
            return pd.concat(all_data).drop_duplicates(
                subset=['time'], keep='last'
            ).sort_values('time').reset_index(drop=True)
        return pd.DataFrame()
```

### 2.6 Data Pipeline Schedule

| Task | Frequency | Latency Budget |
|------|-----------|----------------|
| Historical backfill | One-time (5 years) | N/A (background) |
| Daily data pull | Daily at 00:05 UTC | <5 seconds |
| M5 OHLCV fetch | Every M5 candle close | <200ms |
| M15/H1/H4 fetch | Every M5 candle close | <100ms (cached) |
| DXY fetch | Daily | <2 seconds |
| US10Y fetch | Daily | <2 seconds |
| Feature computation | After data fetch | <300ms |
| Model inference | After features | <100ms |
| **Total pipeline** | **Per M5 candle** | **<1 second** |

---

## 3. Feature Engineering Module

### 3.1 Design Principles

- **Alpha158-style**: Compute 50+ factors covering price, volume, momentum, volatility, structure
- **datasieve Pipeline**: VarianceThreshold -> MinMaxScaler -> optional PCA
- **PIT (Point-in-Time)**: No future data leakage -- all features use `shift(1)` or completed bars only

### 3.2 Complete Feature Set (~50 features)

#### Tier 1: Price & Returns (16 features)

```python
class PriceFeatures:
    def compute(self, df):
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        features = {}

        # Returns at multiple lags
        for lag in [1, 2, 3, 5, 10, 20]:
            features[f'return_{lag}'] = close.pct_change(lag)

        # Log returns
        for lag in [1, 3, 5]:
            features[f'log_return_{lag}'] = np.log(close / close.shift(lag))

        # OHLC ratios
        features['open_close_ratio'] = (close - open_) / open_
        features['high_low_ratio'] = (high - low) / close
        features['close_high_ratio'] = (high - close) / close
        features['close_low_ratio'] = (close - low) / close

        # VWAP distance
        typical_price = (high + low + close) / 3
        volume = df['tick_volume']
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        atr = self._atr(df, 14)
        features['vwap_distance'] = (close - vwap) / atr

        # Session range position
        range_20 = high.rolling(20).max() - low.rolling(20).min()
        features['price_position_20'] = (close - low.rolling(20).min()) / range_20

        return pd.DataFrame(features, index=df.index)

    def _atr(self, df, period):
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
```

#### Tier 2: Technical Indicators (22 features)

```python
class TechnicalFeatures:
    def compute(self, df):
        close = df['close']
        features = {}

        # RSI (multiple periods)
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # EMA slopes
        for period in [8, 21, 50]:
            ema = close.ewm(span=period).mean()
            features[f'ema_{period}_slope'] = ema.pct_change(3)
            features[f'ema_{period}_above'] = (close > ema).astype(int)

        # MACD (5, 35, 5)
        ema_fast = close.ewm(span=5).mean()
        ema_slow = close.ewm(span=35).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=5).mean()
        features['macd_histogram'] = macd_line - signal_line
        features['macd_cross'] = np.where(
            (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)), 1,
            np.where(
                (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1)), -1, 0
            )
        )

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        features['bb_position'] = (close - lower) / (upper - lower)
        features['bb_width'] = (upper - lower) / sma20

        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - close.shift(1)),
            abs(df['low'] - close.shift(1))
        ], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_50'] = tr.rolling(50).mean()
        features['atr_ratio'] = features['atr_14'] / features['atr_50']

        # ADX (period=10)
        adx_data = self._compute_adx(df, period=10)
        features['adx'] = adx_data['adx']
        features['plus_di'] = adx_data['plus_di']
        features['minus_di'] = adx_data['minus_di']
        features['adx_rising'] = (features['adx'] > features['adx'].shift(1)).astype(int)

        # Stochastic RSI
        rsi = features.get('rsi_14', self._rsi(close, 14))
        stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        features['stoch_rsi_k'] = stoch_rsi.rolling(3).mean()
        features['stoch_rsi_d'] = features['stoch_rsi_k'].rolling(3).mean()

        # Volume
        vol_sma20 = df['tick_volume'].rolling(20).mean()
        features['volume_ratio'] = df['tick_volume'] / vol_sma20

        # Candle body ratio
        body = abs(close - df['open'])
        total_range = df['high'] - df['low']
        features['body_ratio'] = body / total_range

        # CVD approximation
        direction = np.where(close >= close.shift(1), 1, -1)
        delta_volume = df['tick_volume'] * direction
        features['cvd_delta'] = delta_volume
        features['cvd_roc_5'] = pd.Series(delta_volume).rolling(5).sum().pct_change(5)

        return pd.DataFrame(features, index=df.index)

    def _compute_adx(self, df, period=10):
        high, low, close = df['high'], df['low'], df['close']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}
```

#### Tier 3: Multi-Timeframe Features (9 features)

```python
class MultiTimeframeFeatures:
    def compute(self, m5_df, m15_df, h1_df, h4_df):
        features = {}

        # H4 trend direction
        h4_ema9 = h4_df['close'].ewm(span=9).mean()
        h4_ema21 = h4_df['close'].ewm(span=21).mean()
        h4_ema200 = h4_df['close'].ewm(span=200).mean()
        h4_price = h4_df['close'].iloc[-1]
        features['h4_bullish'] = int(h4_price > h4_ema200.iloc[-1] and h4_ema9.iloc[-1] > h4_ema21.iloc[-1])
        features['h4_bearish'] = int(h4_price < h4_ema200.iloc[-1] and h4_ema9.iloc[-1] < h4_ema21.iloc[-1])

        # H1 structure
        h1_ema50 = h1_df['close'].ewm(span=50).mean()
        features['h1_above_ema50'] = int(h1_df['close'].iloc[-1] > h1_ema50.iloc[-1])

        # M15 momentum
        m15_rsi = self._rsi(m15_df['close'], 14)
        features['m15_rsi'] = m15_rsi.iloc[-1]
        features['m15_rsi_bullish'] = int(40 <= m15_rsi.iloc[-1] <= 55)
        features['m15_rsi_bearish'] = int(45 <= m15_rsi.iloc[-1] <= 60)

        # MTF alignment score (0-3)
        alignment = 0
        if features['h4_bullish'] or features['h4_bearish']:
            alignment += 1
        if features['h1_above_ema50']:
            alignment += 1
        if features['m15_rsi_bullish'] or features['m15_rsi_bearish']:
            alignment += 1
        features['mtf_alignment'] = alignment

        # BOS on H1
        features['h1_bullish_bos'] = self._detect_bos(h1_df, 'bullish')
        features['h1_bearish_bos'] = self._detect_bos(h1_df, 'bearish')

        return pd.DataFrame(features, index=[0])

    def _detect_bos(self, df, direction, lookback=10):
        if len(df) < lookback + 2:
            return 0
        if direction == 'bullish':
            swing_high = df['high'].iloc[-lookback-2:-2].max()
            return int(df['close'].iloc[-1] > swing_high and df['close'].iloc[-2] <= swing_high)
        else:
            swing_low = df['low'].iloc[-lookback-2:-2].min()
            return int(df['close'].iloc[-1] < swing_low and df['close'].iloc[-2] >= swing_low)
```

#### Tier 4: Session & Macro Features (12 features)

```python
class SessionMacroFeatures:
    def compute(self, timestamp_utc, dxy_data=None, us10y_data=None):
        features = {}
        minute_of_day = timestamp_utc.hour * 60 + timestamp_utc.minute
        features['session_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
        features['session_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)
        features['is_overlap'] = int(13 <= timestamp_utc.hour < 16)
        features['is_london'] = int(8 <= timestamp_utc.hour < 17)
        features['is_ny'] = int(13 <= timestamp_utc.hour < 21)

        dow = timestamp_utc.weekday()
        for i in range(5):
            features[f'dow_{i}'] = int(dow == i)

        if dxy_data is not None and len(dxy_data) > 5:
            dxy_close = dxy_data['Close']
            features['dxy_return_1d'] = dxy_close.pct_change().iloc[-1]
            features['dxy_return_3d'] = dxy_close.pct_change(3).iloc[-1]
            features['dxy_above_ema20'] = int(dxy_close.iloc[-1] > dxy_close.ewm(span=20).mean().iloc[-1])
        else:
            features['dxy_return_1d'] = 0
            features['dxy_return_3d'] = 0
            features['dxy_above_ema20'] = 0

        if us10y_data is not None and len(us10y_data) > 2:
            features['us10y_latest'] = us10y_data['value'].iloc[-1]
            features['us10y_change_1d'] = us10y_data['value'].diff().iloc[-1]
        else:
            features['us10y_latest'] = 0
            features['us10y_change_1d'] = 0

        return pd.DataFrame(features, index=[0])
```

### 3.3 Feature Pipeline (datasieve-style)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

class FeaturePipeline:
    def __init__(self):
        self.pipeline = Pipeline([
            ('variance', VarianceThreshold(threshold=0.001)),
            ('scaler', MinMaxScaler()),
        ])
        self.fitted = False

    def fit_transform(self, X):
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        result = self.pipeline.fit_transform(X_clean)
        self.fitted = True
        return result

    def transform(self, X):
        if not self.fitted:
            raise ValueError("Pipeline not fitted.")
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        return self.pipeline.transform(X_clean)
```

### 3.4 Complete Feature Vector

```python
FEATURE_VECTOR = [
    # Tier 1: Price & Returns (16 features)
    'return_1', 'return_2', 'return_3', 'return_5', 'return_10', 'return_20',
    'log_return_1', 'log_return_3', 'log_return_5',
    'open_close_ratio', 'high_low_ratio', 'close_high_ratio', 'close_low_ratio',
    'vwap_distance', 'price_position_20',

    # Tier 2: Technical Indicators (22 features)
    'rsi_7', 'rsi_14', 'rsi_21',
    'ema_8_slope', 'ema_8_above', 'ema_21_slope', 'ema_21_above',
    'ema_50_slope', 'ema_50_above',
    'macd_histogram', 'macd_cross',
    'bb_position', 'bb_width',
    'atr_14', 'atr_50', 'atr_ratio',
    'adx', 'plus_di', 'minus_di', 'adx_rising',
    'stoch_rsi_k', 'stoch_rsi_d',
    'volume_ratio', 'body_ratio', 'cvd_delta', 'cvd_roc_5',

    # Tier 3: Multi-Timeframe (9 features)
    'h4_bullish', 'h4_bearish', 'h1_above_ema50',
    'm15_rsi', 'm15_rsi_bullish', 'm15_rsi_bearish',
    'mtf_alignment', 'h1_bullish_bos', 'h1_bearish_bos',

    # Tier 4: Session & Macro (12 features)
    'session_sin', 'session_cos',
    'is_overlap', 'is_london', 'is_ny',
    'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4',
    'dxy_return_1d', 'dxy_return_3d', 'dxy_above_ema20',
    'us10y_latest', 'us10y_change_1d',
]
```

---

## 4. Rule-Based Filter Layer

### 4.1 Design Philosophy

The rule-based layer eliminates ~90% of bars before ML ever sees them. This is critical for 3:1 TP:SL because fewer false signals means higher precision on what remains, rules are interpretable and easier to debug, and rules are fast with no ML inference cost for rejected bars.

### 4.2 Complete Filter Pipeline

```python
class RuleBasedFilters:
    def __init__(self, config):
        self.config = config
        self.news_cache = None
        self.news_cache_time = None

    def evaluate(self, market_data, features):
        """Returns: (passed: bool, reason: str or None)"""
        gates = [
            ('session', self._session_gate(market_data['timestamp'])),
            ('spread', self._spread_gate(market_data['spread'])),
            ('volatility', self._volatility_gate(features)),
            ('news', self._news_gate(market_data['timestamp'])),
            ('h4', self._h4_gate(features)),
            ('h1', self._h1_gate(features)),
            ('m15', self._m15_gate(features)),
            ('adx', self._adx_gate(features)),
        ]
        for name, (passed, reason) in gates:
            if not passed:
                return False, reason
        return True, None

    def _session_gate(self, timestamp):
        hour = timestamp.hour
        if not (13 <= hour < 16):
            return False, 'OUTSIDE_SESSION'
        return True, None

    def _spread_gate(self, spread_pips):
        if spread_pips is None or spread_pips > 0.5:
            return False, 'SPREAD_TOO_HIGH'
        return True, None

    def _volatility_gate(self, features):
        atr_ratio = features.get('atr_ratio', 1.0)
        if atr_ratio < 0.5 or atr_ratio > 2.5:
            return False, 'VOLATILITY_OUT_OF_RANGE'
        return True, None

    def _news_gate(self, timestamp):
        import requests
        if self.news_cache is None or \
           (datetime.utcnow() - self.news_cache_time).seconds > 3600:
            try:
                url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
                resp = requests.get(url, timeout=5)
                self.news_cache = resp.json()
                self.news_cache_time = datetime.utcnow()
            except:
                return True, None
        for event in self.news_cache:
            if event.get('impact') != 'High':
                continue
            if event.get('currency') not in ('USD', 'EUR'):
                continue
            event_time = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
            diff_min = abs((timestamp - event_time).total_seconds() / 60)
            if diff_min <= 15:
                return False, 'NEWS_EVENT'
        return True, None

    def _h4_gate(self, features):
        """H4 hard filter: NEVER trade against H4 trend."""
        h4_bullish = features.get('h4_bullish', 0)
        h4_bearish = features.get('h4_bearish', 0)
        if not h4_bullish and not h4_bearish:
            return False, 'H4_RANGING'
        return True, None

    def _h1_gate(self, features):
        mtf = features.get('mtf_alignment', 0)
        if mtf < 2:
            return False, 'H1_NOT_ALIGNED'
        return True, None

    def _m15_gate(self, features):
        m15_rsi = features.get('m15_rsi', 50)
        h4_bullish = features.get('h4_bullish', 0)
        h4_bearish = features.get('h4_bearish', 0)
        if h4_bullish and m15_rsi > 65:
            return False, 'M15_OVERBOUGHT_LONG'
        if h4_bearish and m15_rsi < 35:
            return False, 'M15_OVERSOLD_SHORT'
        return True, None

    def _adx_gate(self, features):
        adx = features.get('adx', 0)
        adx_rising = features.get('adx_rising', 0)
        if adx < 20:
            return False, 'ADX_TOO_LOW'
        if not adx_rising:
            return False, 'ADX_NOT_RISING'
        return True, None
```

### 4.3 Filter Sequence

```
Input: M5 candle close + features
  |
  +-- Gate 1: Session (13:00-16:00 UTC?) -------> NO --> OUTSIDE_SESSION
  |
  +-- Gate 2: Spread (< 0.5 pips?) -------------> NO --> SPREAD_TOO_HIGH
  |
  +-- Gate 3: ATR ratio (0.5-2.5?) -------------> NO --> VOLATILITY_OUT_OF_RANGE
  |
  +-- Gate 4: News (+/-15 min?) ----------------> YES --> NEWS_EVENT
  |
  +-- Gate 5: H4 trend (bullish/bearish?) ------> NO --> H4_RANGING
  |
  +-- Gate 6: H1 alignment (mtf >= 2?) ---------> NO --> H1_NOT_ALIGNED
  |
  +-- Gate 7: M15 momentum (RSI zone?) ---------> NO --> M15_OVERBOUGHT/OVERSOLD
  |
  +-- Gate 8: ADX (> 20 + rising?) -------------> NO --> ADX_TOO_LOW / ADX_NOT_RISING
  |
  +-- ALL PASSED --> Proceed to ML Layer
```

---

## 5. ML Layer

### 5.1 Ensemble Architecture

```
                    +------------------+
                    |   Rule-Based     |
                    |    Filters       |
                    |   (8 gates)      |
                    +--------+---------+
                             | PASSED
                             v
                +---------------------------+
                |    Feature Pipeline       |
                |  (MinMax + Variance)      |
                +------------+--------------+
                             |
                +------------+------------+
                |            |            |
                v            v            v
          +----------+ +----------+ +----------+
          | XGBoost  | | LightGBM | | CatBoost |
          | (300,d=3)| | (150,d=4)| | (200,d=4)|
          +----+-----+ +----+-----+ +----+-----+
               |            |            |
               +------------+------------+
                            |
                            v
                   +------------------+
                   |   Meta-Learner   |
                   | (Ridge Regress.) |
                   +--------+---------+
                            |
                            v
                   +------------------+
                   | Confidence Gate  |
                   |  (threshold=70)  |
                   +--------+---------+
                            |
                            v
                   LONG / SHORT / NO-TRADE
```

### 5.2 Model Configurations

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import Ridge

xgb_params = {
    'objective': 'binary:logistic',
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'scale_pos_weight': 1.0,  # Computed dynamically
    'max_delta_step': 1,
    'eval_metric': 'logloss',
    'early_stopping_rounds': 20,
    'random_state': 42,
}

lgbm_params = {
    'objective': 'binary',
    'n_estimators': 150,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbose': -1,
    'is_unbalance': True,
}

catboost_params = {
    'iterations': 200,
    'depth': 4,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'random_seed': 42,
    'verbose': 0,
    'auto_class_weights': 'Balanced',
}

meta_model = Ridge(alpha=1.0)
```

### 5.3 Triple-Barrier Labeling for 3:1

```python
class TripleBarrierLabeler:
    """
    Lopez de Prado's Triple-Barrier Method adapted for 3:1 TP:SL.
    For each M5 bar, look forward up to `horizon` bars:
    - LONG: price hits +15 pips BEFORE -5 pips --> label = 1
    - SHORT: price hits -15 pips BEFORE +5 pips --> label = -1
    - NO_TRADE: neither barrier hit --> label = 0
    """
    def __init__(self, tp_pips=15, sl_pips=5, horizon=20):
        self.tp = tp_pips * 0.0001  # EUR/USD: 1 pip = 0.0001
        self.sl = sl_pips * 0.0001
        self.horizon = horizon

    def label(self, df):
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        n = len(df)
        labels = np.full(n, 0, dtype=int)

        for i in range(n - 1):
            entry = close[i]
            for j in range(1, min(self.horizon + 1, n - i)):
                hit_tp_long = high[i + j] >= entry + self.tp
                hit_sl_long = low[i + j] <= entry - self.sl
                hit_tp_short = low[i + j] <= entry - self.tp
                hit_sl_short = high[i + j] >= entry + self.sl

                if hit_tp_long and not hit_sl_long:
                    labels[i] = 1
                    break
                if hit_tp_short and not hit_sl_short:
                    labels[i] = -1
                    break
                if hit_tp_long and hit_sl_long:
                    dist_tp = abs(high[i + j] - (entry + self.tp))
                    dist_sl = abs(low[i + j] - (entry - self.sl))
                    labels[i] = 1 if dist_tp < dist_sl else -1
                    break
                if hit_tp_short and hit_sl_short:
                    dist_tp = abs(low[i + j] - (entry - self.tp))
                    dist_sl = abs(high[i + j] - (entry + self.sl))
                    labels[i] = -1 if dist_tp < dist_sl else 1
                    break

        return labels
```

### 5.4 Walk-Forward Training

```python
class WalkForwardTrainer:
    """
    Walk-forward training with purging and embargo.
    [Training] --purge(20 bars)-- [embargo(20 bars)] -- [Test]
    30 rolling windows: 6-month train + 1-month test.
    """
    def __init__(self, n_windows=30, train_months=6, test_months=1,
                 purge_bars=20, embargo_bars=20):
        self.n_windows = n_windows
        self.train_months = train_months
        self.test_months = test_months
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def train(self, X, y, timestamps):
        results = []
        n = len(X)
        train_size = self.train_months * 21 * 12
        test_size = self.test_months * 21 * 12

        for w in range(self.n_windows):
            test_end = n - (self.n_windows - w - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start - self.embargo_bars
            train_start = max(0, train_end - train_size)

            if train_start >= train_end or test_start >= test_end:
                continue

            X_train, y_train = X[train_start:train_end], y[train_start:train_end]
            X_test, y_test = X[test_start:test_end], y[test_start:test_end]
            ts_test = timestamps[test_start:test_end]

            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            scale_pos_weight = n_neg / max(n_pos, 1)

            xgb_model = xgb.XGBClassifier(**{**xgb_params, 'scale_pos_weight': scale_pos_weight})
            xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            lgb_model = lgb.LGBMClassifier(**lgbm_params)
            lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                         callbacks=[lgb.early_stopping(20, verbose=False)])

            cat_model = CatBoostClassifier(**catboost_params)
            cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)

            xgb_oof = xgb_model.predict_proba(X_train)[:, 1]
            lgb_oof = lgb_model.predict_proba(X_train)[:, 1]
            cat_oof = cat_model.predict_proba(X_train)[:, 1]
            meta_features_train = np.column_stack([xgb_oof, lgb_oof, cat_oof])

            meta_model = Ridge(alpha=1.0)
            meta_model.fit(meta_features_train, y_train)

            xgb_test = xgb_model.predict_proba(X_test)[:, 1]
            lgb_test = lgb_model.predict_proba(X_test)[:, 1]
            cat_test = cat_model.predict_proba(X_test)[:, 1]
            meta_features_test = np.column_stack([xgb_test, lgb_test, cat_test])
            meta_probs = meta_model.predict_proba(meta_features_test)[:, 1]

            results.append({
                'window': w,
                'xgb_model': xgb_model,
                'lgb_model': lgb_model,
                'cat_model': cat_model,
                'meta_model': meta_model,
                'predictions': meta_probs,
                'labels': y_test,
                'timestamps': ts_test,
            })

        return results
```

### 5.5 Model Persistence

```python
import joblib

class ModelManager:
    MODELS_DIR = "models"

    def save(self, models_dict, window_id):
        import os
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        for name in ['xgb_model', 'lgb_model', 'cat_model', 'meta_model']:
            joblib.dump(models_dict[name],
                       os.path.join(self.MODELS_DIR, f'w{window_id}_{name.split("_")[0]}.joblib'))

    def load_latest(self):
        import os
        files = sorted([f for f in os.listdir(self.MODELS_DIR) if f.endswith('_xgb.joblib')])
        if not files:
            raise FileNotFoundError("No trained models found.")
        window_id = files[-1].split('_')[0]
        return {
            'xgb_model': joblib.load(os.path.join(self.MODELS_DIR, f'{window_id}_xgb.joblib')),
            'lgb_model': joblib.load(os.path.join(self.MODELS_DIR, f'{window_id}_lgb.joblib')),
            'cat_model': joblib.load(os.path.join(self.MODELS_DIR, f'{window_id}_cat.joblib')),
            'meta_model': joblib.load(os.path.join(self.MODELS_DIR, f'{window_id}_meta.joblib')),
        }
```

---

## 6. Confidence Scoring (8-Factor)

### 6.1 Factor Definitions

```python
class ConfidenceScorer:
    """
    8-factor weighted confidence scoring system.
    Output: 0-100 score. Threshold >= 70 for signal emission.
    """
    FACTORS = {
        'trend_alignment':  {'weight': 0.20, 'desc': 'HTF trend agrees with signal'},
        'momentum':         {'weight': 0.15, 'desc': 'RSI/MACD momentum supports entry'},
        'volatility_regime':{'weight': 0.15, 'desc': 'ATR within normal range'},
        'structure':        {'weight': 0.15, 'desc': 'Near key S/R level or BOS'},
        'pattern_quality':  {'weight': 0.10, 'desc': 'How textbook the entry looks'},
        'mtf_agreement':    {'weight': 0.10, 'desc': 'Alignment across timeframes'},
        'session_quality':  {'weight': 0.10, 'desc': 'Peak liquidity hours'},
        'spread_condition': {'weight': 0.05, 'desc': 'Current spread quality'},
    }

    def score(self, features, direction, spread_pips):
        """Compute confidence score (0-100)."""
        s = {}

        # 1. Trend Alignment (0-10)
        h4b = features.get('h4_bullish', 0)
        h4s = features.get('h4_bearish', 0)
        if (direction == 'LONG' and h4b) or (direction == 'SHORT' and h4s):
            s['trend_alignment'] = 10
        elif (direction == 'LONG' and h4s) or (direction == 'SHORT' and h4b):
            s['trend_alignment'] = 0
        else:
            s['trend_alignment'] = 5

        # 2. Momentum (0-10)
        adx = features.get('adx', 0)
        adx_rising = features.get('adx_rising', 0)
        rsi = features.get('rsi_14', 50)
        macd_h = features.get('macd_histogram', 0)
        ms = 5
        if adx > 25: ms += 2
        if adx_rising: ms += 1
        if direction == 'LONG' and 40 <= rsi <= 55 and macd_h > 0: ms += 2
        elif direction == 'SHORT' and 45 <= rsi <= 60 and macd_h < 0: ms += 2
        s['momentum'] = min(10, ms)

        # 3. Volatility Regime (0-10)
        atr_r = features.get('atr_ratio', 1.0)
        if 0.8 <= atr_r <= 1.5: s['volatility_regime'] = 10
        elif 0.5 <= atr_r <= 2.0: s['volatility_regime'] = 7
        else: s['volatility_regime'] = 3

        # 4. Structure (0-10)
        bos = features.get('h1_bullish_bos', 0) if direction == 'LONG' else features.get('h1_bearish_bos', 0)
        vwap_d = abs(features.get('vwap_distance', 0))
        ss = 5
        if bos: ss += 3
        if vwap_d < 1.0: ss += 2
        s['structure'] = min(10, ss)

        # 5. Pattern Quality (0-10)
        br = features.get('body_ratio', 0.5)
        vr = features.get('volume_ratio', 1.0)
        ps = 5
        if br > 0.6: ps += 2
        if vr > 1.2: ps += 2
        if features.get('macd_cross', 0) != 0: ps += 1
        s['pattern_quality'] = min(10, ps)

        # 6. MTF Agreement (0-10)
        mtf = features.get('mtf_alignment', 0)
        s['mtf_agreement'] = min(10, mtf * 3 + 1)

        # 7. Session Quality (0-10)
        s['session_quality'] = 10 if features.get('is_overlap', 0) else 5

        # 8. Spread Condition (0-10)
        if spread_pips <= 0.1: s['spread_condition'] = 10
        elif spread_pips <= 0.3: s['spread_condition'] = 8
        elif spread_pips <= 0.5: s['spread_condition'] = 6
        else: s['spread_condition'] = 3

        # Weighted sum
        total = sum(s[k] * self.FACTORS[k]['weight'] for k in self.FACTORS)
        return round(total * 10, 1)  # Scale to 0-100
```

---

## 7. Signal Gate Pipeline (Inference)

```python
class SignalEngine:
    """
    Main event loop. Runs every M5 candle close.
    """
    def __init__(self, config):
        self.config = config
        self.filters = RuleBasedFilters(config)
        self.scorer = ConfidenceScorer()
        self.feature_pipeline = FeaturePipeline()
        self.models = None
        self.signal_log = []

    def run(self):
        """Main loop -- call every M5 candle close."""
        import time
        self.models = ModelManager().load_latest()

        while True:
            try:
                self._wait_for_m5_close()

                # 1. Fetch data
                fetcher = DataFetcher()
                market_data = {
                    'timestamp': datetime.utcnow(),
                    'ohlcv': fetcher.fetch_all(),
                    'spread': fetcher.fetch_spread(),
                    'dxy': fetcher.fetch_dxy(),
                    'us10y': fetcher.fetch_us10y(),
                }

                # 2. Compute features
                features = self._compute_all_features(market_data)

                # 3. Rule-based filter
                passed, reason = self.filters.evaluate(market_data, features)
                if not passed:
                    self._log_signal('NO-TRADE', 0, reason)
                    continue

                # 4. ML ensemble inference
                X = self._prepare_features(features)
                xgb_prob = self.models['xgb_model'].predict_proba(X)[:, 1]
                lgb_prob = self.models['lgb_model'].predict_proba(X)[:, 1]
                cat_prob = self.models['cat_model'].predict_proba(X)[:, 1]
                meta_features = np.column_stack([xgb_prob, lgb_prob, cat_prob])
                meta_prob = self.models['meta_model'].predict(meta_features)

                # 5. Determine direction
                direction = 'LONG' if meta_prob > 0.5 else 'SHORT'

                # 6. Confidence scoring
                confidence = self.scorer.score(features, direction, market_data['spread'])

                # 7. Threshold gate
                if confidence >= 70:
                    signal = direction
                    self._send_notification(signal, confidence, market_data)
                else:
                    signal = 'NO-TRADE'

                self._log_signal(signal, confidence, reason)

            except Exception as e:
                self._handle_error(e)

    def _wait_for_m5_close(self):
        import time
        now = datetime.utcnow()
        seconds_until = 300 - (now.minute % 300) * 60 - now.second
        time.sleep(seconds_until + 2)

    def _compute_all_features(self, market_data):
        ohlcv = market_data['ohlcv']
        price_feat = PriceFeatures().compute(ohlcv['M5'])
        tech_feat = TechnicalFeatures().compute(ohlcv['M5'])
        mtf_feat = MultiTimeframeFeatures().compute(
            ohlcv['M5'], ohlcv['M15'], ohlcv['H1'], ohlcv['H4']
        )
        sess_feat = SessionMacroFeatures().compute(
            market_data['timestamp'], market_data['dxy'], market_data['us10y']
        )
        all_feat = pd.concat([price_feat.iloc[[-1]], tech_feat.iloc[[-1]],
                             mtf_feat, sess_feat], axis=1)
        return all_feat.iloc[-1].to_dict()

    def _prepare_features(self, features):
        df = pd.DataFrame([features])
        return self.feature_pipeline.transform(df)

    def _send_notification(self, signal, confidence, market_data):
        # Telegram/Email notification implementation
        pass

    def _log_signal(self, signal, confidence, reason):
        self.signal_log.append({
            'timestamp': datetime.utcnow(),
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
        })

    def _handle_error(self, e):
        print(f"Error: {e}")
```

---

## 8. Notification Layer

```python
import requests

class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send_signal(self, signal, confidence, tp_pips, sl_pips, entry_price=None):
        emoji = '🟢' if signal == 'LONG' else '🔴' if signal == 'SHORT' else '⚪'
        msg = (
            f"{emoji} **EUR/USD Signal: {signal}**\n\n"
            f"Confidence: **{confidence}/100**\n"
            f"TP: {tp_pips} pips\n"
            f"SL: {sl_pips} pips\n"
            f"R:R: 3:1\n"
        )
        if entry_price:
            msg += f"Entry: {entry_price:.5f}\n"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': msg,
            'parse_mode': 'Markdown',
        }
        requests.post(url, json=payload, timeout=5)

class EmailNotifier:
    def __init__(self, smtp_server, smtp_port, sender, password, recipient):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password
        self.recipient = recipient

    def send_signal(self, signal, confidence, tp_pips, sl_pips):
        import smtplib
        from email.mime.text import MIMEText

        subject = f"EUR/USD Signal: {signal} (Confidence: {confidence}/100)"
        body = (
            f"Signal: {signal}\n"
            f"Confidence: {confidence}/100\n"
            f"TP: {tp_pips} pips\n"
            f"SL: {sl_pips} pips\n"
            f"R:R: 3:1\n"
        )

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.sender
        msg['To'] = self.recipient

        with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
            server.login(self.sender, self.password)
            server.send_message(msg)
```

---

## 9. Backtest Layer

### 9.1 Dual Backtester Design

```
VectorBT (Speed)                    Backtrader (Realism)
+---------------------------+       +---------------------------+
| - Numba JIT acceleration  |       | - Event-driven simulation |
| - 100-500x faster         |       | - Slippage modeling        |
| - Parameter sweeps        |       | - Commission modeling      |
| - Broadcasting combos     |       | - Position management      |
| - Walk-forward analyzer   |       | - Realistic fills          |
+---------------------------+       +---------------------------+
         |                                    |
         +-------+-------------------+--------+
                 |                   |
                 v                   v
        +-------------------+  +-------------------+
        | Quick iteration   |  | Final validation  |
        | & optimization    |  | before live       |
        +-------------------+  +-------------------+
```

### 9.2 VectorBT Parameter Sweep (from code analysis)

```python
import vectorbt as vbt

def sweep_parameters(close, entries, exits):
    """
    VectorBT broadcasting for parameter sweeps.
    Example: sweep RSI period 5-25 x MA period 10-50 = 400 combos instantly.
    """
    # MA crossover sweep
    fast = vbt.MA.run_combs(close, window=np.arange(5, 30), r=2, param_names=['fast', 'slow'])
    entries = fast.ma_crossed_above()
    exits = fast.ma_crossed_below()

    pf = vbt.Portfolio.from_signals(close, entries, exits,
                                     init_cash=10000,
                                     fees=0.0002,
                                     slippage=0.0001)
    return pf.stats()
```

### 9.3 Turbulence Index (from FinRL)

```python
class TurbulenceIndex:
    """
    Turbulence index from FinRL.
    Mahalanobis distance of current market state vs historical distribution.
    When > 99 percentile: suppress signals (risk-off).
    """
    def __init__(self, lookback=60):
        self.lookback = lookback

    def compute(self, returns_window):
        """Compute Mahalanobis distance of current state."""
        from scipy.spatial.distance import mahalanobis
        import numpy as np

        if len(returns_window) < self.lookback:
            return 0

        hist = returns_window[-self.lookback:]
        mean = hist.mean(axis=0)
        cov = np.cov(hist.T)
        current = returns_window[-1:]

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return 0

        dist = mahalanobis(current[0], mean, cov_inv)
        return dist

    def is_turbulent(self, returns_window, threshold_percentile=99):
        """Returns True if market is in turbulent state."""
        from scipy.stats import percentileofscore
        current_dist = self.compute(returns_window)
        hist_dists = [self.compute(returns_window[:i+1])
                     for i in range(self.lookback, len(returns_window))]
        pctile = percentileofscore(hist_dists, current_dist)
        return pctile > threshold_percentile
```

### 9.4 Backtest Configuration

```python
BACKTEST_CONFIG = {
    'symbol': 'EUR/USD',
    'period': '2021-2026',      # 5 years
    'timeframe': 'M5',
    'session': '13:00-16:00 UTC',
    'spread': 0.2,               # pips
    'slippage': 0.5,             # pips
    'commission': 7.0,           # USD per lot round-trip
    'tp_pips': 15,
    'sl_pips': 5,
    'risk_per_trade': 0.5,       # % of equity
    'max_daily_loss': 3.0,       # % of equity
}
```

### 9.5 Walk-Forward Backtest

```python
class WalkForwardBacktester:
    """
    Run walk-forward validation with realistic costs.
    Returns per-window metrics: win rate, profit factor, Sharpe, max drawdown.
    """
    def __init__(self, config):
        self.config = config

    def run(self, models, data, feature_pipeline):
        windows = []
        n = len(data)
        test_size = 21 * 12  # 1 month of M5 in overlap session

        for w in range(30):
            test_end = n - (29 - w) * test_size
            test_start = test_end - test_size

            if test_start < 0:
                continue

            test_data = data.iloc[test_start:test_end]
            predictions = self._predict_all(models, test_data, feature_pipeline)
            trades = self._simulate_trades(predictions, test_data)
            metrics = self._compute_metrics(trades)
            windows.append(metrics)

        return windows

    def _predict_all(self, models, data, pipeline):
        """Generate predictions for test period."""
        # Feature computation + ML inference for each bar
        pass

    def _simulate_trades(self, predictions, data):
        """Simulate trades with spread/slippage."""
        trades = []
        for pred in predictions:
            if pred['signal'] != 'NO-TRADE':
                entry = data.loc[pred['index'], 'close']
                # Add spread/slippage
                entry += self.config['spread'] * 0.0001 / 2
                entry += self.config['slippage'] * 0.0001
                trades.append({
                    'entry': entry,
                    'direction': pred['signal'],
                    'tp': self.config['tp_pips'] * 0.0001,
                    'sl': self.config['sl_pips'] * 0.0001,
                })
        return trades

    def _compute_metrics(self, trades):
        """Compute win rate, profit factor, Sharpe, max drawdown."""
        if not trades:
            return {'win_rate': 0, 'profit_factor': 0, 'sharpe': 0, 'max_dd': 0}
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        total = len(trades)
        return {
            'win_rate': wins / total if total > 0 else 0,
            'total_trades': total,
            # ... additional metrics
        }
```

---

## 10. Monitoring & Health

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

    def compute_accuracy(self, lookback_hours=24):
        """Compute win rate over last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        recent = [s for s in self.signals if s['timestamp'] > cutoff]
        # Match with trades, compute precision
        pass

    def check_drift(self, target_wr=0.35, window_days=7):
        """Alert if live accuracy drops below target."""
        live_wr = self.compute_accuracy(lookback_hours=24 * window_days)
        if live_wr < target_wr * 0.85:  # 15% degradation
            return True  # Retrain needed
        return False


class SpreadTracker:
    """Track real-time spread. Suppress if consistently > 0.5 pips."""
    def __init__(self, max_spread=0.5):
        self.spreads = []
        self.max_spread = max_spread

    def update(self, spread):
        self.spreads.append({'timestamp': datetime.utcnow(), 'spread': spread})

    def is_acceptable(self):
        if not self.spreads:
            return True
        recent = [s['spread'] for s in self.spreads[-20:]]
        return np.mean(recent) < self.max_spread
```

---

## 11. Deployment Architecture

### 11.1 Local Deployment (Recommended)

```
+-----------------------------------------+
|            YOUR WINDOWS PC              |
|                                         |
|  +----------------+  +----------------+ |
|  |  MT5 Terminal   |  |  Python Engine  | |
|  |  (EURUSD chart) |<-|  (signal eng.)  | |
|  |                 |  |                  | |
|  |  - Data feed    |  |  - Feature comp  | |
|  |  - (NO orders)  |  |  - ML inference  | |
|  +----------------+  |  - Signal gate    | |
|                       |  - Confidence     | |
|                       +--------+---------+ |
|                                |           |
|                       +--------v---------+ |
|                       |  Notifications    | |
|                       |  - Telegram bot   | |
|                       |  - Email alerts   | |
|                       +------------------+ |
+-----------------------------------------+
```

### 11.2 File Structure

```
eurusd-signal-engine/
+-- src/
|   +-- __init__.py
|   +-- config.py                 # All constants, thresholds, params
|   +-- mt5_connection.py         # MT5 connection, heartbeat
|   +-- data_pipeline.py          # Historical data pull, parquet storage
|   +-- feature_engineering.py    # All feature computation
|   +-- labels.py                 # Triple-barrier labeling
|   +-- model_trainer.py          # Walk-forward training, persistence
|   +-- model_inference.py        # Load model, predict
|   +-- rule_filters.py           # 8-rule filter pipeline
|   +-- confidence_scorer.py      # 8-factor scoring
|   +-- signal_engine.py          # Main event loop
|   +-- notifications.py          # Telegram, email
|   +-- metrics.py                # Accuracy tracking
|   +-- backtest.py               # VectorBT + Backtrader
|   +-- main.py                   # Entry point
+-- models/
|   +-- w0_xgb.joblib
|   +-- w0_lgb.joblib
|   +-- w0_cat.joblib
|   +-- w0_meta.joblib
|   +-- ... (30 windows)
+-- data/
|   +-- eurusd/ (M5/M15/H1/H4 parquet)
|   +-- dxy/
|   +-- us10y/
|   +-- labels/
|   +-- signals/
+-- config/
|   +-- config.yaml
+-- tests/
|   +-- test_features.py
|   +-- test_labels.py
|   +-- test_filters.py
|   +-- test_model.py
+-- scripts/
|   +-- pull_historical_data.py
|   +-- train_models.py
|   +-- backtest.py
|   +-- live_signals.py
+-- requirements.txt
+-- .env
+-- README.md
```

---

## 12. Configuration

```yaml
# config.yaml

symbol: "EURUSD"
pip_value: 0.0001  # 1 pip = 0.0001 for EUR/USD

# Labeling
labeling:
  tp_pips: 15
  sl_pips: 5
  horizon: 20  # M5 bars

# ML Models
models:
  xgboost:
    n_estimators: 300
    max_depth: 3
    learning_rate: 0.05
  lightgbm:
    n_estimators: 150
    max_depth: 4
    learning_rate: 0.05
  catboost:
    iterations: 200
    depth: 4
    learning_rate: 0.05

# Confidence Threshold
threshold:
  min_confidence: 70  # 0-100

# Session Filter
session:
  start_utc: 13
  end_utc: 16

# Volatility Filter
volatility:
  min_atr_ratio: 0.5
  max_atr_ratio: 2.5
  atr_period: 14

# Spread Filter
spread:
  max_pips: 0.5

# News Filter
news:
  block_minutes: 15
  impacts: ["High"]
  currencies: ["USD", "EUR"]

# Risk Management
risk:
  risk_per_trade_pct: 0.5
  max_daily_loss_pct: 3.0

# Walk-Forward Validation
validation:
  n_windows: 30
  train_months: 6
  test_months: 1
  purge_bars: 20
  embargo_bars: 20

# MT5 Connection
mt5:
  heartbeat_interval_sec: 5
  max_heartbeat_failures: 3
  reconnect_delay_sec: 10

# Notifications
telegram:
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"

email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 465
  sender: "${EMAIL_SENDER}"
  password: "${EMAIL_PASSWORD}"
  recipient: "${EMAIL_RECIPIENT}"
```

---

## 13. Training Workflow

### Step 1: Pull Historical Data (one-time)
```bash
python scripts/pull_historical_data.py --symbol EURUSD --years 5
```

### Step 2: Pull Macro Data (one-time)
```bash
python scripts/pull_macro_data.py --start 2021-01-01
```

### Step 3: Generate Labels
```bash
python scripts/generate_labels.py --tp 15 --sl 5 --horizon 20
```

### Step 4: Train Models
```bash
python scripts/train_models.py --windows 30
```

### Step 5: Validate (Walk-Forward)
```bash
python scripts/validate.py --method walk_forward --purge 20 --embargo 20
```

### Step 6: Backtest
```bash
python scripts/backtest.py --spread 0.2 --slippage 0.5 --commission 7
```

### Step 7: Go Live
```bash
python scripts/live_signals.py --mode paper    # Paper trading first
python scripts/live_signals.py --mode live     # After 2 weeks paper
```

---

## 14. Expected Performance Summary

| Metric | Target | Minimum | Notes |
|--------|--------|---------|-------|
| Win Rate | >35% | >30% | Break-even is 25% at 3:1 |
| Profit Factor | >1.5 | >1.2 | After realistic costs |
| Expectancy | >+0.40R | >+0.20R | Per trade |
| Signals/Day | 1-3 | 1-3 | Extreme selectivity |
| Max Drawdown | <10% | <15% | Per walk-forward test fold |
| Walk-forward consistency | >75% | >60% | Windows with positive return |
| Train-Test Gap | <5% | <10% | Overfitting guardrail |
| Sharpe Ratio | >1.0 | >0.5 | After costs |
| Inference Latency | <100ms | <500ms | Per M5 candle close |

---

*Architecture designed for implementation. Every component has specified inputs, outputs, and logic. Ready to begin coding in Phase 2.*

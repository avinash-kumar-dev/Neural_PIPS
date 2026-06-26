import pandas as pd
import numpy as np


class PriceFeatures:
    def compute(self, df):
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        features = {}

        for lag in [1, 2, 3, 5, 10, 20]:
            features[f'return_{lag}'] = close.pct_change(lag)

        for lag in [1, 3, 5]:
            features[f'log_return_{lag}'] = np.log(close / close.shift(lag))

        features['open_close_ratio'] = (close - open_) / open_
        features['high_low_ratio'] = (high - low) / close
        features['close_high_ratio'] = (high - close) / close
        features['close_low_ratio'] = (close - low) / close

        typical_price = (high + low + close) / 3
        volume = df['tick_volume']
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        atr = self._atr(df, 14)
        features['vwap_distance'] = (close - vwap) / atr

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


class TechnicalFeatures:
    def compute(self, df):
        close = df['close']
        features = {}

        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        for period in [8, 21, 50]:
            ema = close.ewm(span=period).mean()
            features[f'ema_{period}_slope'] = ema.pct_change(3)
            features[f'ema_{period}_above'] = (close > ema).astype(int)

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

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        features['bb_position'] = (close - lower) / (upper - lower)
        features['bb_width'] = (upper - lower) / sma20

        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - close.shift(1)),
            abs(df['low'] - close.shift(1))
        ], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_50'] = tr.rolling(50).mean()
        features['atr_ratio'] = features['atr_14'] / features['atr_50']

        adx_data = self._compute_adx(df, period=10)
        features['adx'] = adx_data['adx']
        features['plus_di'] = adx_data['plus_di']
        features['minus_di'] = adx_data['minus_di']
        features['adx_rising'] = (features['adx'] > features['adx'].shift(1)).astype(int)

        rsi = features.get('rsi_14', self._rsi(close, 14))
        stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        features['stoch_rsi_k'] = stoch_rsi.rolling(3).mean()
        features['stoch_rsi_d'] = features['stoch_rsi_k'].rolling(3).mean()

        vol_sma20 = df['tick_volume'].rolling(20).mean()
        features['volume_ratio'] = df['tick_volume'] / vol_sma20

        body = abs(close - df['open'])
        total_range = df['high'] - df['low']
        features['body_ratio'] = body / total_range

        direction = np.where(close >= close.shift(1), 1, -1)
        delta_volume = df['tick_volume'] * direction
        features['cvd_delta'] = delta_volume
        features['cvd_roc_5'] = pd.Series(delta_volume).rolling(5).sum().pct_change(5)

        return pd.DataFrame(features, index=df.index)

    def _rsi(self, close, period):
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

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


class MultiTimeframeFeatures:
    def compute(self, m5_df, m15_df, h1_df, h4_df):
        m5_times = pd.to_datetime(m5_df['time'])
        h4_times = pd.to_datetime(h4_df['time'])
        h1_times = pd.to_datetime(h1_df['time'])
        m15_times = pd.to_datetime(m15_df['time'])

        h4_ema9 = h4_df['close'].ewm(span=9).mean()
        h4_ema21 = h4_df['close'].ewm(span=21).mean()
        h4_ema200 = h4_df['close'].ewm(span=200).mean()
        h4_bullish_raw = ((h4_df['close'] > h4_ema200) & (h4_ema9 > h4_ema21)).astype(int)
        h4_bearish_raw = ((h4_df['close'] < h4_ema200) & (h4_ema9 < h4_ema21)).astype(int)

        h1_ema50 = h1_df['close'].ewm(span=50).mean()
        h1_above_ema50_raw = (h1_df['close'] > h1_ema50).astype(int)

        m15_rsi = self._rsi(m15_df['close'], 14)
        m15_rsi_bull_raw = ((m15_rsi >= 40) & (m15_rsi <= 55)).astype(int)
        m15_rsi_bear_raw = ((m15_rsi >= 45) & (m15_rsi <= 60)).astype(int)

        h1_bos_bull = self._detect_bos_series(h1_df, 'bullish')
        h1_bos_bear = self._detect_bos_series(h1_df, 'bearish')

        features = {
            'h4_bullish': self._ffill_to_m5(h4_bullish_raw.values, h4_times, m5_times),
            'h4_bearish': self._ffill_to_m5(h4_bearish_raw.values, h4_times, m5_times),
            'h1_above_ema50': self._ffill_to_m5(h1_above_ema50_raw.values, h1_times, m5_times),
            'm15_rsi': self._ffill_to_m5(m15_rsi.values, m15_times, m5_times),
            'm15_rsi_bullish': self._ffill_to_m5(m15_rsi_bull_raw.values, m15_times, m5_times),
            'm15_rsi_bearish': self._ffill_to_m5(m15_rsi_bear_raw.values, m15_times, m5_times),
            'h1_bullish_bos': self._ffill_to_m5(h1_bos_bull.values, h1_times, m5_times),
            'h1_bearish_bos': self._ffill_to_m5(h1_bos_bear.values, h1_times, m5_times),
        }

        h4_bull = pd.Series(features['h4_bullish'])
        h4_bear = pd.Series(features['h4_bearish'])
        h1_a50 = pd.Series(features['h1_above_ema50'])
        m15_bull = pd.Series(features['m15_rsi_bullish'])
        m15_bear = pd.Series(features['m15_rsi_bearish'])
        features['mtf_alignment'] = ((h4_bull | h4_bear).astype(int) + h1_a50.astype(int) + (m15_bull | m15_bear).astype(int)).values

        return pd.DataFrame(features, index=m5_df.index)

    def _ffill_to_m5(self, htf_values, htf_times, m5_times):
        s = pd.Series(htf_values, index=htf_times)
        s = s[~s.index.duplicated(keep='last')]
        s = s.sort_index()
        mapped = s.reindex(m5_times, method='ffill')
        return mapped.values

    def _rsi(self, close, period):
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _detect_bos_series(self, df, direction, lookback=10):
        close = df['close']
        high = df['high']
        low = df['low']
        bos = pd.Series(0, index=df.index)
        for i in range(lookback + 2, len(df)):
            if direction == 'bullish':
                swing_high = high.iloc[i-lookback-2:i-2].max()
                if close.iloc[i] > swing_high and close.iloc[i-1] <= swing_high:
                    bos.iloc[i] = 1
            else:
                swing_low = low.iloc[i-lookback-2:i-2].min()
                if close.iloc[i] < swing_low and close.iloc[i-1] >= swing_low:
                    bos.iloc[i] = 1
        return bos


class SessionMacroFeatures:
    def compute(self, timestamp_utc):
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

        return pd.DataFrame(features, index=[0])

    def compute_batch(self, timestamps):
        rows = [self.compute(ts).iloc[0].to_dict() for ts in timestamps]
        return pd.DataFrame(rows, index=timestamps.index)


class FeatureEngineer:
    FEATURE_VECTOR = [
        'return_1', 'return_2', 'return_3', 'return_5', 'return_10', 'return_20',
        'log_return_1', 'log_return_3', 'log_return_5',
        'open_close_ratio', 'high_low_ratio', 'close_high_ratio', 'close_low_ratio',
        'vwap_distance', 'price_position_20',
        'rsi_7', 'rsi_14', 'rsi_21',
        'ema_8_slope', 'ema_8_above', 'ema_21_slope', 'ema_21_above',
        'ema_50_slope', 'ema_50_above',
        'macd_histogram', 'macd_cross',
        'bb_position', 'bb_width',
        'atr_14', 'atr_50', 'atr_ratio',
        'adx', 'plus_di', 'minus_di', 'adx_rising',
        'stoch_rsi_k', 'stoch_rsi_d',
        'volume_ratio', 'body_ratio', 'cvd_delta', 'cvd_roc_5',
        'h4_bullish', 'h4_bearish', 'h1_above_ema50',
        'm15_rsi', 'm15_rsi_bullish', 'm15_rsi_bearish',
        'mtf_alignment', 'h1_bullish_bos', 'h1_bearish_bos',
        'session_sin', 'session_cos', 'is_overlap', 'is_london', 'is_ny',
        'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4',
    ]

    def __init__(self):
        self.price = PriceFeatures()
        self.technical = TechnicalFeatures()
        self.mtf = MultiTimeframeFeatures()
        self.session = SessionMacroFeatures()

    def compute_all(self, m5_df, m15_df, h1_df, h4_df):
        price_feat = self.price.compute(m5_df)
        tech_feat = self.technical.compute(m5_df)
        mtf_feat = self.mtf.compute(m5_df, m15_df, h1_df, h4_df)
        session_feat = self.session.compute_batch(m5_df['time'])

        all_features = pd.concat([price_feat, tech_feat, mtf_feat, session_feat], axis=1)
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        return all_features

    def get_feature_names(self):
        return self.FEATURE_VECTOR

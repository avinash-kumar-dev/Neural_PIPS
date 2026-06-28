import pandas as pd
import numpy as np


def compute_hurst_exponent(series: pd.Series, window: int = 200) -> pd.Series:
    def _hurst(data):
        if len(data) < 20:
            return 0.5
        lags = range(2, min(20, len(data) // 2))
        tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
        if any(t <= 0 for t in tau):
            return 0.5
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return poly[0] * 2.0

    return series.rolling(window, min_periods=window // 2).apply(_hurst, raw=True)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
    )

    atr = pd.Series(tr).ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, min_periods=period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()
    return adx


def compute_choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
    )

    atr_sum = pd.Series(tr).rolling(period).sum()
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()

    chop = 100 * np.log10(atr_sum / (highest - lowest + 1e-10)) / np.log10(period)
    return chop


def compute_regime(
    df: pd.DataFrame,
    hurst_window: int = 200,
    adx_period: int = 14,
    chop_period: int = 14,
    confirm_bars: int = 3,
) -> pd.DataFrame:
    result = df.copy()

    hurst = compute_hurst_exponent(result["close"], window=hurst_window)
    adx = compute_adx(result, period=adx_period)
    chop = compute_choppiness_index(result, period=chop_period)

    result["hurst"] = hurst
    result["adx"] = adx
    result["choppiness"] = chop

    n = len(result)
    raw_regime = np.zeros(n, dtype=int)

    for i in range(n):
        h = hurst.iloc[i] if not np.isnan(hurst.iloc[i]) else 0.5
        a = adx.iloc[i] if not np.isnan(adx.iloc[i]) else 20
        c = chop.iloc[i] if not np.isnan(chop.iloc[i]) else 50

        votes = 0
        if h > 0.55:
            votes += 1
        elif h < 0.45:
            votes -= 1

        if a > 25:
            votes += 1
        elif a < 20:
            votes -= 1

        if c < 38.2:
            votes += 1
        elif c > 61.8:
            votes -= 1

        if votes >= 2:
            raw_regime[i] = 1
        elif votes <= -2:
            raw_regime[i] = -1
        else:
            raw_regime[i] = 0

    confirmed_regime = np.zeros(n, dtype=int)
    current_regime = 0
    count = 0

    for i in range(n):
        if raw_regime[i] == current_regime:
            count += 1
        else:
            count = 1
            current_regime = raw_regime[i]

        if count >= confirm_bars:
            confirmed_regime[i] = current_regime
        else:
            confirmed_regime[i] = confirmed_regime[i - 1] if i > 0 else 0

    result["regime_raw"] = raw_regime
    result["regime"] = confirmed_regime
    result["regime_trending"] = confirmed_regime == 1
    result["regime_ranging"] = confirmed_regime == -1
    result["regime_transition"] = confirmed_regime == 0

    return result

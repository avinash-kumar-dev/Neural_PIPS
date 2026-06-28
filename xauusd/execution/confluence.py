import pandas as pd
import numpy as np


CONFLUENCE_FACTORS = {
    "market_structure": 0.15,
    "regime": 0.12,
    "liquidity_sweep": 0.10,
    "ob_proximity": 0.10,
    "fvg_proximity": 0.08,
    "mtf_alignment": 0.15,
    "session": 0.10,
    "volume": 0.08,
    "rsi_momentum": 0.07,
    "risk_reward": 0.05,
}


def score_market_structure(row: pd.Series, direction: int) -> float:
    score = 0.0
    if direction == 1:
        if row.get("bos_bullish", False) or row.get("choch_bullish", False):
            score = 1.0
        elif row.get("layer1_bias", 0) == 1:
            score = 0.7
    else:
        if row.get("bos_bearish", False) or row.get("choch_bearish", False):
            score = 1.0
        elif row.get("layer1_bias", 0) == -1:
            score = 0.7
    return score


def score_regime(row: pd.Series, direction: int) -> float:
    if direction == 1 and row.get("regime_trending", False):
        return 1.0
    elif direction == -1 and row.get("regime_trending", False):
        return 1.0
    elif row.get("regime_ranging", False):
        return 0.5
    return 0.3


def score_liquidity_sweep(row: pd.Series) -> float:
    if row.get("sweep_high_signal", False) or row.get("sweep_low_signal", False):
        return 1.0
    return 0.0


def score_ob_proximity(row: pd.Series, direction: int) -> float:
    if direction == 1:
        if row.get("ob_bull_signal", False):
            return 1.0
        if row.get("has_breaker_bull", False):
            return 0.8
        if row.get("has_rej_bull", False):
            return 0.5
    else:
        if row.get("ob_bear_signal", False):
            return 1.0
        if row.get("has_breaker_bear", False):
            return 0.8
        if row.get("has_rej_bear", False):
            return 0.5
    return 0.0


def score_fvg_proximity(row: pd.Series, direction: int) -> float:
    if direction == 1:
        if row.get("fvg_bull_hold", False):
            return 1.0
        if row.get("fvg_bull_fill", False):
            return 0.8
        if row.get("has_bpr", False):
            return 1.0
        if row.get("has_ifvg_bull", False):
            return 0.7
    else:
        if row.get("fvg_bear_hold", False):
            return 1.0
        if row.get("fvg_bear_fill", False):
            return 0.8
        if row.get("has_bpr", False):
            return 1.0
        if row.get("has_ifvg_bear", False):
            return 0.7
    return 0.0


def score_mtf_alignment(row: pd.Series, direction: int) -> float:
    score = 0.0
    if direction == 1:
        if row.get("layer1_bias", 0) == 1:
            score += 0.5
        if row.get("in_ote_bull", False):
            score += 0.3
        if row.get("pd_discount", False):
            score += 0.2
    else:
        if row.get("layer1_bias", 0) == -1:
            score += 0.5
        if row.get("in_ote_bear", False):
            score += 0.3
        if row.get("pd_premium", False):
            score += 0.2
    return min(score, 1.0)


def score_session(row: pd.Series) -> float:
    dt = row.get("datetime", None)
    if dt is None:
        return 0.5

    if hasattr(dt, "hour"):
        hour = dt.hour
    else:
        return 0.5

    if 7 <= hour < 10:
        return 1.0
    elif 13 <= hour < 17:
        return 1.0
    elif 10 <= hour < 13:
        return 0.7
    elif 17 <= hour < 20:
        return 0.5
    else:
        return 0.0


def score_volume(row: pd.Series) -> float:
    if row.get("volume_above_ma", False) or row.get("m5_volume_spike", False):
        return 1.0
    return 0.3


def score_rsi_momentum(row: pd.Series, direction: int) -> float:
    rsi = row.get("rsi", 50)
    if direction == 1 and rsi < 70:
        return 1.0 if rsi < 40 else 0.7
    elif direction == -1 and rsi > 30:
        return 1.0 if rsi > 60 else 0.7
    return 0.3


def score_risk_reward(row: pd.Series, direction: int) -> float:
    sl = row.get("long_sl", np.nan) if direction == 1 else row.get("short_sl", np.nan)
    if pd.isna(sl):
        return 0.0
    entry = row.get("close", 0)
    sl_pips = abs(entry - sl) / 0.01
    if sl_pips <= 0:
        return 0.0
    return 1.0


def compute_confluence_score(
    df: pd.DataFrame,
    direction: int,
) -> pd.Series:
    scores = pd.DataFrame(index=df.index)

    for factor, weight in CONFLUENCE_FACTORS.items():
        if factor == "market_structure":
            scores[factor] = df.apply(lambda r: score_market_structure(r, direction), axis=1)
        elif factor == "regime":
            scores[factor] = df.apply(lambda r: score_regime(r, direction), axis=1)
        elif factor == "liquidity_sweep":
            scores[factor] = df.apply(score_liquidity_sweep, axis=1)
        elif factor == "ob_proximity":
            scores[factor] = df.apply(lambda r: score_ob_proximity(r, direction), axis=1)
        elif factor == "fvg_proximity":
            scores[factor] = df.apply(lambda r: score_fvg_proximity(r, direction), axis=1)
        elif factor == "mtf_alignment":
            scores[factor] = df.apply(lambda r: score_mtf_alignment(r, direction), axis=1)
        elif factor == "session":
            scores[factor] = df.apply(score_session, axis=1)
        elif factor == "volume":
            scores[factor] = df.apply(score_volume, axis=1)
        elif factor == "rsi_momentum":
            scores[factor] = df.apply(lambda r: score_rsi_momentum(r, direction), axis=1)
        elif factor == "risk_reward":
            scores[factor] = df.apply(lambda r: score_risk_reward(r, direction), axis=1)

    total = sum(scores[f] * w for f, w in CONFLUENCE_FACTORS.items())
    return total * 100

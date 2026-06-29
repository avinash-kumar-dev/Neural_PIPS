import pandas as pd
import numpy as np


CONFLUENCE_FACTORS = {
    "market_structure": 0.12,
    "regime": 0.08,
    "liquidity_sweep": 0.07,
    "ob_proximity": 0.08,
    "fvg_proximity": 0.07,
    "mtf_alignment": 0.10,
    "session": 0.06,
    "volume": 0.06,
    "rsi_momentum": 0.05,
    "risk_reward": 0.04,
    "m5_confirmation": 0.12,
    "m5_quality": 0.08,
    "price_location": 0.07,
    "cascade_agreement": 0.05,
    "signal_strength": 0.05,
}


def compute_confluence_score(
    df: pd.DataFrame,
    direction: int,
) -> pd.Series:
    n = len(df)

    if direction == 1:
        ms = np.where(
            df.get("bos_bullish", False) | df.get("choch_bullish", False), 1.0,
            np.where(df.get("layer1_bias", 0) == 1, 0.7, 0.0)
        )
    else:
        ms = np.where(
            df.get("bos_bearish", False) | df.get("choch_bearish", False), 1.0,
            np.where(df.get("layer1_bias", 0) == -1, 0.7, 0.0)
        )

    regime = np.where(
        df.get("regime_trending", False), 1.0,
        np.where(df.get("regime_ranging", False), 0.7, 0.5)
    )

    sweep = np.where(
        df.get("sweep_high_signal", False) | df.get("sweep_low_signal", False), 1.0, 0.0
    )

    if direction == 1:
        ob = np.where(
            df.get("ob_bull_signal", False), 1.0,
            np.where(df.get("has_breaker_bull", False), 0.8,
                     np.where(df.get("has_rej_bull", False), 0.5, 0.0))
        )
    else:
        ob = np.where(
            df.get("ob_bear_signal", False), 1.0,
            np.where(df.get("has_breaker_bear", False), 0.8,
                     np.where(df.get("has_rej_bear", False), 0.5, 0.0))
        )

    if direction == 1:
        fvg = np.where(
            df.get("fvg_bull_hold", False), 1.0,
            np.where(df.get("has_bpr", False), 1.0,
                     np.where(df.get("fvg_bull_fill", False), 0.8,
                              np.where(df.get("has_ifvg_bull", False), 0.7, 0.0)))
        )
    else:
        fvg = np.where(
            df.get("fvg_bear_hold", False), 1.0,
            np.where(df.get("has_bpr", False), 1.0,
                     np.where(df.get("fvg_bear_fill", False), 0.8,
                              np.where(df.get("has_ifvg_bear", False), 0.7, 0.0)))
        )

    if direction == 1:
        mtf = np.minimum(
            np.where(df.get("layer1_bias", 0) == 1, 0.5, 0.0) +
            np.where(df.get("in_ote_bull", False), 0.3, 0.0) +
            np.where(df.get("pd_discount", False), 0.2, 0.0),
            1.0
        )
    else:
        mtf = np.minimum(
            np.where(df.get("layer1_bias", 0) == -1, 0.5, 0.0) +
            np.where(df.get("in_ote_bear", False), 0.3, 0.0) +
            np.where(df.get("pd_premium", False), 0.2, 0.0),
            1.0
        )

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        hour = dt.dt.hour.values
        sess = np.where(
            ((hour >= 6) & (hour < 12)) | ((hour >= 13) & (hour < 21)), 1.0,
            np.where((hour >= 12) & (hour < 13), 0.7,
                     np.where((hour >= 21) & (hour < 23), 0.5, 0.0))
        )
    else:
        sess = np.full(n, 0.5)

    vol = np.where(
        df.get("volume_above_ma", False) | df.get("m5_volume_spike", False), 1.0, 0.3
    )

    rsi = df.get("rsi", pd.Series(50, index=df.index)).values
    if direction == 1:
        rsi_score = np.where(rsi < 40, 1.0, np.where(rsi < 70, 0.7, 0.3))
    else:
        rsi_score = np.where(rsi > 60, 1.0, np.where(rsi > 30, 0.7, 0.3))

    sl_col = "long_sl" if direction == 1 else "short_sl"
    sl_vals = df.get(sl_col, pd.Series(np.nan, index=df.index)).values
    rr = np.where(~pd.isna(sl_vals) & (sl_vals > 0), 1.0, 0.0)

    if direction == 1:
        m5_confirm = np.where(df.get("m5_bull_confirm", False), 1.0,
                     np.where(df.get("m5_has_mss", False), 0.7,
                     np.where(df.get("m5_has_rejection", False), 0.6,
                     np.where(df.get("m5_has_engulfing", False), 0.5, 0.0))))
    else:
        m5_confirm = np.where(df.get("m5_bear_confirm", False), 1.0,
                     np.where(df.get("m5_has_mss", False), 0.7,
                     np.where(df.get("m5_has_rejection", False), 0.6,
                     np.where(df.get("m5_has_engulfing", False), 0.5, 0.0))))

    m5_quality = np.where(
        df.get("m5_has_volume", False) & (m5_confirm > 0), 1.0,
        np.where(m5_confirm > 0, 0.5, 0.0)
    )

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df.get("atr", pd.Series(np.nan, index=df.index)).values

    if direction == 1:
        near_ob = df.get("ob_bull_signal", pd.Series(False, index=df.index)).values
        near_fvg = df.get("fvg_bull_hold", pd.Series(False, index=df.index)).values | df.get("has_bpr", pd.Series(False, index=df.index)).values
        near_ote = df.get("in_ote_bull", pd.Series(False, index=df.index)).values
        price_loc = np.where(near_ob, 1.0,
                    np.where(near_fvg, 0.8,
                    np.where(near_ote, 0.6, 0.2)))
    else:
        near_ob = df.get("ob_bear_signal", pd.Series(False, index=df.index)).values
        near_fvg = df.get("fvg_bear_hold", pd.Series(False, index=df.index)).values | df.get("has_bpr", pd.Series(False, index=df.index)).values
        near_ote = df.get("in_ote_bear", pd.Series(False, index=df.index)).values
        price_loc = np.where(near_ob, 1.0,
                    np.where(near_fvg, 0.8,
                    np.where(near_ote, 0.6, 0.2)))

    cascade = df.get("cascade_bias", pd.Series(0, index=df.index)).values
    if direction == 1:
        cascade_agree = np.where(cascade == 1, 1.0,
                        np.where(cascade == 0, 0.5, 0.0))
    else:
        cascade_agree = np.where(cascade == -1, 1.0,
                        np.where(cascade == 0, 0.5, 0.0))

    factors_present = (
        (ms > 0).astype(float) +
        (sweep > 0).astype(float) +
        (ob > 0).astype(float) +
        (fvg > 0).astype(float) +
        (m5_confirm > 0).astype(float) +
        (vol > 0.5).astype(float) +
        (rsi_score > 0.5).astype(float)
    )
    strength = np.clip(factors_present / 7.0, 0.0, 1.0)

    total = (
        ms * CONFLUENCE_FACTORS["market_structure"] +
        regime * CONFLUENCE_FACTORS["regime"] +
        sweep * CONFLUENCE_FACTORS["liquidity_sweep"] +
        ob * CONFLUENCE_FACTORS["ob_proximity"] +
        fvg * CONFLUENCE_FACTORS["fvg_proximity"] +
        mtf * CONFLUENCE_FACTORS["mtf_alignment"] +
        sess * CONFLUENCE_FACTORS["session"] +
        vol * CONFLUENCE_FACTORS["volume"] +
        rsi_score * CONFLUENCE_FACTORS["rsi_momentum"] +
        rr * CONFLUENCE_FACTORS["risk_reward"] +
        m5_confirm * CONFLUENCE_FACTORS["m5_confirmation"] +
        m5_quality * CONFLUENCE_FACTORS["m5_quality"] +
        price_loc * CONFLUENCE_FACTORS["price_location"] +
        cascade_agree * CONFLUENCE_FACTORS["cascade_agreement"] +
        strength * CONFLUENCE_FACTORS["signal_strength"]
    )

    return pd.Series(total * 100, index=df.index)

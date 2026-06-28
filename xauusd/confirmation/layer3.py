import pandas as pd
import numpy as np


def compute_volume_filter(
    df: pd.DataFrame,
    ma_period: int = 20,
) -> pd.DataFrame:
    result = df.copy()
    result["volume_ma"] = result["volume"].rolling(ma_period).mean()
    result["volume_above_ma"] = result["volume"] > result["volume_ma"]
    return result


def compute_candle_body_filter(
    df: pd.DataFrame,
    body_ratio_min: float = 0.6,
) -> pd.DataFrame:
    result = df.copy()
    candle_range = result["high"] - result["low"]
    body = abs(result["close"] - result["open"])
    result["body_ratio"] = np.where(candle_range > 0, body / candle_range, 0)
    result["strong_body"] = result["body_ratio"] >= body_ratio_min
    return result


def compute_layer3(
    df: pd.DataFrame,
    volume_ma_period: int = 20,
    body_ratio_min: float = 0.6,
) -> pd.DataFrame:
    result = compute_volume_filter(df, ma_period=volume_ma_period)
    result = compute_candle_body_filter(result, body_ratio_min=body_ratio_min)
    result["layer3_confirmed"] = result["volume_above_ma"] & result["strong_body"]
    return result

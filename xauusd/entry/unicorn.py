import pandas as pd
import numpy as np


def detect_unicorn(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()
    n = len(result)

    has_bull_breaker = result.get("has_breaker_bull", pd.Series(False, index=result.index)).values if "has_breaker_bull" in result.columns else np.zeros(n, dtype=bool)
    has_bear_breaker = result.get("has_breaker_bear", pd.Series(False, index=result.index)).values if "has_breaker_bear" in result.columns else np.zeros(n, dtype=bool)

    bull_fvg_h = result.get("fvg_bull_high", pd.Series(np.nan, index=result.index)).values if "fvg_bull_high" in result.columns else np.full(n, np.nan)
    bull_fvg_l = result.get("fvg_bull_low", pd.Series(np.nan, index=result.index)).values if "fvg_bull_low" in result.columns else np.full(n, np.nan)
    bear_fvg_h = result.get("fvg_bear_high", pd.Series(np.nan, index=result.index)).values if "fvg_bear_high" in result.columns else np.full(n, np.nan)
    bear_fvg_l = result.get("fvg_bear_low", pd.Series(np.nan, index=result.index)).values if "fvg_bear_low" in result.columns else np.full(n, np.nan)

    breaker_bull_h = result.get("breaker_bull_high", pd.Series(np.nan, index=result.index)).values if "breaker_bull_high" in result.columns else np.full(n, np.nan)
    breaker_bull_l = result.get("breaker_bull_low", pd.Series(np.nan, index=result.index)).values if "breaker_bull_low" in result.columns else np.full(n, np.nan)
    breaker_bear_h = result.get("breaker_bear_high", pd.Series(np.nan, index=result.index)).values if "breaker_bear_high" in result.columns else np.full(n, np.nan)
    breaker_bear_l = result.get("breaker_bear_low", pd.Series(np.nan, index=result.index)).values if "breaker_bear_low" in result.columns else np.full(n, np.nan)

    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values

    unicorn_bull = np.zeros(n, dtype=bool)
    unicorn_bear = np.zeros(n, dtype=bool)
    unicorn_bull_mid = np.full(n, np.nan)
    unicorn_bear_mid = np.full(n, np.nan)

    for i in range(n):
        if has_bull_breaker[i] and not np.isnan(breaker_bull_h[i]) and not np.isnan(bull_fvg_h[i]):
            fvg_mid = (bull_fvg_h[i] + bull_fvg_l[i]) / 2
            if bull_fvg_l[i] <= breaker_bull_h[i] and bull_fvg_h[i] >= breaker_bull_l[i]:
                if lows[i] < bull_fvg_l[i] and closes[i] > bull_fvg_l[i]:
                    unicorn_bull[i] = True
                    unicorn_bull_mid[i] = fvg_mid

        if has_bear_breaker[i] and not np.isnan(breaker_bear_h[i]) and not np.isnan(bear_fvg_h[i]):
            fvg_mid = (bear_fvg_h[i] + bear_fvg_l[i]) / 2
            if bear_fvg_l[i] <= breaker_bear_h[i] and bear_fvg_h[i] >= breaker_bear_l[i]:
                if highs[i] > bear_fvg_h[i] and closes[i] < bear_fvg_h[i]:
                    unicorn_bear[i] = True
                    unicorn_bear_mid[i] = fvg_mid

    result["unicorn_bull"] = unicorn_bull
    result["unicorn_bear"] = unicorn_bear
    result["unicorn_bull_mid"] = unicorn_bull_mid
    result["unicorn_bear_mid"] = unicorn_bear_mid
    return result

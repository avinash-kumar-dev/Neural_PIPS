# US30 / US100 Strategy Reference — "Supertrend + EMA200 + BOS"
# Last known good: commit 3147bb6 on multi-asset-a1 branch
# Walk-forward: +857 pips/month combined, 54% profitable windows

## Entry Signal (Trend + Momentum)
- Supertrend(10, 3.0) direction = 1 (LONG) or -1 (SHORT)
- OR Supertrend(20, 4.0) direction = 1 or -1
- Price > EMA(200) for LONG, Price < EMA(200) for SHORT
- Confirmation: BOS bull/bear OR Bullish/Bearish Engulfing OR Pin Bar
- ADX(14) > 20
- RSI(14) < 70 (LONG), > 30 (SHORT)

## Session Filter
- US30: 13:00-21:00 UTC
- US100: 13:00-21:00 UTC

## Anti-Clustering
- Max 5 trades per day per instrument
- Minimum 4 bars between trades

## Stop Loss
- 1.5x ATR(14) below entry (LONG) or above entry (SHORT)

## Take Profit
- Engine TP1 = 2x risk, TP2 = 3x risk (hardcoded)

## Trade Management (Engine)
- Breakeven trigger: 2R MFE -> SL moves to entry + 1 pip
- Trailing start: 3R MFE -> trail at (MFE_R - 1) * SL_dist
- Stale exit: 80 bars if PnL < 0.3R
- Max bars: 300 bars hard exit
- Max concurrent: 10 trades

## Costs
- Spread: 1.0 pip
- Slippage: 0.5 pip
- Commission: $7/lot

## Walk-Forward Results (13 windows, 6mo train / 1mo test)
| Instrument | Total PnL | WR | PF |
|---|---|---|---|
| US30 | +9,721 | 38% | 1.44 |
| US100 | +2,617 | 35% | 1.29 |

## How to Revert
```
git log --oneline | head -20
git checkout <commit_hash> -- multi_asset/data/indicators.py multi_asset/backtest/engine.py
```

## Files
- Signal generation: inline in `run_iteration2.py` (vectorized)
- Indicators: `multi_asset/data/indicators.py`
- Backtest engine: `multi_asset/backtest/engine.py`
- Walk-forward: `multi_asset/scripts/run_walk_forward.py`

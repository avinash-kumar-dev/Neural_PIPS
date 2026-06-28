# XAUUSD Gold Bot — Master Plan

## Architecture: H4 → M15 → M5 Cascade

Each timeframe has ONE job. No mixing.

### H4 (Direction Bias)
- **Job**: WHERE are we going?
- **Tools**: BOS/CHoCH, EMA 50/200, regime detection (Hurst+ADX+Choppiness)
- **Output**: LONG_ONLY / SHORT_ONLY / NO_BIAS
- **Rule**: NEVER trade against H4 bias

### M15 (Setup)
- **Job**: WHAT is the setup?
- **Tools**: PD Array Matrix (OB, FVG, BPR, Breaker, IFVG, Unicorn, Rejection)
- **Plus**: Premium/Discount zones, OTE, Liquidity Pools, VWAP, Asian Range
- **Output**: Which PD array is price at? Is it fresh? Is it in premium/discount zone?
- **Rule**: Only enter at high-quality PD array zones

### M5 (Entry Trigger)
- **Job**: WHEN exactly do I enter?
- **Tools**: MSS/CHoCH confirmation, rejection candle, engulfing, volume spike, displacement
- **Output**: Precise entry candle with exact entry price
- **Rule**: Wait for M5 confirmation at M15 PD array zone

## Risk Management

### Minimum R:R = 2:1 (HARD FLOOR)
- SL: M15 structural level (swing low/high, OB boundary, FVG edge)
- TP1: 1x SL (partial exit optional)
- TP2: 2x SL (minimum target)
- SL must be placed at a STRUCTURAL level, not arbitrary distance

### Trailing Stop
- After 1R profit: Move SL to breakeven + 1 pip
- After 2R profit: Trail SL at 1.5R behind the peak
- This lets winners run while protecting profits

### Position Sizing
- Risk per trade: 3% of equity
- Max 3 trades per session
- Max 3 consecutive losses → halt for the session

## Session Windows
- London: 06:00-12:00 UTC
- Overlap: 12:00-17:00 UTC  
- New York: 17:00-21:00 UTC
- NO TRADES outside these windows

## Confluence Scoring (10 factors, 0-100)
1. Market Structure (15%): BOS/CHoCH direction
2. Regime (12%): Trending > Ranging > Transition
3. Liquidity Sweep (10%): Stop hunt detected
4. OB Proximity (10%): At order block
5. FVG Proximity (8%): At fair value gap
6. MTF Alignment (15%): H4+M15+M5 all agree
7. Session (10%): Active trading hours
8. Volume (8%): Above average
9. RSI Momentum (7%): Not overbought/oversold
10. Risk Reward (5%): R:R >= 2:1

## Anti-Clustering
- 6-bar cooldown between trades (M15)
- Max 3 trades per session
- Max 3 consecutive losses → halt

## Key Parameters
| Parameter | Value |
|-----------|-------|
| Symbol | XAUUSD only |
| Entry TF | M15 |
| Confirmation TF | M5 |
| Bias TF | H4 |
| SL | M15 structural level |
| TP | 2x SL minimum |
| R:R | >= 2:1 (hard floor) |
| Session | 06:00-21:00 UTC |
| Confidence | >= 70/100 |
| H4 Filter | HARD (never trade against) |

## Pipeline Order
1. H4 → regime + structure + EMA filter → direction bias
2. M15 → PD Array Matrix → which setup?
3. M5 → entry confirmation → when to enter?
4. Confluence scoring → is this good enough?
5. Anti-clustering → not overtrading
6. Session filter → right time?
7. Execute signal

## Target
- 1-2 quality trades per day
- Catching 200+ pip moves
- 4,000 pips/month

*Created: 2026-06-29*

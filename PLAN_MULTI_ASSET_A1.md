# Multi-Asset A1 Trading System — Architecture Plan
> Branch: multi-asset-a1 | Created: 2026-06-29

---

## GOAL
Build a **multi-asset A1 signal generator** targeting **4,000+ pips/month** across:
- US100 (Nasdaq 100)
- US30 (Dow Jones)
- EUR/USD
- GBP/USD
- USD/JPY
- XAUUSD (Gold) — kept but deprioritized

**Only A1 setups** — highest accuracy, best quality, no compromise.

---

## CORE PRINCIPLES (From Research)

1. **Specialists earn 52x MORE** → Focus on 2-3 instruments max per strategy
2. **Winning traders use 2.7 indicators** → Keep it SIMPLE
3. **Liquidity Sweep + MSS + FVG** is #1 setup → Build around this
4. **1-2 trades per day per instrument** → Quality over quantity
5. **Minimum 2:1 R:R** → Hard floor on every trade
6. **Session timing matters** → Only trade during optimal hours per instrument

---

## ARCHITECTURE

### Layer 1: Multi-Asset Data Pipeline
```
MT5 (Forex/Gold) + yfinance (US100/US30)
         │
         ▼
    Unified OHLCV format
    Per-instrument config (spread, session, ATR, pip value)
```

### Layer 2: A1 Setup Detection (Per Instrument)
```
Each instrument gets its OWN setup config:
  - US100: Trend Following + Breakout (US session)
  - US30: Range Trading + News (US session)
  - EUR/USD: Mean Reversion + Liquidity Sweep (London-NY)
  - GBP/USD: Trend Following + Momentum (London)
  - USD/JPY: Carry Trade + Trend (Tokyo+NY)
  - XAUUSD: ICT/SMC + Regime Adaptive (London-NY)
```

### Layer 3: A1 Signal Quality Gate
```
MULTI-FACTOR QUALITY CHECK (must pass ALL):
  ✓ Trend alignment (HTF direction)
  ✓ Key level proximity (OB, FVG, VWAP, S/R)
  ✓ Confirmation pattern (MSS, engulfing, pin bar)
  ✓ Volume confirmation
  ✓ R:R >= 2:1
  ✓ Correct session timing
  ✓ No conflicting signals
  ✓ Spread within limits
```

### Layer 4: Risk Management
```
Per-instrument position sizing:
  - US100: 1% risk (high volatility)
  - US30: 1% risk
  - EUR/USD: 2% risk (tighter spreads)
  - GBP/USD: 1.5% risk
  - USD/JPY: 1.5% risk
  - XAUUSD: 1% risk (high volatility)

Portfolio-level:
  - Max 3 concurrent positions
  - Max 5% total portfolio heat
  - Daily loss limit: 3%
```

### Layer 5: Notification + Dashboard
```
Telegram alerts per instrument:
  - Instrument + Direction + Entry + SL + TP
  - Session timing
  - Quality score (A1/A2/B)
  - Reason for trade
```

---

## A1 SETUP DEFINITIONS

### Setup 1: Liquidity Sweep + MSS + FVG (Highest Priority)
**Instruments**: EUR/USD, GBP/USD, US100
**Timeframes**: H1 bias → M15 setup → M5 entry
**Win Rate Target**: 60%+
**R:R Target**: 2.5:1+

**Rules**:
1. HTF (H1) shows clear trend direction
2. Price sweeps a known liquidity pool (EQH/EQL, session high/low)
3. LTF (M15) shows MSS (Market Structure Shift) or BOS
4. Entry on retrace into FVG or OB
5. SL below/above the sweep low/high
6. TP at next liquidity level

### Setup 2: VWAP Reclaim + Volume (For Indices)
**Instruments**: US100, US30
**Timeframes**: M15 bias → M5 entry
**Win Rate Target**: 65%+
**R:R Target**: 2:1+

**Rules**:
1. Price below VWAP for 2+ hours
2. Strong volume spike on reclaim above VWAP
3. Entry on first pullback to VWAP
4. SL below VWAP
5. TP at session high/low

### Setup 3: Bollinger Squeeze Breakout (For Trending Markets)
**Instruments**: US100, GBP/USD, USD/JPY
**Timeframes**: H4 setup → M15 entry
**Win Rate Target**: 60%+
**R:R Target**: 2.5:1+

**Rules**:
1. BB contracts inside Keltner Channel (squeeze)
2. Squeeze lasts 10+ bars
3. Breakout candle with volume confirmation
4. Entry on breakout direction
5. SL at squeeze midpoint
6. TP at 2x the squeeze width

### Setup 4: RSI + MACD Double Divergence (For Reversals)
**Instruments**: EUR/USD, GBP/USD, USD/JPY
**Timeframes**: H4 setup → M15 entry
**Win Rate Target**: 65%+
**R:R Target**: 2:1+

**Rules**:
1. Price makes new high/low
2. RSI makes opposite high/low (divergence)
3. MACD histogram confirms divergence
4. Entry on price action confirmation (engulfing, pin bar)
5. SL beyond the divergence point
6. TP at next S/R level

### Setup 5: ORB + Price Action Confirmation (Opening Range Breakout)
**Instruments**: US100, US30, EUR/USD
**Timeframes**: M5 entry, M15 bias
**Win Rate Target**: 60%+
**R:R Target**: 2:1+

**Rules**:
1. Define opening range: First 15/30/60 min of session (configurable)
2. Mark OR high and OR low
3. Wait for breakout above OR high (long) or below OR low (short)
4. Confirmation: Volume spike + bullish/bearish engulfing OR RSI confirmation
5. Entry on breakout candle close
6. SL at opposite side of OR (or midpoint for tighter risk)
7. TP at 1.5x OR range, 2x OR range, or next S/R level
8. No entry if breakout happens in first 5 min (false break)
9. No entry if OR range is too small (< min ATR * 0.3) or too large (> ATR * 2)

**Key Insight**: ORB works best on indices (US100, US30) during US market open. The opening range captures institutional positioning. Combining with volume + candle confirmation filters false breakouts.

### Setup 6: Ichimoku + Supertrend MTF (For Strong Trends)
**Instruments**: USD/JPY, US100, GBP/USD
**Timeframes**: Daily bias → H4/H1 entry
**Win Rate Target**: 65%+
**R:R Target**: 2:1+

**Rules**:
1. Daily Ichimoku shows clear trend (price above/below cloud)
2. H4 Supertrend confirms direction
3. Entry on pullback to Supertrend line
4. SL below Supertrend
5. TP at next resistance/support

---

## INSTRUMENT-SPECIFIC CONFIGS

### US100 (Nasdaq 100)
```
Sessions: US (9:30-16:00 ET)
Best Hours: 9:30-11:00 AM ET (opening drive)
Pip Value: 0.01 (1 point = $0.01)
Min ATR(14) M15: 20 points
Max Spread: 3 points
Risk per Trade: 1%
Setups: VWAP Reclaim, BB Squeeze, Trend Following
```

### US30 (Dow Jones)
```
Sessions: US (9:30-16:00 ET)
Best Hours: 9:30-11:00 AM ET
Pip Value: 0.01 (1 point = $0.01)
Min ATR(14) M15: 15 points
Max Spread: 3 points
Risk per Trade: 1%
Setups: Range Trading, News Trading, VWAP Bounce
```

### EUR/USD
```
Sessions: London (06-12), NY (13-21)
Best Hours: 13:00-16:00 UTC (overlap)
Pip Value: 0.0001 (1 pip = $0.0001)
Min ATR(14) M15: 5 pips
Max Spread: 1.0 pips
Risk per Trade: 2%
Setups: Liquidity Sweep + MSS + FVG, Mean Reversion
```

### GBP/USD
```
Sessions: London (06-12), NY (13-21)
Best Hours: 08:00-11:00 UTC (London open)
Pip Value: 0.0001 (1 pip = $0.0001)
Min ATR(14) M15: 6 pips
Max Spread: 2.0 pips
Risk per Trade: 1.5%
Setups: Trend Following, Momentum, Liquidity Sweep
```

### USD/JPY
```
Sessions: Tokyo (00-08), London (06-12), NY (13-21)
Best Hours: 00:00-02:00 UTC (Tokyo), 13:00-15:00 UTC (NY)
Pip Value: 0.01 (1 pip = $0.01)
Min ATR(14) M15: 8 pips
Max Spread: 1.5 pips
Risk per Trade: 1.5%
Setups: Carry Trade, Trend Following, Ichimoku + Supertrend
```

### XAUUSD (Gold) — Deprioritized
```
Sessions: London (06-12), NY (13-21)
Best Hours: 13:00-16:00 UTC
Pip Value: 0.10 (1 pip = $0.10)
Min ATR(14) M15: 40 pips
Max Spread: 0.5 pips
Risk per Trade: 1%
Setups: ICT/SMC (existing system), Regime Adaptive
```

---

## IMPLEMENTATION PHASES

### Phase 1: Data Pipeline (Week 1)
- [ ] Multi-asset data fetcher (MT5 + yfinance)
- [ ] Unified OHLCV format
- [ ] Per-instrument config loader
- [ ] Data quality checks

### Phase 2: A1 Setup Modules (Week 2-3)
- [ ] Liquidity Sweep + MSS + FVG module
- [ ] VWAP Reclaim module
- [ ] BB Squeeze module
- [ ] RSI + MACD Divergence module
- [ ] Ichimoku + Supertrend module

### Phase 3: Signal Quality Gate (Week 3-4)
- [ ] Multi-factor A1 scoring (simplified from 15 to 7 factors)
- [ ] Session timing per instrument
- [ ] Spread filter per instrument
- [ ] Anti-clustering per instrument

### Phase 4: Backtesting (Week 4-5)
- [ ] VectorBT for fast parameter sweeps
- [ ] Walk-forward validation (3 periods)
- [ ] Per-instrument performance metrics
- [ ] Portfolio-level backtest

### Phase 5: Live Paper Trading (Week 5-6)
- [ ] Multi-instrument signal generation
- [ ] Telegram alerts per instrument
- [ ] Dashboard with multi-instrument view
- [ ] Performance tracking

---

## SUCCESS CRITERIA

| Metric | Target | Minimum |
|--------|--------|---------|
| Monthly PnL | 4,000+ pips | 2,000 pips |
| Win Rate | 60%+ | 55% |
| R:R | 2.5:1+ | 2:1 |
| Profit Factor | 2.0+ | 1.5 |
| Max Drawdown | <15% | <20% |
| Trades/Day | 3-8 total | 2-3 total |
| Instruments | 3-5 active | 2 active |

---

## KEY DIFFERENCES FROM XAUUSD SYSTEM

| Aspect | Old (XAUUSD) | New (Multi-Asset A1) |
|--------|-------------|---------------------|
| Instruments | 1 (XAUUSD) | 5-6 |
| Indicators | 27 modules | 5-7 per instrument |
| Confluence | 15 factors | 7 factors (simplified) |
| Session | Fixed hours | Per-instrument hours |
| SL | Structural levels | ATR-based + structural |
| TP | Trailing only | Fixed TP + trailing hybrid |
| Risk | 3% flat | Per-instrument (1-2%) |
| Backtest | Single run | Walk-forward 3 periods |

---

*Plan created from 5 parallel research agents + web search + FTMO interviews*

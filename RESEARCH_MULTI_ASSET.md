# Multi-Asset A1 Strategy Research
> Compiled: 2026-06-29 | Branch: multi-asset-a1

---

## 1. CURRENT SYSTEM AUDIT

### What We Use (27 Modules)
- **Structure**: Swing Points, BOS, CHoCH, EMA Filter, Layer1, Regime (Hurst+ADX+Chop), Premium/Discount, OTE, Liquidity Pools, Asian Range, Cascade (H4+H1)
- **Entry**: Order Blocks, FVG, BPR, Breaker Blocks, IFVG, Unicorn, Rejection Blocks, VWAP, RSI, MACD, M5 Confirmation, M5 Alignment
- **Strategy**: Trend Structure, Breakout Retest, Liquidity Sweep, Voting (2-of-3)
- **Filters**: Session, Anti-Clustering, Spread, Layer3 (Volume+Body), Confluence Score (15 factors), R:R Gate
- **Risk**: Structural SL, BE at 2R, Trailing from 3R (step 1R), Stale exit 100 bars, Max bars 500

### Current Performance
| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Monthly PnL | ~825 pips | 4,000+ pips | 4.8x short |
| Signal Frequency | ~1-2/day | 5-10/day | Need more |
| Instruments | XAUUSD only | US100, US30, forex | Multi-asset |
| Win Rate | 42% | 55%+ | Need higher |
| R:R | 2.17 | 2.5+ | Close |

### What Works
- Combined voting (2/3 modules): 62.5% WR, 3.76 PF on H4
- Confluence scoring (15 factors)
- Anti-clustering
- Regime detection (Hurst+ADX+Chop)
- Break-even + trailing protection

### What Doesn't Work
- Too few signals (1-2/day on M15)
- XAUUSD only (limited opportunities)
- VWAP computed but not used as trigger
- RSI/MACD only in confluence (not entry triggers)
- Liquidity pools computed but not gated
- Asian range computed but not gated
- No spread filter actually applied
- No walk-forward validation

---

## 2. INSTRUMENT COMPARISON

| Characteristic | US100 | US30 | EUR/USD | GBP/USD | USD/JPY | AUD/USD |
|---|---|---|---|---|---|---|
| **Avg Daily Range** | 250-400 pts | 100-300 pts | 70-90 pips | 80-120 pips | 60-100 pips | 50-70 pips |
| **ATR(14) M15** | 30-60 pts | 20-50 pts | 5-10 pips | 6-12 pips | 8-15 pips | 4-8 pips |
| **Spread** | 1-3 pts | 1-3 pts | 0.0-1.0 pips | 1-2 pips | 0.5-1.5 pips | 1.0-2.0 pips |
| **Best Session** | US (9:30-16 ET) | US (9:30-16 ET) | London-NY overlap | London-NY | Tokyo+NY | Asian+London |
| **Trend vs Mean-Rev** | Strong trending | Moderate trending | Mean-reverting | Trending | Strong trending | Range-bound |
| **Liquidity** | Very High | High | Highest | High | Very High | Moderate |

### Best Instruments by Strategy
| Strategy | Best Instruments | Reason |
|---|---|---|
| Scalping | EUR/USD, USD/JPY | Tightest spreads, highest liquidity |
| Trend Following | US100, USD/JPY, GBP/USD | Strong trending characteristics |
| Mean Reversion | EUR/USD, USD/CAD | Range-bound behavior |
| Breakout | US100, GBP/USD | Volatile, news-driven |

---

## 3. TOP 20 A1 SETUPS (Ranked by Quality)

### Tier 1: Highest Conviction (WR >60%, R:R >2:1)

1. **Liquidity Sweep + FVG + OB Confluence** — WR 55-75%, R:R 2-3:1
   - Sweep liquidity pool → MSS/BOS on LTF → Enter at FVG/OB retest
   - Best: EUR/USD, GBP/USD, NAS100, M15/M5

2. **VWAP + Volume Profile Triple Combo** — WR 65-78%, R:R 1.5-2.5:1
   - Anchored VWAP + High Volume Node + Swing level convergence
   - Best: NQ, ES, liquid large-caps, 5M-15M

3. **Bollinger Band Squeeze (TTM Squeeze)** — WR 59-65%, R:R 2-3:1
   - BB contraction inside Keltner → breakout candle entry
   - Best: BTC, NQ, indices, 4H/Daily

4. **VWAP Reclaim** — WR 59-73%, R:R 1:2-1:3
   - Price below VWAP → reclaim above with volume
   - Best: NQ, 2x ETFs, 5M-15M

### Tier 2: Strong Edge (WR 50-65%, R:R >1.5:1)

5. **Gartley Harmonic Pattern** — WR 52-70%, R:R 1.8-2:1
6. **Ichimoku + Supertrend MTF** — WR 60-68%, R:R 1.5-2:1
7. **RSI + MACD Double Divergence** — WR 63-75%, R:R 2:1+
8. **Absorption Fade (Order Flow)** — WR 70-80%, R:R 1.5-2.5:1
9. **Keltner Channel + MACD** — WR 51-77%, R:R 2-2.5:1
10. **Bat Harmonic Pattern** — WR 64-67%, R:R 2.5-2.7:1

### Tier 3: Reliable Systems (WR 45-60%, R:R >2:1)

11. Triple Supertrend + EMA + ADX — WR 50-60%
12. MACD + RSI + 200 SMA Triple — WR 58-62%
13. Supertrend ATR Trailing — WR 42-67%, R:R 3-4:1
14. VWAP Bounce + Volume — WR 62-75%
15. Liquidity Sweep + CHoCH Reversal — WR 55-70%

### Tier 4: Conditional Edge

16. CCI + RSI + Stochastic Triple — WR 52-73%
17. RSI + BB Combo Mean Reversion — WR 43-53%
18. Footprint Stacked Imbalances — WR 75-85%
19. Dark Pool Accumulation — WR 55-65%
20. Butterfly Harmonic — WR 53-78%

---

## 4. PROFITABLE TRADER INSIGHTS

### Key Data Points
- **Specialists earn 52x MORE** than multi-instrument traders (43% of payouts from 6% of population)
- **Winning traders use 2.7 indicators** vs losing traders use 8.3
- **1-2 trades per day** is optimal for most profitable traders
- **Minimum 2:1 R:R** (ideally 3:1) is the standard
- **Liquidity Sweep + MSS + FVG** is the most consistently profitable setup

### What Profitable Traders Actually Do
1. Trade 1-2 instruments only
2. Use 1-2 indicators + price action
3. Wait for A+ setups only (1-5 trades/day)
4. Trade same session daily
5. Backtest 10+ years of data
6. Journal every trade
7. Stop after 1-2 losses per session
8. Risk 0.25-1% per trade
9. Daily loss limit: 3-5% max

### Timeframes
- **Forex scalping**: M5 execution, M15/H1 for bias
- **Index scalping**: M1/M5 entries, M15 for bias, 9:30-11:00 AM ET optimal
- **Trailing vs Fixed**: Scalping prefers fixed TP, intraday/swing prefers trailing

---

## 5. TECH STACK RECOMMENDATIONS

| Component | Best Choice | Why |
|-----------|-------------|-----|
| Indicators | pandas-ta (192+) + TA-Lib (speed) | Most comprehensive |
| Backtesting | VectorBT PRO (research) + Backtrader (validation) | 40x faster research |
| Data: Forex/Gold | MT5 (headless Docker) | Real forex data |
| Data: US100/US30 | yfinance (`^NDX`, `^DJI`) | Free, unlimited |
| ML | XGBoost + LightGBM + CatBoost ensemble | Best accuracy |
| Feature Selection | SHAP + VarianceThreshold | Explainable |
| Risk | Quarter Kelly + Volatility-adjusted sizing | Conservative |

---

## 6. CRITICAL LESSONS FROM XAUUSD FAILURE

1. **XAUUSD is too volatile** for our M15 system — needs wider stops, fewer signals
2. **1 instrument is not enough** — need 5-6 instruments for consistent opportunities
3. **We had too many indicators** (27 modules) — profitable traders use 2-3
4. **Confluence scoring was compressed** (14-point range) — couldn't differentiate good vs bad
5. **No walk-forward validation** — single-run backtests are unreliable
6. **PIP_VALUE confusion** cost us weeks — always validate with real broker data

---

*Research compiled from: 5 parallel research agents, web search, FTMO interviews, Reddit, YouTube, TradingView community*

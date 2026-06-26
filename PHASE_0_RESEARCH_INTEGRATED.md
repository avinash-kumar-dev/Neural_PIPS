# PHASE 0: INTEGRATED RESEARCH — XAU/USD Scalping Signal Engine

*Synthesized from: primary research (web, academic papers, quant blogs), secondary research from AI agent analysis.*

---

## 1. XAU/USD Behavioral Foundations

### 1.1 Correlation Matrix

| Driver | Correlation | Timeframe Strength | Notes |
|--------|-----------|-------------------|-------|
| DXY | **-0.80 to -0.95** | Strongest on M5/M15 during NY | DXY leads gold by 1-3 candles on M1/M5. This is the #1 alpha source. |
| US10Y Real Yield | **-0.70 to -0.85** | Strongest on H1+ | Real yield = DGS10 minus breakeven inflation. 30% weight in institutional gold models. |
| US2Y Rate Outlook | **~ -0.60** | Medium | Affects short-term rate expectations; less impactful for scalping. |
| VIX | **+0.30 to +0.60** | Spikes during risk-off | Correlation breaks during crisis — both DXY and gold can rise together. |
| COMEX Gold (GC) | **~ +0.99** | Instantaneous | Same instrument — futures-to-spot arb keeps them aligned. |

**Critical insight**: DXY + US10Y Real Yield = **55% combined weight** in institutional macro scoring for gold (TradingView Gold Macro Score methodology).

### 1.2 Session Behavior

| Session | UTC | Avg Range | Spread | Scalping Verdict |
|---------|-----|-----------|--------|-----------------|
| Asian | 22:00-06:00 | 10-18 pips | 0.25-0.40 | **AVOID** — low volatility, liquidity pools form (hunted by London) |
| London Open | 06:00-08:00 | 15-35 pips | 0.20-0.30 | **MODERATE** — first volatility spike, but choppy |
| London | 08:00-13:00 | 30-50 pips | 0.18-0.28 | **GOOD** — steady directional movement |
| **London/NY Overlap** | **13:00-17:00** | **40-80+ pips** | **0.15-0.25** | **BEST — primary scalping window. ~60-65% of all meaningful directional moves.** |
| NY | 13:00-21:00 | 30-60 pips | 0.15-0.25 | GOOD — DXY moves sharpest here |
| Post-NY | 21:00-22:00 | 5-10 pips | 0.30-0.50 | **AVOID** — low liquidity, erratic fills |

### 1.3 Scalping Economics (Pip Reality)

| Parameter | Conservative | Standard | Aggressive |
|-----------|-------------|----------|------------|
| Take Profit | 8 pips | 10 pips | 15 pips |
| Stop Loss | 5 pips | 6 pips | 8 pips |
| Risk:Reward | 1:1.6 | 1:1.67 | 1:1.88 |
| Break-even win rate | 38.5% | 37.5% | 34.8% |
| Spread cost | 0.20 pips | 0.20 pips | 0.20 pips |
| Slippage (entry+exit) | 1.5 pips | 1.5 pips | 1.5 pips |
| **Requirement for profit** | >50% accuracy | >50% accuracy | >50% accuracy |

**Conclusion**: With 10-pip TP and 6-pip SL, even 55-60% directional accuracy is profitable. The 90% target is for HIGH-CONFIDENCE signals only (LONG/SHORT excluding NO-TRADE).

---

## 2. Multi-Timeframe Confirmation Architecture

This is NOT a single-timeframe system. Every signal requires hierarchical confirmation across 4 timeframes.

### 2.1 The Hierarchy

```
H1  ─── Trend Direction (higher timeframe anchor)
 │        EMA 50/200 slope, DXY H1 trend, macro bias
 │
M15 ─── Market Structure / Pullback Detection
 │        Swing highs/lows, order blocks, FVG zones
 │
M5  ─── Confirmation Timeframe (primary model input)
 │        XGBoost ternary prediction at M5 resolution
 │
M1  ─── Entry Timing (secondary model input)
          XGBoost ternary prediction at M1 resolution
          Spread check, slippage buffer
```

### 2.2 Confirmation Rules

**H1 Trend Filter** — Cannot trade against H1 trend:
- H1 uptrend (price above EMA 50, higher highs) → Only LONG or NO-TRADE
- H1 downtrend (price below EMA 50, lower lows) → Only SHORT or NO-TRADE
- H1 ranging (EMA 50 flat, equal swing highs/lows) → Both directions allowed

**M15 Structure Filter** — Must be in alignment zone:
- LONG: Price above M15 VWAP, pulling back to M15 order block / FVG
- SHORT: Price below M15 VWAP, rallying to M15 order block / FVG
- No structure alignment → NO-TRADE regardless of M1/M5 model output

**M5 + M1 Model Agreement** — Both models must agree:
- M5 predicts LONG with >75% confidence AND M1 predicts LONG with >85% confidence → SIGNAL
- M5 predicts SHORT with >75% confidence AND M1 predicts SHORT with >85% confidence → SIGNAL
- Any disagreement → NO-TRADE

### 2.3 Session Gates

- **Only signal during**: London Session (08:00-17:00 UTC) + NY/London Overlap (13:00-17:00 UTC)
- **Block signals**: 15 min before through 15 min after any high-impact USD economic event
- **Volatility gate**: Block if ATR(14) ratio (current / rolling mean 20) < 0.5 (dead market) or > 2.5 (news spike)

---

## 3. Feature Set — Ranked by Alpha

### 3.1 Tier 1: Highest Alpha (Build First)

| # | Feature | Alpha Justification | Implementation |
|---|---------|-------------------|----------------|
| 1 | **DXY Lag 1-3 Candle Returns** | DXY leads gold 1-3 M1 candles. Single highest-alpha feature. | `DXY_return[t-1]`, `DXY_return[t-2]`, `DXY_return[t-3]` as direct model inputs. Pull DXY from MT5 (if broker carries symbol) or FRED API (daily). |
| 2 | **DXY Rate of Change (3-period)** | Momentum of DXY leads gold momentum. | `DXY_ROC_3 = (close[t] - close[t-3]) / close[t-3]` |
| 3 | **US10Y Real Yield Change** | 30% weight in institutional gold models. Opportunity cost of holding gold. | `DGS10_change[t] - T10YIE_change[t]`. FRED API daily. For intraday, use MT5 US10Y if available. |
| 4 | **CVD Divergence Signal** | Price making new high but CVD lower high = bearish divergence (65-75% reversal probability). | Compute CVD per candle: `Σ(tick_volume * sign(price_change))`. Then compare CVD trend vs price trend over last 5 candles. |
| 5 | **FVG Presence + Size** | 12.8% feature importance in SMC+XGBoost model. XAU/USD fills FVGs ~78-83% in same session. | Detect 3-candle FVG pattern. Record gap midpoint and size relative to ATR. Feature: distance from price to nearest unfilled FVG. |
| 6 | **VWAP Distance (Normalized)** | Price reverts to session VWAP. ~65-70% mean reversion probability at 1+ std dev. | `(close - session_VWAP) / ATR`. Session VWAP anchored at London open (06:00 UTC). |
| 7 | **Session Cyclical Encoding** | Market microstructure varies dramatically by hour. Model needs to know time context. | `sin(2π * minute_of_day / 1440)`, `cos(...)`. Binary flags: `is_london_overlap`, `is_news_window`. |

### 3.2 Tier 2: Supporting Features

| Feature | Importance | Notes |
|---------|-----------|-------|
| Liquidity Pool Proximity | MEDIUM | Equal highs/lows within 0.5 pip tolerance over 50 candles. Feature: distance to nearest pool. |
| BOS/CHoCH Detection | MEDIUM | Break of Structure / Change of Character over last 20 candles. Label: 1=CHoCH, 0.5=BOS, 0=none. |
| ATR(14) Ratio | **CRITICAL FILTER** | `current_ATR / rolling_mean(ATR, 20)`. If >2.5 (spike) or <0.5 (dead) → NO-TRADE. |
| EMA 8/21/50 Slope (M5, M15, H1) | MEDIUM | Trend direction filter per timeframe. |
| RSI(14) on M1 and M5 | MEDIUM | 11.5% feature importance in SMC+XGBoost. |
| Stochastic RSI (5,3,3) | LOW-MEDIUM | Fast scalping setting. Good for overextended detection. |
| Bollinger Band Position | MEDIUM | Is price above upper BB (overextended) or below lower BB? |
| Candle Body/Wick Ratio | LOW | Doji = indecision, long body = conviction. Minor feature. |
| Volume / Avg Volume (20) | MEDIUM | Volume spike confirms breakout; fade = exhaustion. |

### 3.3 Tier 3: Macro Context Features

| Feature | Purpose | Source |
|---------|---------|--------|
| High-impact news in next 60 min | **Suppress all signals** ±15 min around release | Forex Calendar Pro API / ForexFactory JSON |
| DXY H1 Trend (8/21 EMA) | Is DXY trending up or down? Longer context for lead-lag. | Computed from DXY data |
| US10Y H1 Trend (8/21 EMA) | Is yield rising (bearish gold) or falling (bullish gold)? | Computed from yield data |
| Session type | Asian/London/NY/Overlap/Post-NY | Derived from UTC time |

---

## 4. Data Stack — $0 Solution

| Data Type | Primary Source | Backup | Cost |
|-----------|---------------|--------|------|
| M1 OHLCV | **MT5** `mt5.copy_rates_range()` | Dukascopy JForex | **FREE** |
| M5 OHLCV | **MT5** (aggregated from M1) | Same | **FREE** |
| Tick Data (CVD) | **MT5** `mt5.copy_ticks_range()` | Dukascopy tick CSV | **FREE** |
| DXY (intraday) | **MT5** (if broker offers DXY symbol) | Scrape TradingView / FRED (daily) | **FREE** |
| DXY (daily backfill) | **FRED API** `DTWEXBGS` | — | **FREE** |
| US10Y Yield | **FRED API** `DGS10` | — | **FREE** |
| US2Y Yield | **FRED API** `DGS2` | — | **FREE** |
| 10Y Breakeven Inf. | **FRED API** `T10YIE` | — | **FREE** |
| Economic Calendar | **ForexFactory JSON** `nfs.faireconomy.media/ff_calendar_thisweek.json` | Finnhub | **FREE** |
| **Total** | | | **$0.00** |

**Broker recommendation**: IC Markets or Pepperstone raw/ECN accounts. Both offer XAU/USD with raw spreads, MT5 access, and often carry DXY as a CFD symbol.

---

## 5. Model Architecture Decision

### 5.1 Problem Framing

| Approach | Recommendation | Rationale |
|----------|---------------|-----------|
| **Triple-Barrier Labeling** | **USE** | Lopez de Prado method. For each M1 candle, look forward N candles. Label LONG if price hits +TP first, SHORT if -SL first, NO-TRADE if neither. Eliminates forward-looking ambiguity. |
| **3-Class Classification** | **USE** | LONG / SHORT / NO-TRADE. NO-TRADE is not a cop-out — it's the mechanism for >90% precision on actionable signals. |
| **Selective Abstention** | **USE** | Based on Chalkidis & Savani (2021): selective classifiers dramatically outperform non-selective ones. Set confidence threshold high; accept 2-15% coverage. |
| **Meta-Labeling** | **Optional** | First model predicts direction, second model predicts whether the trade will succeed. Adds complexity but can improve precision by ~3-5%. |

### 5.2 Recommended Model Stack

```
                 ┌─────────────────────────┐
                 │   M1 XGBoost (primary)   │
                 │ 3-class, multi:softprob  │
                 │ n_est=500, max_depth=4   │
                 │ lr=0.05, subsample=0.8   │
                 └──────────┬──────────────┘
                            │
                 ┌──────────▼──────────────┐
                 │   M5 XGBoost (secondary) │
                 │ Same architecture,       │
                 │ different lookback       │
                 └──────────┬──────────────┘
                            │
                 ┌──────────▼──────────────┐
                 │  Agreement Gate          │
                 │ Both models agree on     │
                 │ direction AND confidence │
                 │ > threshold → SIGNAL     │
                 └──────────┬──────────────┘
                            │
                 ┌──────────▼──────────────┐
                 │  Macro/Risk Gates        │
                 │ Session, News, Volatility│
                 │ Spread, H1 trend filter  │
                 └──────────┬──────────────┘
                            │
                      FINAL SIGNAL
                  (LONG/SHORT/NO-TRADE)
```

### 5.3 LLM Agents / LangChain Assessment

**RESEARCH FINDING**: LLM agents (LangChain, LangGraph) are **NOT suitable for the core signal engine** on M1/M5 scalping. Evidence:

1. **Latency kills scalping**: LLM inference is 500ms-2s+. At M1, every millisecond counts. XGBoost inference is <1ms.
2. **Proven ineffectiveness on short timeframes**: The ElliottAgents paper explicitly states: *"short-term effectiveness is limited due to large minute-to-minute fluctuations and the presence of other high-frequency trading algorithms."*
3. **Chalkidis & Savani (2021)**: Best results on 30-min bars came from **logistic regression**, not deep models. LSTMs were **never profitable** at realistic slippage.

**Where LLM agents COULD add value (optional, Phase 2+)**:

| Use Case | Runs | Value |
|----------|------|-------|
| **Pre-session macro briefing** | Once per session (not per candle) | LLM reads FRED data, recent news, calendar → generates market regime classification (trending/ranging/volatile) |
| **Post-hoc signal explanation** | When signal fires | LLM explains which features drove the decision in natural language |
| **Risk adjustment** | Once per hour | LLM adjusts position sizing based on current macro conditions |
| **Model retraining trigger** | Daily | LLM monitors live accuracy drift; flags when retraining is needed |

**Recommendation**: Build the core signal engine with XGBoost only. No LangChain, no agent framework, no LLM in the signal path. The LLM is a thin "context layer" that runs infrequently and does NOT touch the real-time signal pipeline.

### 5.4 Overfitting Prevention (Critical)

**DO NOT USE standard K-Fold cross-validation.** It leaks future information and overestimates accuracy on financial time series.

| Method | Priority | Description |
|--------|----------|-------------|
| **Combinatorial Purged CV (CPCV)** | **MUST** | Multiple train/test paths with no temporal overlap. Lopez de Prado gold standard. |
| **Purged Walk-Forward CV** | **ACCEPTABLE** | Expanding window, 5-candle purge zone + 10-candle embargo. Default if CPCV is too complex for initial build. |
| Standard K-Fold | **NEVER** | Will give falsely high accuracy. Skip entirely. |

**Label overlap purging**: With triple-barrier labeling looking ahead N candles, any training sample whose N-candle forward window overlaps with a test sample must be PURGED from training data.

**Detection metrics**:
- Train-test accuracy gap > 10% → overfitted
- Precision unstable across folds → overfitted
- Backtest Sharpe > 4.0 → suspicious
- Model achieves >50% accuracy on shuffled labels → data leakage exists

---

## 6. Label Construction (Triple-Barrier Method)

### 6.1 Labeling Rules

For each candle at index `i`, look forward up to `horizon` candles:

```
Look forward up to 10 candles (M1) or 6 candles (M5)

For LONG label:
  - If price hits entry + TP_pips (10 pips M1, 12 pips M5) before hitting entry - SL_pips:
    → LABEL = 1 (LONG)
  - If price hits entry - SL_pips (6 pips M1, 8 pips M5) first:
    → LABEL = -1 (SHORT)
  - If neither within horizon:
    → LABEL = 0 (NO TRADE)
```

### 6.2 Expected Label Distribution

| Label | M1 (10-pip TP, 6-pip SL, 10-candle horizon) | M5 |
|-------|----------------------------------------------|-----|
| LONG | ~25-30% | ~25-30% |
| SHORT | ~25-30% | ~25-30% |
| NO-TRADE | ~40-50% | ~40-50% |

The model MUST learn the NO-TRADE class well. This is what separates great models from overfitted garbage.

---

## 7. Backtesting Protocol

### 7.1 Realistic Cost Injection

| Item | Value | Applied |
|------|-------|---------|
| Spread | 0.20 pips | Subtract from LONG entry, add to SHORT entry |
| Commission | $7/lot round-turn | ~$0.07 per 0.01 lot trade |
| Entry slippage | 1.0 pip | Use close of NEXT candle, not signal candle |
| Exit slippage | 0.5 pip | For TP/SL hits — add to SL distance, subtract from TP |
| **Total round-trip** | **~2.0-2.5 pips** | |

### 7.2 Validation Thresholds

| Metric | Target | Overfit Warning |
|--------|--------|-----------------|
| LONG Precision (selective) | >0.90 | >0.95 — suspicious |
| SHORT Precision (selective) | >0.90 | >0.95 — suspicious |
| Combined Recall | >0.15 (signal on 15%+ of candles) | — |
| Sharpe Ratio | >1.5 | >4.0 — too good |
| Max Drawdown | <15% | <3% — unrealistic |
| Win Rate (including costs) | >65% | >85% — check for look-ahead |
| Train-Test Accuracy Gap | <5% | >10% — overfitting |

### 7.3 Signal Gate Pipeline (Inference)

```
1. Is it a valid session? (08:00-17:00 UTC only)         → NO → NO-TRADE
2. Is ATR(14) ratio in [0.5, 2.5]?                        → NO → NO-TRADE
3. Is news event within ±15 min?                           → YES → NO-TRADE
4. Is spread < 0.30 pips?                                  → NO → NO-TRADE
5. Is H1 trend filter satisfied?                           → NO → NO-TRADE
6. Is M15 structure aligned?                               → NO → NO-TRADE
7. M5 model prediction (confidence > 0.757)                → NO → NO-TRADE
8. M1 model prediction (confidence > 0.85? OR > 0.90?)     → NO → NO-TRADE
9. Both models agree on direction?                         → NO → NO-TRADE
                                                              ↓
                                                     FINAL SIGNAL
```

---

## 8. Implementation Sequencing (Build Order)

| Phase | What | Duration | Dependencies |
|-------|------|----------|--------------|
| **1. Data Pipeline** | MT5 connection → pull 2-3 years M1 + DXY + tick data → store as Parquet. FRED API pull for yields. | — | MT5 broker account |
| **2. Feature Engineering** | Compute all Tier 1 features. Unit tests for FVG/CVD/BOS/VWAP detection. | — | Phase 1 |
| **3. Label Generation** | Triple-barrier labeling with purge/embargo logic. Plot label distribution. | — | Phase 2 |
| **4. XGBoost M1 Baseline** | Train on M1 with walk-forward CV. Measure selective precision. | — | Phase 3 |
| **5. XGBoost M5 Second Model** | Train M5 model. Implement agreement gate. | — | Phase 4 |
| **6. Macro Gates** | Session filter, news gate, volatility gate, H1 trend filter. | — | Phase 5 |
| **7. Backtest (Realistic Costs)** | Full backtest with spread/slippage/delay. Walk-forward CV. | — | Phase 6 |
| **8. Live Inference Loop** | MT5 subscription → feature compute → model inference → signal. | — | Phase 7 |

---

## 9. Summary: The Path to >90% Directional Accuracy

**This is not achieved through better models. It's achieved through extreme selectivity + correct features + proper validation.**

1. **Features**: DXY lag (1-3 candles) + US10Y yield + CVD divergence + FVG + VWAP distance + session timing. These 6 features carry the alpha. Everything else is support.

2. **Hierarchy**: H1 trend → M15 structure → M5 model → M1 model. Four layers of confirmation before a signal is fired. Any disagreement → NO-TRADE.

3. **Selectivity**: Train the model to say "I don't know" (NO-TRADE) for 80-95% of candles. The remaining 5-20% are where you achieve >90% precision.

4. **Validation**: CPCV or purged walk-forward. Never standard K-Fold. Realistic costs (2.0-2.5 pips round-trip). If it looks too good, it's overfitted.

5. **Architecture**: XGBoost is the right tool. No LangChain, no LLM agents in the signal path. LLM is optional for pre-session macro context only.

6. **Cost**: Zero-dollar data stack. MT5 + FRED API + free economic calendar JSON. No paid APIs needed.

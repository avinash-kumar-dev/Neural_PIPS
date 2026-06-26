# Phase 2: Research Findings & Analysis

## Date: 2026-06-26
## Context: Live session analysis, threshold investigation, architectural review

---

## Table of Contents

1. [Live Session Analysis (26 Jun 2026)](#1-live-session-analysis)
2. [Threshold Investigation (70 vs 78)](#2-threshold-investigation)
3. [Root Cause Analysis](#3-root-cause-analysis)
4. [Regime Detection Research](#4-regime-detection-research)
5. [ML Model Improvements Research](#5-ml-model-improvements-research)
6. [Signal Filtering & Risk Management Research](#6-signal-filtering--risk-management-research)
7. [Current System Flow Analysis](#7-current-system-flow-analysis)
8. [Unified vs Two-Stage ML Architecture](#8-unified-vs-two-stage-ml-architecture)
9. [Implementation Plan](#9-implementation-plan)

---

## 1. Live Session Analysis

### Session: 13:00-16:00 UTC, 26 Jun 2026

34 checks completed. Confidence range: 59-76. Peak: 76.75 at check #27 (15:22 UTC).

**All confidence scores:**
```
59, 59, 60, 62, 63, 63, 64, 64, 66, 66, 66, 66,
67, 67, 67, 68, 68, 68, 68, 68, 69, 69, 69, 69,
70, 70, 70, 70, 71, 71, 72, 72, 74, 76
```

**Checks above T=70:** 10 signals would have fired
- All were SHORT
- 7 hit SL (LOSS), 3 open/unresolved
- **Total P&L at T=70: -102.4 pips**
- **Total P&L at T=78: 0.0 pips (correctly filtered)**

### Check #27 Reconstruction (Confidence 76.75)

| Detail | Value |
|--------|-------|
| Direction | SHORT |
| Entry | 1.14019 |
| Confidence | 76.75 |
| ATR(14) | 10.0 pips |
| TP | 45.0 pips → 1.13569 |
| SL | 15.0 pips → 1.14169 |
| Ratio | 3.0:1 |
| Outcome | SL hit at bar 7 (35 min), **-15.0 pips** |
| Confidence breakdown | structure: 90, session: 90, trend: 85, volatility: 85, spread: 80, pattern: 65, momentum: 60, mtf: 50 |

**H4 was BULLISH during entire session:**
- 04:00 UTC: BULLISH (C=1.13770)
- 08:00 UTC: BULLISH (C=1.14013)
- 12:00 UTC: BULLISH (C=1.14272)

**H1 during session:**
- 13:00: BEAR (C=1.14048)
- 14:00: BEAR (C=1.13983)
- 15:00: BULL (C=1.14272) — big bullish candle

---

## 2. Threshold Investigation

### Architecture Doc Says 70 (4 places)
- Line 31: `Confidence threshold | >= 70/100`
- Line 1021: `Output: 0-100 score. Threshold >= 70 for signal emission.`
- Line 1161: `if confidence >= 70:`
- Line 1609: `min_confidence: 70  # 0-100`

### Research Doc Says 70 (5 places)
- Line 491: `Hard threshold on confidence score (recommend >= 70)`
- Line 615: `if confidence >= 70:`
- Line 750: `confidence > 0.70`
- Line 784: `Confidence Scoring: 8-factor weighted score with threshold >= 70`
- Line 822: `Confidence threshold >= 70`

### Config.yaml Was Written With 78 From First Commit
- No documented rationale for 78 vs 70
- Architecture and research both specify 70
- The discrepancy was never investigated

### V3 Holdout Data (June 2026)

| Threshold | SHORT signals/session | SHORT accuracy | In target range? |
|-----------|----------------------|----------------|------------------|
| T=70 | 5.25 | 96.2% | Yes (3-6 target) |
| T=78 | 1.25 | 92.0% | No (too few) |

### Signal Frequency Target (from research)
- 3-6 signals/day is optimal for 3:1 TP:SL
- T=70 produces 5.25 SHORT signals/session (in target)
- T=78 produces 1.25 SHORT signals/session (below target)

### Today's Reality
T=78 saved -102.4 pips. But the root cause is NOT the threshold — it's the model predicting SHORT against bullish H4.

---

## 3. Root Cause Analysis

### Root Cause 1: Wrong Prediction Target

**Current**: Model asks "which direction will price go?" (LONG vs SHORT)
**Should ask**: "Is this a good 3:1 setup?" (binary: trade vs no-trade)

From MQL5 scalping research (2026):
> "On M1/M5 chart, a 2-pip move is often just random order flow ('Brownian Motion'). The ML model mistakes this noise for a pattern (Overfitting)."

### Root Cause 2: H4 Regime Is Soft Feature, Not Hard Gate

The h4_regime feature contributes 10% to confidence score. XGBoost can override it with other features. Today it did — 10 times.

**Should be**: Hard binary gate. If H4 is bullish, SHORT is impossible. Period.

### Root Cause 3: Class Imbalance in Training

Triple-barrier labeling with asymmetric barriers (4.5x ATR TP vs 1.5x ATR SL):
- LONG trades need 3x the price movement to be labeled positive
- Creates fewer LONG labels, more SHORT labels
- Model learns SHORT bias from training data

### Root Cause 4: Live Script Bypasses Engine

`live_paper_trading.py` reimplements the pipeline instead of using `SignalEngine.generate_signal()`.
- Engine's built-in gates are dead code
- H4 check that exists in engine is never executed
- Inconsistencies between training and live behavior

### Root Cause 5: Confidence Scorer Doesn't Weight MTF Enough

Current weights:
- trend: 20%, momentum: 15%, volatility: 15%, structure: 15%
- pattern: 10%, mtf: 10%, session: 10%, spread: 5%

MTF alignment is only 10%. H4 regime alignment is buried in "mtf" factor.

---

## 4. Regime Detection Research

### How Profitable Systems Detect Regime

**Freqtrade (51.8k stars):**
- Three states: bull (full trading), bear (block entries), chop (minimal trading)
- Regime check in `confirm_trade_entry` (not vectorized)
- Position sizing by regime: bull: 1.0, chop: 0.4, bear: 0.15

**Microsoft Qlib (45.1k stars):**
- DDG-DA (Data Distribution Generation for Predictable Concept Drift Adaptation)
- Meta-learning: trains predictor to estimate future data distribution
- 11.3% improvement in signal-based metrics

**FinRL (15.5k stars):**
- Mahalanobis distance as turbulence/crisis detector
- When turbulence exceeds 99th percentile → cash/defensive positioning
- Fast risk-off (3-day shock detector) + slow regime (26-week trend + VIX)

### H4 Trend Definition Methods

**Method 1: Structure Breaks (Most Reliable)**
- H4 Bullish: Price making Higher Highs (HH) and Higher Lows (HL)
- H4 Bearish: Price making Lower Highs (LH) and Lower Lows (LL)
- H4 Neutral: Mixed structure

**Method 2: EMA Alignment (Most Quantifiable)**
- H4 Bullish: Price > EMA(50) > EMA(200) on H4, both EMAs sloping up
- H4 Bearish: Price < EMA(50) < EMA(200) on H4, both EMAs sloping down
- 21 EMA on H4 = "momentum line"

**Method 3: ADX + DI Direction**
- H4 Bullish: ADX > 25 AND +DI > -DI
- H4 Bearish: ADX > 25 AND -DI > +DI
- H4 Ranging: ADX < 20 (no trade)

### Anti-Correlation Filtering

**Research shows:**
- With-Trend win rate: 60-70%
- Counter-Trend win rate: 35-45%
- Adding H4 alignment improves win rate by 10-20 percentage points

**TechniTute EUR/USD analysis:** "60-70% win rate versus 45% when trading against the trend"
**Backtest 015 (inv.jp):** EUR/USD H4 trend following = PF 13.13, 68% win rate, 3414 pips over 7.5 years
**Backtest 008 (inv.jp):** Counter-trend on H4 = PF 1.37, 46.3% win rate, only 657 pips — marked "Usability: NO"

### MTF Alignment Impact

| Configuration | Signals/Session | Est. Win Rate | Quality |
|---------------|-----------------|---------------|---------|
| M5 only (no filters) | 15-25 | 45-50% | Low |
| H1 + M5 | 8-12 | 55-60% | Medium |
| H4 + H1 + M5 | 4-7 | 60-65% | High |
| **H4 + H1 + M15 + M5** | **2-4** | **65-70%+** | **Very High** |

---

## 5. ML Model Improvements Research

### Walk-Forward Validation Best Practices

| Parameter | Recommended | Current |
|-----------|-------------|---------|
| Training window | 6-12 months | 6 months |
| Test window | 1-3 months | 1 month |
| Walk-forward windows | 25-30 | 9 |
| Purge bars | 20 (= horizon) | 20 |
| Embargo bars | 20-40 | 20 |

### Overfit Detection

| Metric | Good | Bad |
|--------|------|-----|
| OOS/IS ratio | > 0.70 | < 0.50 |
| Degradation ratio | 10-30% | > 50% |
| Win rate stability | Consistent across windows | Large variance |
| Profit factor | > 1.0 in every window | Any negative window |

### Class Imbalance Handling

**Do NOT use SMOTE** — research says it usually doesn't help and likely makes things worse.

Instead:
- XGBoost: `scale_pos_weight=count_negative/count_positive`
- LightGBM: `is_unbalance=True`
- CatBoost: `auto_class_weights='Balanced'`

### Model Retraining

| Data Type | Recommended Frequency |
|-----------|----------------------|
| M5 forex | Weekly to bi-weekly |
| After regime change | Immediate |

### Drift Detection

| Method | Threshold | Action |
|--------|-----------|--------|
| PSI < 0.1 | No drift | Continue |
| PSI 0.1-0.25 | Moderate | Monitor |
| PSI > 0.25 | Significant | Retrain |
| Rolling accuracy drop > 15% | Concerning | Retrain |

### Feature Importance for EUR/USD

| Rank | Feature | Why |
|------|---------|-----|
| 1 | ATR(14) on M5 | Regime detection — #1 predictor |
| 2 | RSI(14) divergence | Strongest at extremes |
| 3 | Session volatility (lagged) | London vs Asian behavior |
| 4 | Price position within daily range | Mean-reversion vs trend trigger |
| 5 | Bollinger Band width | Expansion/contraction cycles |
| 6 | DXY direction (1H) | Direct inverse correlation |
| 7 | Volume vs session average | Smart money footprint |
| 8 | US-DE yield spread change | Interest rate differential |
| 9 | Fourier hour encoding (3 harmonics) | Bimodal intraday volatility |
| 10 | MACD histogram slope | Trend acceleration |

### Realistic Accuracy Benchmarks

| Source | Accuracy | Timeframe |
|--------|----------|-----------|
| Guyard & Deriaz (2024) | 58.52% | Daily |
| Balcıoğlu & Merter (2026) | 58.5% | Hourly |
| LSTM study (2026) | 65.4% signal accuracy | Daily |
| Practitioner (Sehat Dr) | 65-68% | M15-H1 |
| PipReaper (2026) | 54-57% | Various |

**The honest answer**: 58-60% is realistic ceiling for direction prediction on EUR/USD. 65%+ usually means overfitting.

---

## 6. Signal Filtering & Risk Management Research

### Signal Filtering Layers

| Layer | Timeframe | Purpose |
|-------|-----------|---------|
| Anchor/Directional | H4 | Trend direction (never trade against) |
| Confirmation | H1/M15 | Trend strength + momentum |
| Entry | M5 | Precision entry timing |

### Time-of-Day

- London-NY overlap (13:00-16:00 UTC) is canonical window
- ~38% of global daily forex volume during London session
- 50%+ of daily volume during LN overlap
- Spreads: 0.0-0.2 pips during overlap vs 1.5+ during Asian

### Spread/Volatility Filters

- Max spread: 0.5-1.0 pips for EUR/USD scalping
- Minimum ATR floor: skip entries when market too quiet
- ATR-based spread ratio: if spread > 0.5× ATR(14) on M5, negative expectancy

### Signal Cooldown & Anti-Clustering

| Method | Cooldown Period | Best For |
|--------|----------------|----------|
| Time-based | 5-15 minutes | High-frequency scalping |
| Bar-based | 3-10 M5 bars | Preventing chasing entries |
| Loss-based | 10-20 bars after loss | Preventing revenge trading |
| Position-based | 1 trade per symbol at a time | Risk containment |

### Maximum Consecutive Loss Protection

- 3+ consecutive losses → activate cooldown
- Daily loss kill switch at 85% of daily limit
- Horizon Scalper: "after losing trade stops trading for N bars (default 10)"

### TP/SL Optimization

**ATR-based multipliers (proven):**
| Component | ATR Multiplier | Notes |
|-----------|---------------|-------|
| SL (scalping) | 1.5× ATR(14) on M5 | Tight but respects noise |
| TP (scalping) | 4.5× ATR(14) on M5 | Your 3:1 ratio |

**Trailing stop approaches:**
1. Chandelier Exit: `Stop = Highest Close - (ATR × 3)`
2. Breakeven at +1× ATR: move SL to entry + spread
3. Tiered: +1× ATR → breakeven, +2× ATR → trail at 50-EMA on H4

### Proven Accuracy Benchmarks

**Realistic win rates for EUR/USD scalping:**
| Metric | Range |
|--------|-------|
| Scalping win rate | 55-65% |
| With filters | 60-70% |
| Counter-trend | 35-45% |
| Minimum for profitability | 55%+ |

**Best MQL5 signal providers:**
| Signal | Win Rate | PF | Trades |
|--------|----------|-----|--------|
| TrP EURUSD | 74.65% | 2.19 | 2,572 |
| ScalpingAlgo | 70.87% | 4.25 | 333 |
| Hybrid EUR USD | 81.41% | 3.28 | 113 |

**Key insight**: Best performers show 70-75% win rate with PF 2.0+ over 2000+ trades. This is the realistic ceiling.

---

## 7. Current System Flow Analysis

### ML Model Output

```python
def _predict(self, features_row, model, meta_model):
    feat_cols = [c for c in features_row.index if c != 'time']
    X = features_row[feat_cols].values.reshape(1, -1).astype(float)
    base_preds = np.column_stack([m.predict_proba(X)[:, 1] for m in model])
    meta_pred = np.clip(meta_model.predict(base_preds), 0, 1)
    return int(meta_pred[0] > 0.5), float(meta_pred[0])
```

**Returns**: `ml_pred` (0 or 1), `ml_confidence` (0.0-1.0)

### Signal Engine Flow

```
Features → _predict() → ml_pred (0 or 1)
    │
    ├─ ml_pred == 0 → NO-TRADE (in engine.generate_signal)
    │
    └─ ml_pred == 1 → Confidence Scorer → confidence >= 78? → Signal
```

### Live Script Flow (DIFFERENT from engine!)

```python
ml_pred, ml_conf = engine._predict(feat_series, models, meta_model)
direction = 'LONG' if ml_pred == 1 else 'SHORT'  # <-- KEY DIFFERENCE

confidence, scores = scorer.score(feat_series, raw_series)
if confidence < engine.confidence_threshold:
    return None, f'low_confidence_{confidence:.0f}'
```

**The live script treats ml_pred=0 as SHORT and generates SHORT signals.**
**The engine treats ml_pred=0 as "model rejected" and returns NO-TRADE.**

This is a code inconsistency — the live script bypasses the engine's `generate_signal()` method.

### Critical Bug

The live script fetches H4 data (`h4_df = fetcher.fetch_ohlcv('H4', 50)`) but NEVER uses it for any H4 regime check. It's passed to feature engineering but not used as a hard filter.

---

## 8. Unified vs Two-Stage ML Architecture

### Research Findings

| Approach | Used By | Verdict |
|----------|---------|---------|
| Single model, probability = quality | FreqAI, Qlib, most MQL5 EAs | Simplest, most proven |
| Multi-class (LONG/SHORT/NO-TRADE) | Mellish thesis (82.6%), Nguyen et al. | Viable, captures NO-TRADE |
| Two-stage meta-labeling | Lopez de Prado | Overkill for us |
| Ensemble disagreement as quality | NBER paper, Falcon AI | Best quality signal |

### Single Model With Probability-As-Quality

When XGBoost outputs `P(LONG) = 0.7`:
- **Direction**: LONG (because P > 0.5)
- **Quality**: 0.7 (because 70% of similar historical setups resulted in profitable longs)

**One number, two meanings.** No need for two models.

### Why Single Model Is Better for Our Case

| Factor | Single Model | Two-Stage |
|--------|-------------|-----------|
| Complexity | 1 training loop | 2 training loops |
| Overfitting risk | Lower | Higher |
| Deployment | 1 model to serve | 2 models to serve |
| Retraining | 1x effort | 2x effort |
| Proven? | Yes (FreqAI, Qlib) | Yes (Lopez de Prado) |

### Meta-Labeling Is Overkill Because

From QuantConnect analysis (2023):
> "This technique can only improve the performance of existing discretionary trading models but cannot improve the performance of a main ML model trained end-to-end on the same data."

Meta-labeling is designed for when you have a white-box signal generator to layer ML on top of. We're building from scratch — single model is simpler and equally effective.

### Recommended Architecture

```
Step 1: Train XGBoost + LightGBM + CatBoost
        Each outputs P(UP) via predict_proba()

Step 2: Calibrate probabilities (Platt scaling)
        Fix overconfident outputs

Step 3: Soft voting ensemble
        mean_prob = average of all model P(UP)
        direction = LONG if mean_prob > 0.5, else SHORT

Step 4: Ensemble agreement = quality signal
        agreement = 1 - std(model_probabilities)
        If models disagree → NO TRADE

Step 5: Confidence threshold
        If mean_prob < 0.65 → NO TRADE (not confident enough)

Step 6: Rule filters (H4 regime, spread, session)
        Hard gates that must pass

Step 7: TP/SL calculation
        ATR-based 3:1 ratio
```

---

## 9. Implementation Plan

### Phase 1: Fix the Prediction Problem (Most Critical)

1. **H4 Regime Gate (hard filter)** — computed from H4 candles, not a feature:
   - H4 bullish: only LONG allowed
   - H4 bearish: only SHORT allowed
   - H4 neutral: no trade
   - Goes BEFORE ML model, not after

2. **Fix class imbalance** — add `scale_pos_weight` to XGBoost and `is_unbalance=True` to LightGBM

3. **Fix the live script** — use `SignalEngine.generate_signal()` instead of reimplementing

### Phase 2: Retrain the Model

1. Recompute labels — check LONG/SHORT distribution
2. Add class weights to training
3. Increase walk-forward windows from 9 to 25-30
4. Verify purge/embargo are correct (20 bars each)

### Phase 3: Rewrite Confidence Scorer

**New weights:**
| Factor | Current | New |
|--------|---------|-----|
| H4 regime alignment | implicit | 25% |
| H1 confirming H4 | implicit | 15% |
| M15 structure aligned | implicit | 10% |
| Trend (ADX) | 20% | 15% |
| Momentum | 15% | 10% |
| Volatility | 15% | 10% |
| Session | 10% | 5% |
| Spread | 5% | 5% |
| Pattern | 10% | 5% |

### Phase 4: Signal Cooldown (Anti-Clustering)

- Minimum 3 bars (15 min) between signals in same direction
- 10 bars (50 min) cooldown after a loss
- Max 3-5 signals per session
- Max 3 consecutive losses → halt

### Phase 5: Long-Term — Probability Calibration

- Add Platt scaling to fix overconfident model outputs
- Add ensemble agreement as quality signal
- Add PSI drift detection for retraining triggers

---

## Realistic Expectations After Fixes

| Metric | Current | After Fixes |
|--------|---------|-------------|
| Win rate (with-trend) | unknown | 60-70% |
| Win rate (against-trend) | 0% | 0% (blocked) |
| Signals per session | 0 | 2-4 |
| Monthly signals | ~25 | 40-80 |
| Profit factor | unknown | 1.5-2.5 |
| Max drawdown | -102 pips (1 day) | <15% monthly |

The realistic ceiling is 58-65% accuracy with 3:1 TP:SL. At 35% win rate you're already profitable. The H4 gate alone would have prevented today's loss.

---

*Research compiled: 2026-06-26*
*Sources: Freqtrade (51.8k stars), Microsoft Qlib (45.1k stars), FinRL (15.5k stars), VectorBT (7.7k stars), Backtrader (22.1k stars), MQL5, ForexFactory, academic papers, prop firm requirements*

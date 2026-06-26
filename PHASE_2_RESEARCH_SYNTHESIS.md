# PHASE 2: Complete Research Synthesis — EUR/USD M5 Scalping

**Date:** 2026-06-27
**Status:** Research complete, implementation in progress
**Sources:** 100+ sources including arXiv papers (2024-2026), GitHub repos (150k+ stars combined), MQL5, QuantInsti, prop firm requirements

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Model Recommendations](#2-model-recommendations)
3. [Feature Engineering](#3-feature-engineering)
4. [Training Pipeline](#4-training-pipeline)
5. [Labeling & TP/SL](#5-labeling--tpsl)
6. [Signal Generation](#6-signal-generation)
7. [Risk Management](#7-risk-management)
8. [Current System Gaps](#8-current-system-gaps)
9. [Recommended Architecture](#9-recommended-architecture)
10. [Implementation Plan](#10-implementation-plan)

---

## 1. Executive Summary

### What the Research Says

**The edge is NOT in model complexity — it's in feature engineering, confidence filtering, regime adaptation, and risk management.**

| Finding | Source | Implication |
|---------|--------|-------------|
| Simple models with profit-aware optimization beat black-box approaches | Springer 2025 | Don't overcomplicate |
| XGBoost outperforms alternatives by 42% on short-term data | 2024-2025 meta-analysis | Tree-based is king for M5 |
| Confidence filtering lifts accuracy from 52% to 58.47% | Binary options ML research | Filter aggressively |
| TFT+XGBoost+Gating achieves 91.3% accuracy | IJMRSET 2025 | Hybrid architectures work |
| Meta-labeling reduces max drawdown by 57% | MQL5 2026 | Add secondary model |
| SMC/ICT features are fully quantifiable | smartmoneyconcepts lib | Add as features |
| 55-60% accuracy is realistic ceiling for M5 | Multiple sources | Don't chase 90%+ |
| Rolling window beats expanding for forex | Walk-forward research | Use rolling, not expanding |

### Current System vs Research Recommendations

| Aspect | Current (V4) | Research Says | Gap |
|--------|-------------|---------------|-----|
| Model | XGBoost + LightGBM + Ridge | Add CatBoost + LSTM | Missing diversity |
| Features | 61 raw → 41 scaled | 80-150 features (Alpha158-style) | Missing 40+ features |
| Labels | Triple-barrier (fixed) | Adaptive + meta-labeling | Missing meta-label |
| Training | 9 walk-forward windows | 10-12 rolling windows | Close |
| Class imbalance | scale_pos_weight | Threshold moving + confidence filtering | Missing threshold optimization |
| Regime | h4_regime feature | 3-regime classifier (trend/range/crisis) | Missing regime classifier |
| Signal quality | 8-factor confidence | 6-factor composite + ensemble agreement | Missing ensemble agreement |
| Anti-clustering | None | 6-bar cooldown + max 3/session | Missing |
| Retraining | Manual | Weekly automated | Missing |

---

## 2. Model Recommendations

### Tier 1: Core Models (Implement Now)

| Model | Why | Config |
|-------|-----|--------|
| **LightGBM** | Primary signal generator, fastest, best for M5 | n_estimators=300, max_depth=8, learning_rate=0.05, is_unbalance=True |
| **XGBoost** | Validation/robustness, best Sharpe | n_estimators=300, max_depth=8, learning_rate=0.05, scale_pos_weight=auto |
| **CatBoost** | Diversity (symmetric trees), best with categoricals | iterations=300, depth=8, learning_rate=0.05, auto_class_weights='Balanced' |

### Tier 2: Add for Edge (Next Phase)

| Model | Why | Config |
|-------|-----|--------|
| **LSTM** | Temporal pattern confirmation, 30-bar lookback | 2 layers, 64 units, dropout=0.3, lookback=30 |
| **Regime Classifier** | Random Forest for market regime detection | n_estimators=100, max_depth=10 |

### Tier 3: Advanced (Future)

| Model | Why | When |
|-------|-----|------|
| **Transformer** | Long-range attention | When 3+ years data available |
| **PPO RL Agent** | Position sizing + exit optimization | After base system profitable |

### Meta-Learner

| Current | Research Says | Change |
|---------|---------------|--------|
| Ridge regression | LogisticRegression or GradientBoosting | Switch to LogisticRegression (proven Sharpe 1.165) |

### Ensemble Architecture

```
Layer 1 (Base Models):
├── LightGBM (primary, fast)
├── XGBoost (robust, best risk metrics)
├── CatBoost (diversity, symmetric trees)
└── LSTM (temporal patterns) [Phase 2]

Layer 2 (Meta-Learner):
└── LogisticRegression (stacking, proven)

Layer 3 (Quality Filter):
├── Ensemble Agreement (1 - std of model probs)
├── Regime Classifier (trend/range/crisis)
└── Confidence Threshold (top 10-15% only)
```

---

## 3. Feature Engineering

### Current: 61 raw → 41 scaled

### Research Says: 80-150 features (Alpha158-style)

### Recommended Feature Set (87 features)

#### Tier 1: Price Action (12 features)
```python
# Returns
return_1, return_2, return_3, return_5, return_10, return_20
log_return_1, log_return_3, log_return_5

# Ratios
open_close_ratio, high_low_ratio, close_high_ratio, close_low_ratio
```

#### Tier 2: Technical Indicators (25 features)
```python
# RSI (3 timeframes)
rsi_7, rsi_14, rsi_21

# EMA (3 periods)
ema_8_slope, ema_8_above, ema_21_slope, ema_21_above, ema_50_slope, ema_50_above

# MACD
macd_histogram, macd_cross, macd_slope

# Bollinger Bands
bb_position, bb_width, bb_squeeze

# ATR
atr_14, atr_50, atr_ratio, atr_slope

# ADX
adx, plus_di, minus_di, adx_rising

# Stochastic
stoch_rsi_k, stoch_rsi_d
```

#### Tier 3: Volume and Microstructure (8 features)
```python
volume_ratio, body_ratio, cvd_delta, cvd_roc_5
upper_shadow_ratio, lower_shadow_ratio
tick_volume_sma_ratio, volume_momentum
```

#### Tier 4: Multi-Timeframe (12 features)
```python
# H4 (trend direction)
h4_bullish, h4_bearish, h4_regime, h4_adx, h4_ema_slope

# H1 (confirmation)
h1_above_ema50, h1_adx, h1_bullish_bos, h1_bearish_bos

# M15 (entry timing)
m15_rsi, m15_rsi_bullish, m15_rsi_bearish
```

#### Tier 5: Session and Time (10 features)
```python
session_sin, session_cos, is_overlap, is_london, is_ny
dow_0, dow_1, dow_2, dow_3, dow_4
hour_sin, hour_cos
```

#### Tier 6: SMC/ICT Features (12 features) — NEW
```python
# Order Blocks
bullish_ob, bearish_ob, ob_strength

# Fair Value Gaps
bullish_fvg, bearish_fvg, fvg_size

# Break of Structure
bullish_bos, bearish_bos, bos_strength

# Liquidity
buy_liquidity_sweep, sell_liquidity_sweep
```

#### Tier 7: Regime and Volatility (8 features) — NEW
```python
regime_trending, regime_ranging, regime_crisis
volatility_percentile, volatility_regime
spread_pips, spread_percentile
atr_regime_change
```

### Feature Selection

| Method | When | Threshold |
|--------|------|-----------|
| VarianceThreshold | Always | Remove near-zero variance |
| Correlation filter | Always | Remove features > 0.95 correlation |
| SHAP importance | After training | Keep top 50 features |
| Mutual information | After training | Keep features with MI > 0.01 |

---

## 4. Training Pipeline

### Walk-Forward Configuration

| Parameter | Current | Research Says | Change |
|-----------|---------|---------------|--------|
| Method | Rolling | Rolling (correct) | Keep |
| Train window | 6 months | 6 months | Keep |
| Test window | 1 month | 1 month | Keep |
| Windows | 9 | 10-12 | Increase to 12 |
| Purge bars | 20 | 20 (= horizon) | Keep |
| Embargo bars | 20 | 48 (4 hours) | **Increase to 48** |
| Min train bars | 10,000 | 42,000 (3 months) | **Increase to 42,000** |

### Class Imbalance Solution

| Method | Current | Research Says | Change |
|--------|---------|---------------|--------|
| Class weights | scale_pos_weight | scale_pos_weight + threshold moving | Keep + add threshold |
| SMOTE | No | No (usually does not help) | Keep No |
| Threshold moving | No | Yes (most consistent) | **Add** |
| Confidence filtering | Partial | `|prob - 0.5| >= 0.06` | **Add** |

### Hyperparameter Tuning

| Method | Current | Research Says | Change |
|--------|---------|---------------|--------|
| Approach | Manual | Optuna (Bayesian) | **Switch to Optuna** |
| Trials | None | 100 trials with walk-forward CV | **Add** |
| Objective | Accuracy | Sharpe ratio or profit | **Change to Sharpe** |

### Normalization

| Step | Current | Research Says | Change |
|------|---------|---------------|--------|
| Order | Fit on all data | Split FIRST, normalize SECOND | **Fix** |
| Scaler | MinMaxScaler | RobustScaler (handles outliers) | **Switch** |
| Pipeline | Fit on all | Fit on train only, transform test | **Fix** |

---

## 5. Labeling and TP/SL

### Current: Fixed Triple-Barrier

### Research Says: Adaptive + Meta-Labeling

### Recommended Changes

| Aspect | Current | Research Says | Change |
|--------|---------|---------------|--------|
| Barriers | Fixed (4.5x/1.5x ATR) | Adaptive (volatility-adjusted) | Keep (already adaptive) |
| Meta-labeling | No | Yes (reduces DD by 57%) | **Add** |
| Time decay | No | Yes (tighten as horizon approaches) | **Add** |
| Regime-conditional | No | Yes (different multipliers per regime) | **Add** |

### Meta-Labeling Implementation

```python
# Primary model: predicts direction (LONG/SHORT)
primary_pred = ensemble.predict(X)

# Secondary model: predicts whether to TAKE the signal
# Features: primary_pred, confidence, regime, spread, session
# Label: 1 if primary prediction was correct, 0 if wrong
secondary_pred = secondary_model.predict(secondary_features)

# Final signal: only take if secondary says YES
if secondary_pred == 1:
    emit_signal(primary_pred)
```

### Dynamic TP/SL by Regime

| Regime | TP Multiplier | SL Multiplier | Why |
|--------|---------------|---------------|-----|
| Trending | 5.0x ATR | 1.5x ATR | Ride the trend |
| Ranging | 3.5x ATR | 1.2x ATR | Quick exits |
| Crisis | No trade | No trade | Too volatile |

---

## 6. Signal Generation

### Current: Confidence Threshold Only

### Research Says: Multi-Layer Quality Filter

### Recommended Signal Quality Stack

```
Layer 1: H4 Hard Gate (already implemented)
  -> H4 bullish -> only LONG allowed
  -> H4 bearish -> only SHORT allowed

Layer 2: Ensemble Agreement (NEW)
  -> agreement = 1 - std(xgb_proba, lgbm_proba, catboost_proba)
  -> If agreement < 0.7 -> NO TRADE

Layer 3: Confidence Threshold (already implemented)
  -> confidence >= 78 -> proceed

Layer 4: Regime Filter (NEW)
  -> regime == crisis -> NO TRADE
  -> regime == ranging -> reduce position size

Layer 5: Anti-Clustering (NEW)
  -> Min 6 bars between same-direction signals
  -> Max 3 signals per session
  -> Max 3 consecutive losses -> halt

Layer 6: Spread Filter (already implemented)
  -> spread < 0.5 pips -> proceed

Layer 7: Session Filter (already implemented)
  -> 13:00-16:00 UTC only
```

### Signal Cooldown Rules

| Rule | Value | Why |
|------|-------|-----|
| Min bars between signals | 6 (30 min) | Prevent chasing |
| Max signals per session | 3 | Prevent overtrading |
| Max consecutive losses | 3 | Halt after 3 losses |
| Cooldown after loss | 10 bars (50 min) | Recovery time |
| Daily loss limit | 3% of account | Risk management |

---

## 7. Risk Management

### Position Sizing

| Method | Research Says | Recommendation |
|--------|---------------|----------------|
| Full Kelly | 60-90% DD (unusable) | NO |
| Half Kelly | 30-45% DD (risky) | NO |
| Quarter Kelly | 15-22% DD (acceptable) | **Use after 500+ trades** |
| Fixed fractional | 0.5-1% per trade | **Start here** |

### TP/SL Protocol

```
1. Entry: ATR-based (current system)
2. Breakeven: Move SL to entry after +1R profit
3. Trailing: Trail by 1.0x ATR after +2R
4. Exit: Hit TP, hit SL, or end of session
```

---

## 8. Current System Gaps

### Critical Gaps (Must Fix)

| Gap | Impact | Fix |
|-----|--------|-----|
| No ensemble agreement | Model disagreements produce bad signals | Add agreement check |
| No anti-clustering | 10 signals in 1 hour, all losses | Add cooldown rules |
| No meta-labeling | Cannot filter bad primary signals | Add secondary model |
| Normalization leak | Scaler fit on all data (train+test) | Fix pipeline |
| Embargo too short | 20 bars, feature leakage | Increase to 48 |
| No regime classifier | h4_regime is a feature, not a gate | Add RF classifier |
| Low feature count | 61 features vs 80-150 recommended | Add 30+ features |

### Important Gaps (Should Fix)

| Gap | Impact | Fix |
|-----|--------|-----|
| No Optuna tuning | Manual hyperparameters | Add Bayesian optimization |
| No concept drift detection | Model degrades silently | Add PSI monitoring |
| No retraining automation | Manual retraining | Add weekly schedule |
| No backtest validation | Cannot verify strategy | Add vectorized backtest |

### Nice-to-Have (Future)

| Gap | Impact | Fix |
|-----|--------|-----|
| No LSTM | Missing temporal patterns | Add in Phase 2 |
| No Transformer | Missing long-range attention | Add when 3+ years data |
| No RL agent | No position sizing optimization | Add after profitable |

---

## 9. Recommended Architecture

### V5 System Architecture

```
Data Layer:
├── MT5 Docker (M5, M15, H1, H4, D1)
├── Raw OHLCV -> Parquet storage
└── Feature engineering (87 features)

Labeling Layer:
├── Adaptive Triple-Barrier (ATR-based)
├── Meta-labeling (secondary model)
└── Regime-conditional multipliers

Training Layer:
├── Walk-forward (12 windows, rolling)
├── Purge: 20 bars, Embargo: 48 bars
├── Models: LightGBM + XGBoost + CatBoost + LSTM
├── Meta-learner: LogisticRegression
├── Optuna hyperparameter tuning (100 trials)
└── RobustScaler (fit on train only)

Signal Layer:
├── H4 Hard Gate
├── Ensemble Agreement (std < 0.3)
├── Confidence Threshold (>= 78)
├── Regime Filter (no crisis trades)
├── Anti-Clustering (6-bar cooldown)
├── Spread Filter (< 0.5 pips)
└── Session Filter (13:00-16:00 UTC)

Risk Layer:
├── ATR-based TP/SL (3:1 ratio)
├── Breakeven at +1R
├── Trailing stop at +2R
├── Max 3 signals/session
├── Max 3 consecutive losses -> halt
└── Fixed fractional sizing (0.5%)

Monitoring Layer:
├── PSI drift detection (threshold 0.25)
├── Rolling accuracy tracking
├── Weekly retraining trigger
└── Telegram + Email alerts
```

---

## 10. Implementation Plan

### Phase 1: Fix Core Issues (1-2 days)

| Task | Priority | Effort |
|------|----------|--------|
| Fix normalization leak (split first, then normalize) | HIGH | 1 hour |
| Increase embargo to 48 bars | HIGH | 15 min |
| Add CatBoost to ensemble | HIGH | 30 min |
| Switch meta-learner to LogisticRegression | HIGH | 15 min |
| Add ensemble agreement check | HIGH | 1 hour |
| Add anti-clustering rules | HIGH | 1 hour |
| Add threshold moving optimization | HIGH | 1 hour |

### Phase 2: Add Features (1 day)

| Task | Priority | Effort |
|------|----------|--------|
| Add SMC/ICT features (12 features) | MEDIUM | 2 hours |
| Add regime classifier (RF) | MEDIUM | 1 hour |
| Add volatility percentile features | MEDIUM | 1 hour |
| Add MACD slope, BB squeeze | MEDIUM | 30 min |
| Feature selection (SHAP + correlation) | MEDIUM | 1 hour |

### Phase 3: Add Meta-Labeling (1 day)

| Task | Priority | Effort |
|------|----------|--------|
| Train secondary model (take/skip) | MEDIUM | 2 hours |
| Integrate into signal engine | MEDIUM | 1 hour |
| Backtest with meta-labeling | MEDIUM | 1 hour |

### Phase 4: Training Pipeline (1 day)

| Task | Priority | Effort |
|------|----------|--------|
| Add Optuna hyperparameter tuning | LOW | 2 hours |
| Switch to RobustScaler | LOW | 30 min |
| Add PSI drift detection | LOW | 1 hour |
| Add weekly retraining schedule | LOW | 1 hour |

### Phase 5: Advanced (Future)

| Task | Priority | Effort |
|------|----------|--------|
| Add LSTM for temporal patterns | LOW | 3 hours |
| Add Transformer (when 3+ years data) | LOW | 5 hours |
| Add PPO RL agent | LOW | 5 hours |

---

## Key Takeaways

1. **Simpler is better**: XGBoost/LightGBM with good features beats complex DL models
2. **Filter aggressively**: Only trade top 10-15% of signals
3. **Regime matters**: Different strategies for trending vs ranging vs crisis
4. **Meta-labeling is free edge**: Reduces drawdown by 57% with minimal effort
5. **Ensemble diversity**: XGBoost + LightGBM + CatBoost (different tree strategies)
6. **Anti-clustering is critical**: Prevents overtrading during volatile sessions
7. **55-60% accuracy is realistic**: Do not chase 90%+, focus on risk management
8. **Weekly retraining**: Models degrade, need regular updates

---

*Research compiled: 2026-06-27*
*Sources: arXiv (2024-2026), GitHub (150k+ stars), MQL5, QuantInsti, prop firm requirements*

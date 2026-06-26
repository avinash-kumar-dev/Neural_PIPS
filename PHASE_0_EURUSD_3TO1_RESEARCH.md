# PHASE_0: EUR/USD M5 Scalping — Feature Engineering, Training & Data Prep Research

**Date:** 2026-06-27
**Focus:** EUR/USD M5 scalping prediction with 3:1 TP:SL
**Sources:** 40+ web sources including academic papers, Qlib docs, trading research, ML best practices

---

## 1. Feature Engineering for Forex M5 Scalping

### 1.1 What Actually Works on M5

Based on multiple research sources and live trading systems, the most predictive features for EUR/USD M5 scalping fall into these categories:

#### A. Price Action Features (Highest Signal)
- **Candle body ratio**: `(close-open)/(high-low)` — directional conviction within the bar
- **Upper/lower shadow ratios**: Selling/buying pressure indicators
- **Price position**: `(2*close-high-low)/(high-low)` — where close sits in the range
- **N-bar momentum**: Returns over 1, 2, 3, 5, 10, 15, 20 M5 bars
- **Rate of change**: `Ref(close, d)/close` at multiple lookbacks
- **Gap features**: Open relative to previous close

#### B. Trend Features (Medium Signal, High Filter Value)
- **EMA crossovers**: 8/21 EMA on M5 (most cited for scalping), 5/13 for faster signals
- **EMA position relative to price**: Binary features for above/below
- **Multi-TF trend alignment**: H1 and H4 EMA direction as binary filters
- **Linear regression slope**: Over 20-30 bars, indicates trend strength
- **Supertrend direction**: (7, 2.0) setting for M5

#### C. Momentum/Oscillator Features
- **RSI(14)**: Classic, but use compressed periods (7-10) for M5 scalping
- **Stochastic %K**: (14, 3, 3) or faster (5, 3, 3) for M5
- **MACD histogram**: (5, 13, 1) for scalping — faster than standard (12, 26, 9)
- **CCI(20)**: Commodity Channel Index for overbought/oversold
- **Williams %R(14)**: Alternative to RSI
- **Momentum(10)**: Price difference over 10 bars

#### D. Volatility Features
- **ATR(14)**: Critical for TP/SL sizing — use 1.5x ATR for SL, 4.5x for TP
- **Bollinger Band width**: Squeeze detection (period 20, dev 2.0)
- **BB position**: Where price sits within the bands (0-1 scale)
- **Rolling standard deviation**: Over 10, 20, 50 bars
- **Historical volatility ratio**: Short-term vs long-term vol

#### E. Volume/Flow Features
- **Volume-weighted average price (VWAP)**: Deviation from VWAP
- **On-balance volume (OBV)**: Cumulative volume flow
- **Volume momentum**: Volume change rate
- **Buy/sell pressure**: Volume on up-bars vs down-bars ratio

#### F. Session/Time Features (Critical for Forex)
- **Forex session flags**: Asian (00:00-08:00 UTC), London (08:00-16:00 UTC), NY (13:00-21:00 UTC)
- **Overlap flag**: London-NY overlap (13:00-16:00 UTC) — highest liquidity
- **Hour-of-day sin/cos encoding**: Fourier harmonics for cyclical time
- **Session volatility**: Rolling std of returns within each session
- **Days to month-end**: Institutional rebalancing effects

### 1.2 Features That Don't Work on M5
- Slow moving averages (SMA 50, 100, 200) — too lagging for scalping
- Daily pivot points — not granular enough for M5
- Long-horizon momentum (60+ bars) — noise dominates
- Raw price levels without normalization — scales meaningless

---

## 2. Alpha158 Feature Set from Qlib

### 2.1 Structure
Alpha158 contains **158 pre-built technical factors** organized into 7 categories:

| Category | Count | Key Features | EUR/USD Relevance |
|----------|-------|--------------|-------------------|
| K-line patterns | 9 | KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2 | **HIGH** — candle structure |
| Price features | 4 | OPEN0, HIGH0, LOW0, VWAP0 | **MEDIUM** — relative positions |
| Momentum/ROC | 29 | ROC over 5,10,20,30,60 windows | **HIGH** — rate of change |
| Moving averages | 29 | SMA/EMA ratios at multiple periods | **MEDIUM** — trend detection |
| Volatility | 29 | STD of returns, BB width, ATR | **HIGH** — risk measurement |
| Volume | 29 | VSUMP, VSUMD, OBV ratios | **LOW** for EUR/USD (limited volume) |
| Correlation/Regression | 29 | Rolling correlations, regression slopes | **MEDIUM** — pattern detection |

### 2.2 Most Important Alpha158 Features for EUR/USD
From research and feature importance analysis:

1. **ROC_5, ROC_10, ROC_20** — Short-term momentum is king for M5
2. **STD_5, STD_10** — Volatility clustering matters
3. **KMID2, KSFT2** — Candle structure within range
4. **MA_5/close, MA_10/close** — Short-term mean reversion
5. **VWAP deviation** — Institutional reference price

### 2.3 Warning
Alpha158 is public (15K+ GitHub stars). Everyone uses these 158 factors → **alpha erosion**. Use as baseline, then build custom factors that others don't have.

---

## 3. Feature Selection Methods

### 3.1 Recommended Pipeline for EUR/USD

```
Step 1: VarianceThreshold (threshold=0.01)
  → Remove near-constant features

Step 2: Correlation filter (|r| > 0.95)
  → Remove redundant features

Step 3: Mutual Information (mutual_info_classif)
  → Rank features by MI score, keep top 30-50

Step 4: Model-based importance (XGBoost/LightGBM)
  → SHAP values for final selection

Step 5: Stability check
  → Verify feature importance is stable across walk-forward windows
```

### 3.2 VarianceThreshold
- Remove features with variance < 0.01 (normalized scale)
- For forex, this removes features that don't move meaningfully
- Scikit-learn: `VarianceThreshold(threshold=0.01)`

### 3.3 Mutual Information
- **Model-agnostic**: Works regardless of final model choice
- **Captures non-linear relationships**: Unlike correlation
- **Handles mixed data types**: Works with both continuous and categorical
- Use `mutual_info_classif()` for classification targets
- Retain features with MI > median + 1 std

### 3.4 SHAP-Based Selection
- Train XGBoost, compute SHAP values
- Rank features by mean |SHAP value|
- Use iterative removal (shap-select framework)
- Keep features with statistically significant SHAP contributions

### 3.5 Key Research Finding
From "Forex forecasting: The critical role of feature selection" (2025):
> "Prediction accuracy does NOT correlate linearly with feature quantity."
> More features ≠ better. Optimal is usually 15-40 features for forex ML.

---

## 4. Walk-Forward Validation Best Practices

### 4.1 Recommended Configuration for EUR/USD M5

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Train window | 6 months (~25,000 M5 bars) | Enough data for pattern learning |
| Test window | 1 month (~4,200 M5 bars) | Enough trades for statistical significance |
| Purge period | 20 bars (100 min) | Prevents label overlap |
| Embargo period | 20 bars (100 min) | Prevents contamination after test |
| Number of windows | 30 | ~2.5 years of rolling validation |
| Window type | Rolling (not anchored) | More realistic for regime changes |

### 4.2 Purge and Embargo Rules
- **Purge**: Remove training samples whose forward-looking labels overlap test period
- **Embargo**: Block samples immediately after test window (label contamination)
- **Horizon-aware purge**: If labels look forward by N bars, purge last N training samples
- For M5 with 20-bar horizon: purge = 20 bars minimum

### 4.3 Walk-Forward Efficiency
- Calculate OOS performance / IS performance ratio
- Ratio close to 1.0 = robust strategy
- Ratio < 0.5 = overfitting likely
- Target: WF efficiency > 0.6

### 4.4 Common Mistakes
- Using anchored (expanding) windows — inflates performance with old data
- Not including transaction costs in both IS and OOS
- Optimizing more than sqrt(N) parameters (for 500 observations: <22 params)
- Too few trades in test window (minimum 20-30)

---

## 5. Class Imbalance in Forex

### 5.1 The Problem
EUR/USD with 3:1 TP:SL produces naturally imbalanced labels:
- Typically: ~50-60% SHORT, ~40-50% LONG, or worse
- Sometimes: 85% SHORT vs 15% LONG (market regime dependent)
- NO-TRADE labels further complicate distribution

### 5.2 Recommended Solutions (Priority Order)

1. **Class weights** (Try first — computationally free)
   - `class_weight='balanced'` in sklearn
   - XGBoost: `scale_pos_weight = neg_count/pos_count`
   - LightGBM: `is_unbalance=True`
   - No synthetic data, no distribution distortion

2. **SMOTE** (If class weights insufficient)
   - Generate synthetic minority samples
   - Use only on training data, never on test
   - k_neighbors=5 default works for forex
   - **SMOTE-ENN** hybrid for better quality

3. **Threshold adjustment** (Post-training)
   - Instead of 0.5 threshold, optimize for F1 or profit
   - Use precision-recall curve to find optimal threshold

4. **Focal loss** (If using neural networks)
   - Down-weights easy examples, focuses on hard ones
   - γ=2.0, α=0.25 works well

### 5.3 Evaluation Metrics for Imbalanced Forex
- **DO NOT use accuracy** — misleading with imbalance
- Use: F1-score, AUC-PR, AUC-ROC, Balanced Accuracy
- For trading: Use risk-adjusted return as final metric
- Track: Precision at fixed recall levels (e.g., recall@10%)

---

## 6. Data Normalization

### 6.1 Scaler Comparison for Forex Features

| Scaler | How It Works | Best For | Forex Suitability |
|--------|-------------|----------|-------------------|
| **RobustScaler** | Median ± IQR | Outlier-heavy data | **BEST for forex** — price spikes common |
| **StandardScaler** | Mean ± Std | Gaussian-like features | Good for normalized returns |
| **MinMaxScaler** | Min-Max to [0,1] | Bounded features | Avoid — outlier-sensitive |
| **MaxAbsScaler** | Max absolute value | Sparse data | Alternative for returns |

### 6.2 Recommended Pipeline
```
1. Returns-based features → StandardScaler (returns are roughly Gaussian)
2. Indicator features (RSI, etc.) → RobustScaler (bounded but with outliers)
3. Ratio features → MinMaxScaler (already bounded)
4. Price-based features → Convert to returns first, then StandardScaler
```

### 6.3 Critical Rules
- **Fit on training data only**, transform test data with training scaler
- **Never refit scaler on production data**
- **Save scaler with model** using joblib
- **Winsorize extreme values BEFORE scaling** (clip at ±5σ)

### 6.4 Production Consideration
For tree-based models (XGBoost, LightGBM, CatBoost): **scaling is optional** — trees are scale-invariant. But scaling helps:
- Feature importance comparison
- Regularization effectiveness
- Neural network convergence (if used in ensemble)

---

## 7. Feature Importance for EUR/USD

### 7.1 Most Predictive Indicators (Research-Backed)

**Tier 1 — Highest Predictive Power:**
1. **ATR(14)** — Volatility is the strongest predictor for scalping exits
2. **RSI(7-10)** — Fast RSI for momentum timing
3. **EMA(8)/EMA(21) cross** — Trend direction
4. **Candle body ratio** — Intraday conviction
5. **Volume momentum** — Participation confirmation

**Tier 2 — Strong Supporting Features:**
6. **BB width squeeze** — Volatility expansion precursor
7. **MACD histogram** — Momentum direction change
8. **Price relative to VWAP** — Institutional reference
9. **Session overlap flag** — Liquidity context
10. **Stochastic %K** — Overbought/oversold timing

**Tier 3 — Weak Alone, Useful in Combination:**
11. CCI(20)
12. Williams %R
13. Linear regression slope
14. Rolling correlation features
15. Calendar effects (month-end, Friday close)

### 7.2 Key Finding from Research
From "Transitioning from static to temporal ML models" (2026):
> "Oil price dynamics, geopolitical uncertainty, and short-term memory are the most relevant predictors for EUR/USD."

From the EUR/USD ML framework paper:
> "LSTM achieved R²=0.9234 for price prediction, but macroeconomic indicators showed limited explanatory power for forecast errors — models learn from price patterns, not fundamentals."

**Implication**: For M5 scalping, focus on price-derived features, not macro data.

---

## 8. Multi-Timeframe Features

### 8.1 Proper MTF Architecture for M5 Scalping

```
Entry Timeframe: M5 (signal generation)
Confirmation:    M15 (trend filter)
Context:         H1 (structure)
Hard Filter:     H4 (never trade against)
```

### 8.2 How to Combine MTF Features

**Method 1: Feature Concatenation**
```python
# M5 features: RSI, EMA, ATR, candle pattern, momentum
# M15 features: RSI, EMA, trend direction
# H1 features: trend direction, key levels
# H4 features: trend direction (binary: up/down)
# Concatenate all into one feature vector
```

**Method 2: Higher-TF as Filters**
```python
if H4_trend == "down" and signal == "LONG":
    reject_signal()
if H1_trend == "down" and signal == "LONG":
    reduce_confidence()
```

**Method 3: Hierarchical Features**
```python
# Compute indicators on each timeframe
# Then compute CROSS-timeframe features:
mtf_rsi_divergence = RSI_M5 - RSI_H1
mtf_ema_alignment = EMA_M5_direction == EMA_H1_direction
mtf_vol_ratio = ATR_M5 / ATR_H1
```

### 8.3 Recommended Approach for ML
Use **Method 1 + Method 2**:
- Include M15, H1, H4 indicators as input features (Method 1)
- Add binary H4 trend alignment as a hard filter (Method 2)
- The ML model learns the interaction between timeframes

### 8.4 Timeframe Separation Rule
Use timeframes separated by factor of 4-6x:
- M5 → M15 (3x) → H1 (4x) → H4 (4x)
- This ensures each timeframe adds distinct information

---

## 9. Lookback Window Optimization

### 9.1 Recommended Lookback Windows for M5

| Feature Type | Lookback | Rationale |
|-------------|----------|-----------|
| Momentum/ROC | 5, 10, 15, 20 bars | 25-100 minutes |
| Volatility (ATR) | 14 bars | 70 minutes |
| Moving averages | 8, 21, 50 bars | 40-250 minutes |
| RSI | 7-14 bars | 35-70 minutes |
| Bollinger Bands | 20 bars | 100 minutes |
| Session volatility | Full session bars | ~48-72 bars per session |
| Trend filters (H1) | 20-50 bars | 20-50 hours |
| Trend filters (H4) | 10-20 bars | 40-80 hours |

### 9.2 Optimization Warning
**DO NOT optimize lookback periods** on the same data used for model training — this introduces data snooping bias. Instead:

1. Use standard lookbacks (14 for ATR, 20 for BB, etc.)
2. Verify strategy works across a RANGE of lookback values
3. If results differ significantly with different lookbacks → overfitting
4. Fix lookback periods before walk-forward testing

### 9.3 Adaptive Lookback (Advanced)
Use volatility-adaptive lookbacks:
- Low volatility → longer lookback (more smoothing)
- High volatility → shorter lookback (faster reaction)
- Formula: `adaptive_period = base_period * (avg_vol / current_vol)`

---

## 10. Outlier Detection

### 10.1 Outlier Types in Forex Data
- **Data errors**: Bad ticks, feed glitches (extreme prices)
- **News spikes**: NFP, FOMC, ECB — legitimate but extreme
- **Flash crashes**: Real but rare events
- **Weekend gaps**: Low-liquidity gaps

### 10.2 Detection Methods

| Method | Best For | Threshold |
|--------|----------|-----------|
| **IQR Method** | General use | 1.5× IQR from Q1/Q3 |
| **Z-Score** | Normal distributions | |z| > 3 |
| **Modified Z-Score** | Skewed data (forex) | |MZ| > 3.5 |
| **Winsorization** | Keep data, reduce impact | Clip at 5th/95th percentile |

### 10.3 Recommended Pipeline for Forex
```python
# 1. Detect using Modified Z-Score (most robust for forex)
from scipy.stats import median_abs_deviation
mad = median_abs_deviation(returns)
modified_z = 0.6745 * (returns - np.median(returns)) / mad
outliers = np.abs(modified_z) > 3.5

# 2. Winsorize (don't remove — extreme moves are real)
from scipy.stats import mstats
winsorized = mstats.winsorize(returns, limits=[0.05, 0.05])

# 3. Separate handling by cause
# Data errors → remove
# News spikes → keep but flag
# Flash crashes → keep (they're real market events)
```

### 10.4 Critical Rule
**Never automatically remove outliers** in forex — they represent real market events (NFP, FOMC). Instead:
1. Flag them with a binary feature `is_outlier`
2. Winsorize extreme values at 5th/95th percentile
3. Let the model learn from extreme events — they're important for risk management

---

## 11. Concept Drift

### 11.1 What Causes Drift in EUR/USD
- Central bank policy changes (Fed, ECB)
- Regime shifts (trending → ranging → trending)
- Volatility regime changes (calm → chaotic)
- Liquidity changes (session transitions)
- Geopolitical events (elections, crises)

### 11.2 Detection Methods

| Method | Type | Best For |
|--------|------|----------|
| **ADWIN** | Adaptive windowing | Sudden drift |
| **KSWIN** | Kolmogorov-Smirnov test | Distribution shift |
| **Page-Hinkley** | Cumulative sum | Gradual drift |
| **DDM** | Drift Detection Method | Error rate changes |
| **Rolling IC** | Information coefficient decay | Model degradation |

### 11.3 Recommended Monitoring Pipeline
```python
# 1. Monitor prediction accuracy rolling window
accuracy_window = rolling_accuracy(predictions, window=100)

# 2. Monitor feature drift (PSI - Population Stability Index)
for feature in features:
    psi = calculate_psi(reference_dist, current_dist)
    if psi > 0.25:
        alert(f"Feature drift: {feature}")

# 3. Monitor model performance decay
ic_rolling = rolling_ic(predictions, actuals, window=500)
if ic_rolling < 0.02:
    trigger_retraining()

# 4. ADWIN for sudden drift detection
from river import drift
adwin = drift.ADWIN()
for error in prediction_errors:
    adwin.update(error)
    if adwin.drift_detected:
        trigger_retraining()
```

### 11.4 Adaptation Strategy
1. **Walk-forward retraining**: Every 1 month, retrain on last 6 months
2. **Concept drift detection**: Monitor rolling IC, trigger if < threshold
3. **Ensemble approach**: Keep multiple models, weight by recent performance
4. **Online learning**: Incremental updates with new data (if using linear models)

---

## 12. Profit-Aware Training

### 12.1 The Problem
Traditional ML optimizes for accuracy/F1, but:
- High accuracy ≠ profitable trading
- A model predicting "always hold" has 100% accuracy but 0% profit
- Classification accuracy ignores trade magnitude

### 12.2 Profit-Aware Approaches

**A. Custom Loss Function**
```python
def profit_loss(y_true, y_pred, tp=4.5, sl=1.5):
    """
    Custom loss that penalizes wrong direction more when
    the trade would have hit SL before TP
    """
    direction_correct = (y_true == y_pred)
    # Weight losses by potential profit/loss ratio
    loss = tf.where(direction_correct, 
                    -y_pred * y_true * tp,  # profit term
                    y_pred * y_true * sl)    # loss penalty
    return tf.reduce_mean(loss)
```

**B. Mean Absolute Directional Loss (MADL)**
From recent research (2025):
> "Models optimized with MADL consistently outperform accuracy-optimized models in risk-adjusted terms."

```python
def madl(y_true, y_pred):
    """Optimizes for directional accuracy weighted by returns"""
    return -np.mean(y_true * y_pred)
```

**C. Sharpe Ratio as Objective**
```python
def neg_sharpe(returns):
    """Negative Sharpe ratio for minimization"""
    return -np.mean(returns) / (np.std(returns) + 1e-8)
```

**D. Expectile Regression**
- Optimize for specific quantiles of the return distribution
- More robust than mean-based optimization

### 12.3 Recommended Approach
1. **Primary training**: Use standard cross-entropy loss
2. **Threshold optimization**: Post-training, optimize decision threshold for profit
3. **Ensemble weighting**: Weight model outputs by their contribution to Sharpe ratio
4. **Walk-forward profit evaluation**: Final model selection based on OOS profit, not accuracy

### 12.4 Key Research Finding
From Springer (2025):
> "Simpler, interpretable models—particularly logistic regression with MADL optimization—consistently outperform complex architectures in risk-adjusted terms."

**Implication**: Don't over-complicate. A well-tuned logistic regression with profit-aware loss can beat a complex LSTM.

---

## Summary: Actionable Configuration

### Feature Set (Recommended ~40 features)
```
Tier 1 (15): ATR(14), RSI(7), RSI(14), EMA(8)/EMA(21) cross,
  candle_body_ratio, candle_shadow_ratio, price_position,
  ROC_5, ROC_10, ROC_20, momentum_10, BB_width, BB_position,
  session_overlap, volume_momentum

Tier 2 (15): MACD_hist, stochastic_K, CCI_20, williams_R,
  rolling_std_10, rolling_std_20, atr_ratio_5_20,
  ema_50_position, linear_reg_slope_20,
  M15_RSI, M15_ema_trend, H1_ema_trend, H4_trend_binary,
  hour_sin, hour_cos

Tier 3 (10): VWAP_deviation, OBV_momentum, bb_squeeze,
  price_vs_session_high, price_vs_session_low,
  atr_percentile, returns_skewness_20, returns_kurtosis_20,
  range_ratio, body_ratio_3bar
```

### Training Configuration
```
Model: XGBoost + LightGBM ensemble with meta-learner
Loss: Cross-entropy (primary), MADL (threshold optimization)
Walk-forward: 30 windows, 6mo train / 1mo test
Purge: 20 bars, Embargo: 20 bars
Class handling: class_weight='balanced' + SMOTE on minority
Normalization: RobustScaler for indicators, StandardScaler for returns
Feature selection: VarianceThreshold → Correlation → Mutual Information → SHAP
Confidence threshold: >= 70/100
```

### Data Preparation Pipeline
```
1. Fetch M5 OHLCV + spread from MT5
2. Fetch M15, H1, H4 OHLCV for MTF features
3. Compute 40+ features (price, momentum, volatility, session, MTF)
4. Apply triple-barrier labeling (4.5x ATR TP, 1.5x ATR SL, 20-bar horizon)
5. Winsorize extreme returns at 5th/95th percentile
6. Apply RobustScaler/StandardScaler (fit on train only)
7. VarianceThreshold → Correlation filter → MI selection
8. Walk-forward split with purge/embargo
9. Train ensemble, optimize threshold for profit
10. Monitor concept drift with ADWIN + rolling IC
```

---

*Research compiled: 2026-06-27*
*Sources: Qlib documentation, MQL5 articles, ScienceDirect papers, QuantInsti research, arXiv papers, sklearn docs, TradingView strategies, walk-forward validation guides*

# PHASE_0: Data Pipeline & Training Methodology Research for EUR/USD M5

> Research compiled from 40+ sources including academic papers, GitHub repos (binary-options-ml, applied-ml-trading, FOREX-TRADING-BOT), ML for Trading textbooks, QuantInsti, Marcos López de Prado's framework, and production forex ML systems.

---

## 1. How Much M5 Data Is Needed?

### Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| **M5 bars per year** | ~105,120 (252 trading days × 288 bars/day) | Calculation |
| **16 months M5 bars** | ~140,160 | Calculation |
| **Minimum for GBM training** | ~50,000 bars (~6 months M5) | binary-options-ml repo (816k rows from 10yr multi-pair) |
| **Minimum for deep learning** | ~200,000+ bars | transformer-eurusd repo (10yr optimal) |
| **Optimal for LightGBM** | 2-5 years M5 (210k-525k bars) | Binary-options-ml: "Less than 5 years will degrade model quality" |
| **Sweet spot for transformer** | 10 years M5 (~1M bars) | Applied-ml-trading-transformer: 10yr optimal, 20yr degraded |

### Your 16-Month Constraint

With 16 months (~140k M5 bars), you have:
- **Sufficient** for tree-based models (XGBoost/LightGBM/CatBoost) — these need ~10-50k samples minimum
- **Marginal** for deep learning (LSTM/Transformer) — better with 3+ years
- **Excellent** if you supplement with multi-pair training (EUR/USD + GBP/USD + USD/JPY = ~420k bars)

### Recommendation for Your Pipeline
```
Primary: 16 months EUR/USD M5 (~140k bars)
Augmented: + GBP/USD + USD/JPY for training (~420k total)
Target: LightGBM/XGBoost ensemble (NOT deep learning)
```

---

## 2. Walk-Forward vs Expanding Window vs Rolling Window

### Comparison Matrix

| Method | Best For | Forex Suitability | Pros | Cons |
|--------|----------|-------------------|------|------|
| **Rolling Window** | Non-stationary, regime changes | **BEST for M5 forex** | Adapts to regime shifts, forgets stale data | Uses less data per window |
| **Expanding Window** | Stationary processes | Good for longer timeframes | More data as you go, stable estimates | Slow to adapt, old regimes pollute |
| **Anchored (Expanding)** | Long-term relationships | Decent for daily/weekly | Growing training set | Stale data can mislead |

### Expert Consensus for Forex M5

From Susan Potter (quant practitioner):
> "I use rolling windows when markets are non-stationary or the strategy adapts to recent conditions. FX markets in particular go through regime shifts (changes in central bank policy, volatility regimes, correlation structures) where data from three years ago may actively mislead."

From MetricGate:
> "Choose a rolling window when the process drifts, has regime changes, or contains structural breaks; old data can then actively harm forecasts."

### Your Configuration
```yaml
# RECOMMENDED: Rolling window for EUR/USD M5
window_type: rolling
train_window: 6 months  # ~84,000 M5 bars
test_window: 1 month    # ~14,000 M5 bars
step_size: 1 month      # slide forward monthly
n_splits: 10            # 10 walk-forward windows from 16 months
```

---

## 3. Optimal Train/Test Split for M5 Forex

### Industry Benchmarks

| Source | Train | Test | Method |
|--------|-------|------|--------|
| binary-options-ml (EUR/USD M5) | 2013-2022 (845k rows) | 2025+ (136k rows) | Time-based, chronological |
| Applied-ML-Transformer | 60% | 20% val + 20% test | 60/20/20 chronological |
| Walk-forward standard | 6-12 months train | 1-3 months test | Rolling |
| QuantInsti (daily) | 70% train | 30% test | Single split |

### For Your 16-Month Dataset

```
Option A: Single Split (quick validation)
  Train: months 1-12 (~126,000 bars)
  Val:   months 13-14 (~28,000 bars)
  Test:  months 15-16 (~28,000 bars)

Option B: Walk-Forward (recommended)
  10 windows, each:
    Train: 6 months rolling (~84,000 bars)
    Test:  1 month forward (~14,000 bars)
    Embargo: 48 bars (4 hours M5)
```

### Specific Ratios
- **Minimum train set**: 3 months M5 (~42,000 bars) for stable GBM
- **Optimal train set**: 6-12 months for rolling windows
- **Test set**: 1-3 months to get 20-30+ trade signals for statistical significance
- **With 16 months**: Use 6-month rolling windows = 10 folds maximum

---

## 4. Purge and Embargo Periods

### What the Research Says

From López de Prado's framework (Wikipedia, QuantInsti, NovaQuantLab):
- **Purging**: Remove training observations whose label windows overlap with the test period
- **Embargo**: Add buffer gap AFTER test fold to account for serial correlation in features

### Specific Values for M5 Forex

| Timeframe | Purge Period | Embargo Period | Source |
|-----------|-------------|----------------|--------|
| **M5 (5-min)** | 20 bars (label horizon) | 12-48 bars (1-4 hours) | Skill feed, crypto guidelines |
| M1 (1-min) | 60-240 bars | 1-4 hours | Purged-CV library |
| Hourly | 6-24 bars | 6-24 hours | Skill feed |
| Daily | 2-5 bars | 2-5 days | De Prado framework |

### Your Configuration (M5 with 20-bar label horizon)
```yaml
purge_period: 20 bars    # Match your label horizon (4.5x ATR = ~20 bars)
embargo_period: 48 bars  # 4 hours M5 (covers M15/H1 feature lag)
# Embargo >= 2x label computation horizon for M5
```

### Implementation Detail
```python
# From purged-cross-validation library
def purge_and_embargo(train_idx, test_idx, purge_len=20, embargo_len=48):
    """
    Purge: remove training rows where label window overlaps test
    Embargo: remove training rows within embargo_len after test end
    """
    test_start = test_idx.min()
    test_end = test_idx.max()
    
    # Purge: drop training rows with label windows hitting test
    purge_mask = (train_idx < test_start - purge_len) | (train_idx > test_end)
    
    # Embargo: drop training rows within embargo after test
    embargo_mask = train_idx > test_end + embargo_len
    
    clean_train = train_idx[purge_mask & embargo_mask]
    return clean_train
```

---

## 5. Feature Normalization Order

### Universal Rule (100% Consensus)

**Split FIRST, normalize SECOND. Always.**

From every source consulted:
- StackOverflow (6+ answers, all agree)
- Towards Data Science
- Sophelio (2026 pipeline guide)
- Lightly.ai
- Stats StackExchange

### Correct Order
```
1. Load raw data
2. Split into train/test (chronologically)
3. Fit scaler on TRAINING data only
4. Transform training data with scaler
5. Transform test data using SAME scaler (no refit!)
6. Train model
```

### Why
- Normalization parameters (mean, std, min, max) computed on full dataset leak test set information
- Even a single outlier in test set shifts the scaling
- Scalers must be fit on training data, then applied to test data as-is

### Implementation
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Split FIRST
train_data, test_data = chronological_split(df, cutoff_date)

# Fit on train ONLY
scaler = StandardScaler()  # or MinMaxScaler()
scaler.fit(train_data[feature_cols])

# Transform both
train_scaled = scaler.transform(train_data[feature_cols])
test_scaled = scaler.transform(test_data[feature_cols])  # NO refit!

# Save scaler for production
joblib.dump(scaler, 'scaler.pkl')
```

---

## 6. Data Leakage Prevention

### Common Mistakes in Forex ML Pipelines

| Mistake | Impact | Prevention |
|---------|--------|------------|
| **Normalizing before split** | Test set statistics leak into train | Split first, fit scaler on train only |
| **Lookahead in labels** | Model sees future in training | Use point-in-time data, no future features |
| **Feature engineering on full dataset** | Rolling features use future data | Compute features within each fold |
| **Shuffling time series** | Train on future, test on past | Never shuffle, use chronological splits |
| **No purge/embargo** | Label windows overlap | Apply purge + embargo at fold boundaries |
| **Using test set for early stopping** | Peeking at test performance | Use validation set, hold test for final |
| **Retuning after seeing test** | Overfitting to test set | One-time test evaluation only |
| **Survivorship bias** | Only using available pairs | Include all historical instruments |
| **Ignoring spread/slippage** | Unrealistic P&L | Model transaction costs in backtest |

### Critical Check: Are Your Features Causal?
```python
# WRONG: Rolling features that include future data
df['volatility'] = df['close'].rolling(20).std()  # Uses future close!

# CORRECT: Shift to ensure only past data
df['volatility'] = df['close'].shift(1).rolling(20).std()
```

---

## 7. Concept Drift Detection

### Method Comparison

| Method | What It Measures | Best For | Threshold |
|--------|-----------------|----------|-----------|
| **PSI** | Distribution shift (binned) | Feature & prediction monitoring | <0.1 stable, 0.1-0.25 investigate, >0.25 act |
| **KS Test** | Distributional change (p-value) | Continuous features | p<0.05 significant (but hypersensitive at scale) |
| **Rolling Accuracy** | Model performance decay | Label-level monitoring | Compare to baseline ±1 std |
| **CUSUM** | Sequential error accumulation | Persistent shifts | Custom slack calibration |
| **ADWIN** | Adaptive window change point | Streaming detection | Automatic |

### Expert Recommendation (from multiple sources)

> "Start with PSI for feature distributions and a rolling information ratio or Spearman correlation for label-level decay. These two cover the most common failure modes and are lightweight to implement." — StockAlpha.ai

> "Using multiple metrics in combination produces a more complete picture than any single measure alone. A model that triggers elevated readings across KS, JSD, and PSI simultaneously is sending a much stronger signal." — darwintIQ

### Your Monitoring Stack
```yaml
primary_monitor: PSI
  - Monitor top 10 features by importance
  - Monitor prediction score distribution
  - Threshold: 0.10 warning, 0.25 action

secondary_monitor: Rolling Accuracy
  - Window: 200 predictions
  - Threshold: drops below baseline - 1 std

confirmation: KS Test
  - Use when PSI fires to confirm
  - Pair with effect-size threshold (not just p-value)

trigger: Retrain when:
  1. PSI > 0.25 for 2 consecutive windows, OR
  2. Rolling accuracy drops below threshold for 500+ predictions, OR
  3. Known market event (central bank decision, flash crash)
```

---

## 8. Retraining Frequency

### Industry Benchmarks

| Cadence | Use Case | Source |
|---------|----------|--------|
| **Weekly** | High-frequency scalping M1-M5 | FxRobotEasy, GoldStrike AI |
| **Monthly** | Standard ML-filter products | FxRobotEasy (industry standard) |
| **Quarterly** | Trend-following, mean-reversion | Lower frequency strategies |
| **Every 3 days** | Aggressive adaptation | Pigent_forex_pro (XGBoost) |
| **Annual** | Insufficient for serious ML | Warning sign |

### Your Configuration (M5 scalping, 16 months data)
```yaml
# RECOMMENDED: Weekly retraining
retrain_frequency: weekly
training_window: 6 months rolling  # ~84k bars
retrain_day: saturday  # Low-volatility period
validation: walk-forward on held-out week

# Alternative: Monthly (lower overhead)
retrain_frequency: monthly
training_window: 12 months rolling
```

### When to Emergency Retrain
1. PSI > 0.25 on prediction score for 2+ windows
2. Daily drawdown exceeds 2%
3. Major central bank announcement (ECB, Fed)
4. Volatility regime shift detected (ATR > 2x average)

---

## 9. Ensemble Diversity

### Why Diversity Matters

From research papers:
> "Stacking ensembles with diverse base models improve robustness — the V3 stacking ensemble (XGBoost+LightGBM+CatBoost → Logistic meta-learner) achieved Sharpe 1.165 vs 0.343 for single models." — suraj-phanindra/quantitative-ml-dl-ensemble-algotrading

### How to Ensure Diversity

| Strategy | Implementation |
|----------|---------------|
| **Algorithm diversity** | XGBoost + LightGBM + CatBoost (different tree growth, loss functions) |
| **Feature diversity** | Different feature subsets per model (technical, microstructure, regime) |
| **Hyperparameter diversity** | Different max_depth, learning_rate per model |
| **Training data diversity** | Different time windows or bootstrap samples |
| **Objective diversity** | One model maximizes accuracy, another maximizes Sharpe |

### Proven Ensemble Architectures

**Architecture 1: Weighted Average (Simple)**
```python
ensemble_pred = 0.4 * xgb_pred + 0.35 * lgbm_pred + 0.25 * cat_pred
```

**Architecture 2: Stacking (Best Performance)**
```python
# Base models
xgb_model = XGBClassifier(...)
lgbm_model = LGBMClassifier(...)
cat_model = CatBoostClassifier(...)

# Meta-learner
meta_model = LogisticRegression()
# Train on out-of-fold predictions from base models
```

**Architecture 3: Regime-Adaptive (Advanced)**
```python
# Weight models based on current regime
if volatility_regime == 'low':
    weights = [0.2, 0.5, 0.3]  # favor LightGBM
elif volatility_regime == 'high':
    weights = [0.5, 0.2, 0.3]  # favor XGBoost
```

### Correlation Check
```python
# Ensure base model predictions are not too correlated
corr_matrix = pd.DataFrame({
    'xgb': xgb_preds,
    'lgbm': lgbm_preds,
    'cat': cat_preds
}).corr()

# If correlation > 0.85, models are too similar
# Consider removing one or adding more diversity
```

---

## 10. Cross-Validation for Time Series

### Method Comparison

| Method | Leakage Risk | Computational Cost | Forex Suitability |
|--------|-------------|-------------------|-------------------|
| **Random K-Fold** | **HIGH** | Low | NEVER use for time series |
| **Walk-Forward (Rolling)** | None | Medium | **BEST for M5 forex** |
| **Walk-Forward (Expanding)** | None | Medium | Good for longer timeframes |
| **Purged K-Fold** | None | Medium-High | Good for label-based |
| **CPCV (Combinatorial)** | None | Very High | Research/academic |

### Recommended for Your Pipeline

**Primary: Walk-Forward with Purge + Embargo**
```python
from purgedcv import WalkForwardSplit

wf = WalkForwardSplit(
    n_splits=10,
    train_size=84000,   # 6 months M5
    test_size=14000,    # 1 month M5
    purge_size=20,      # label horizon
    embargo_size=48,    # 4 hours M5
    expanding=False     # rolling, not expanding
)
```

**Secondary: Purged K-Fold for final validation**
```python
from purgedcv import PurgedKFold

pkf = PurgedKFold(
    n_splits=5,
    purge_size=20,
    embargo_pct=0.01  # 1% of data
)
```

---

## 11. Hyperparameter Tuning

### Optuna vs GridSearch

| Aspect | GridSearchCV | Optuna |
|--------|-------------|--------|
| **Search strategy** | Exhaustive brute force | Bayesian (TPE) learning |
| **Efficiency** | Low (tests all combos) | High (learns from trials) |
| **Custom objectives** | Limited | Any metric (Sharpe, profit factor) |
| **Pruning** | No | Yes (early stop bad trials) |
| **Speed** | Slow for large spaces | 30-50% faster |
| **Best for** | Small parameter spaces | Complex, non-linear spaces |

### Expert Consensus
> "Optuna is the modern choice for serious tuning — it is smarter (Bayesian), faster (pruning bad trials), and gives you built-in visualization of what matters most. If you are still using only GridSearch, try Optuna on your next project." — DriveDataScience

### Optuna Configuration for Your Models
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    # Use walk-forward CV, NOT random CV
    scores = walk_forward_cv(X_train, y_train, params, n_splits=5)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=3600)

# Best params from binary-options-ml reference:
# trees: 276, max_depth: 6, num_leaves: 50, 
# min_child_samples: 300, learning_rate: 0.01
```

---

## 12. Class Imbalance Solutions

### Forex M5 Reality
- Binary direction (UP/DOWN) is roughly 50/50 — **minimal imbalance**
- With 3-class (LONG/SHORT/HOLD) — HOLD often dominates (35-40%)
- With ATR-based TP/SL labels — hit rates vary (30-40% for TP hits)

### Solution Comparison

| Method | Effectiveness | Best For | Implementation |
|--------|--------------|----------|----------------|
| **Threshold Moving** | **Most consistent** | Any method | Post-training calibration |
| **Class Weights** | Good baseline | Tree models | `class_weight='balanced'` |
| **SMOTE** | Moderate | Tabular data, 1:10-1:100 imbalance | `imblearn.over_sampling.SMOTE` |
| **Undersampling** | Risky with small data | Large datasets | `RandomUnderSampler` |
| **Focal Loss** | Advanced | Deep learning | Custom loss function |

### Research Finding
> "Decision Threshold Calibration emerged as the most consistently effective technique, offering significant performance gains across various datasets and models." — arXiv 2409.19751 (9,000 experiments)

### Your Implementation
```python
# Step 1: Train with class weights (if imbalanced)
model = LGBMClassifier(class_weight='balanced')

# Step 2: After training, tune threshold on validation set
from sklearn.metrics import precision_recall_curve

y_proba = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

# Find threshold that maximizes F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Step 3: Apply at inference
predictions = (model.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)

# For 3-class (LONG/SHORT/HOLD):
# Use confidence filtering: only trade when |prob - 0.5| >= 0.06
# From binary-options-ml: lifts accuracy from 52% → 58.47%
```

---

## Summary: Complete Configuration for EUR/USD M5 (16 Months)

```yaml
# ===== DATA =====
symbol: EUR/USD
timeframe: M5
data_duration: 16 months (~140k bars)
augmented_pairs: [GBP/USD, USD/JPY]  # Optional: ~420k total

# ===== SPLIT =====
method: walk-forward-rolling
train_window: 6 months (~84k bars)
test_window: 1 month (~14k bars)
n_splits: 10
step_size: 1 month

# ===== PURGE/EMBAGO =====
purge_bars: 20       # = label horizon
embargo_bars: 48     # = 4 hours (2x M15 feature lag)

# ===== NORMALIZATION =====
order: split-first, fit-on-train-only
scaler: StandardScaler  # or MinMaxScaler for neural nets
scaler_save: scaler.pkl

# ===== FEATURES =====
engineering_order:
  1. Causal rolling features (shift before rolling)
  2. Technical indicators (RSI, MACD, ATR, BB)
  3. Multi-timeframe (M15, H1, H4 confirmations)
  4. Temporal (session, hour, day-of-week)
  5. Volatility regime features
n_features_target: 50-80

# ===== MODEL =====
ensemble: [XGBoost, LightGBM, CatBoost]
meta_learner: LogisticRegression  # or weighted average
diversity_check: correlation < 0.85

# ===== HYPERPARAMETER TUNING =====
method: Optuna
n_trials: 100
objective: walk-forward F1 or Sharpe
cv: walk-forward with purge+embargo

# ===== CLASS IMBALANCE =====
strategy: threshold-moving (primary)
backup: class_weight='balanced'
confidence_filter: |prob - 0.5| >= 0.06

# ===== DRIFT MONITORING =====
primary: PSI (threshold: 0.10 warning, 0.25 act)
secondary: Rolling accuracy (window: 200, threshold: baseline - 1std)
confirmation: KS test

# ===== RETRAINING =====
frequency: weekly
emergency_triggers:
  - PSI > 0.25 for 2 windows
  - Rolling accuracy below threshold
  - Major central bank event
```

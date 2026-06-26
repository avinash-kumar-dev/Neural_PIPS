# AGENTS.md — EUR/USD 3:1 TP:SL Signal Generator

## What We Are Building

A **EUR/USD scalping signal GENERATOR** (NOT an execution bot). It outputs:
- **LONG** / **SHORT** / **NO-TRADE** alerts
- Via **Telegram** and **Email**
- With **ATR-based dynamic TP/SL** (minimum 3:1 TP:SL ratio)
- During **London-NY overlap only** (13:00-16:00 UTC)
- Target: **1-3 high-conviction signals per day**
- Target win rate: **>35%** (break-even is 25% at 3:1)

**NO MT5 order execution. NO position management. NO automation of trades.**

---

## Critical Rules for All Agents

### Rule 1: No Fallbacks Without Confirmation
- If something doesn't work (library, API, method), **do NOT silently use a fallback**
- **Search** for the correct solution first
- **Confirm with the user** before switching approaches
- Document the issue and the proposed fix

### Rule 2: Step by Step
- Implement one module at a time
- Verify each module works before moving to the next
- Run lint/typecheck after each implementation step

### Rule 3: Preserve Context
- All research is in `./PHASE_0_*.md` files
- Architecture is in `./PHASE_1_EURUSD_3TO1_ARCHITECTURE.md`
- Do NOT overwrite or delete research files

### Rule 4: No Comments Unless Asked
- Do not add comments to code unless the user explicitly requests it

### Rule 5: Security
- Never commit `.env` files or credentials
- Never log secrets or API keys

---

## Project Structure

```
tradding_app/
+-- AGENTS.md                    <-- THIS FILE (persistent instructions)
+-- PHASE_0_EURUSD_3TO1_RESEARCH.md
+-- PHASE_0_HIGH_STAR_GITHUB_REPOS.md
+-- PHASE_0_GITHUB_BOTS_RESEARCH.md
+-- PHASE_0_INDICATORS_STRATEGIES.md
+-- PHASE_0_LIBRARIES_FRAMEWORKS.md
+-- PHASE_0_SIGNAL_FREQUENCY_ANALYSIS.md
+-- PHASE_1_EURUSD_3TO1_ARCHITECTURE.md
+-- PHASE_1_ARCHITECTURE_DESIGN.md        (old XAU/USD, kept for reference)
+-- src/                                  <-- Implementation goes here
+-- models/
+-- data/
+-- config/
+-- tests/
+-- scripts/
+-- requirements.txt
+-- .env
```

---

## Core Parameters (DO NOT CHANGE WITHOUT ASKING)

| Parameter | Value |
|-----------|-------|
| Symbol | EUR/USD only |
| TP | 4.5x ATR(14) on M5 — adapts to volatility |
| SL | 1.5x ATR(14) on M5 — adapts to volatility |
| TP:SL Ratio | Minimum 3:1 (always enforced) |
| Entry Timeframe | M5 |
| Confirmation Timeframes | M15, H1, H4 |
| Session | 13:00-16:00 UTC ONLY |
| Spread Max | 0.5 pips |
| Confidence Threshold | >= 70/100 |
| H4 Filter | HARD (never trade against) |
| Walk-Forward Windows | 30 |
| Train Window | 6 months |
| Test Window | 1 month |
| Purge Bars | 20 |
| Embargo Bars | 20 |

---

## Implementation Order (Phase 2)

Execute these steps IN ORDER. Do not skip ahead.

### Step 1: Project Scaffolding
- Create `src/` directory with `__init__.py`
- Create `config/config.yaml`
- Create `requirements.txt`
- Create `.env.example` (not .env with real secrets)

### Step 2: MT5 Connection + Data Pipeline
- `src/mt5_connection.py` — REST API connection to headless-mt5 Docker container
- `src/data_pipeline.py` — fetch OHLCV, fetch spread, store as Parquet
- `docker-compose.yml` — headless-mt5 container (MT5 via Wine + FastAPI REST)
- Test: verify MT5 Docker connection works, fetch 100 M5 candles

### Step 3: Feature Engineering
- `src/feature_engineering.py` — all 4 tiers of features (~50 factors)
- `src/feature_pipeline.py` — datasieve-style Pipeline (VarianceThreshold + MinMaxScaler)
- Test: compute features on fetched data, verify no NaN/inf in output

### Step 4: Triple-Barrier Labeling
- `src/labels.py` — ATR-based dynamic TP (4.5x ATR) and SL (1.5x ATR), horizon=20 M5 bars
- Test: generate labels on historical data, verify distribution

### Step 5: Rule-Based Filters
- `src/rule_filters.py` — 8 sequential gates
- Test: run filters on sample data, verify rejection reasons

### Step 6: ML Ensemble Training
- `src/model_trainer.py` — XGBoost + LightGBM + CatBoost + Meta-learner
- `src/model_trainer.py` — Walk-forward with purge/embargo
- Test: train on 1 window, verify models save/load

### Step 7: Confidence Scoring
- `src/confidence_scorer.py` — 8-factor weighted scoring
- Test: score sample predictions, verify threshold gating

### Step 8: Signal Engine (Main Loop)
- `src/signal_engine.py` — M5 candle close event loop
- Test: dry run, verify signals emitted correctly

### Step 9: Notifications
- `src/notifications.py` — Telegram + Email
- Test: send test signal, verify delivery

### Step 10: Backtest
- `src/backtest.py` — VectorBT + Backtrader integration
- Test: run walk-forward backtest, verify metrics

### Step 11: Monitoring
- `src/metrics.py` — accuracy tracking, spread monitoring, drift detection
- Test: verify retraining trigger works

---

## Tech Stack (DO NOT CHANGE WITHOUT ASKING)

```txt
requests             # MT5 REST API (headless-mt5), FRED API, ForexFactory calendar
pandas               # Data manipulation
numpy                # Numerical computing
scipy                # Statistical tests
yfinance             # DXY fallback data
python-dotenv        # .env loading
pandas-ta            # Technical indicators (if needed)
scikit-learn         # Pipeline, preprocessing, metrics
xgboost              # Model 1
lightgbm             # Model 2
catboost             # Model 3
optuna               # Hyperparameter optimization (future)
vectorbt             # Fast backtesting
backtrader           # Realistic backtesting
joblib               # Model persistence
pyarrow              # Parquet support
matplotlib           # Visualization
plotly               # Interactive charts
docker               # headless-mt5 container (MT5 on Linux via Wine)
```

---

## Research Files Reference

| File | Content |
|------|---------|
| `PHASE_0_EURUSD_3TO1_RESEARCH.md` | EUR/USD behavior, 3:1 framework, features, MTF hierarchy, model architecture, confidence scoring, validation, backtesting |
| `PHASE_0_HIGH_STAR_GITHUB_REPOS.md` | 25+ repos (Freqtrade 51.8k, Qlib 45.1k, Backtrader 22.1k, VectorBT 7.7k, FinRL 15.5k) with deep code analysis |
| `PHASE_0_GITHUB_BOTS_RESEARCH.md` | 27+ EUR/USD specific repos |
| `PHASE_0_INDICATORS_STRATEGIES.md` | 15+ indicator types, 10+ strategy types, SMC/ICT |
| `PHASE_0_LIBRARIES_FRAMEWORKS.md` | Python library guide with code examples |
| `PHASE_0_SIGNAL_FREQUENCY_ANALYSIS.md` | Signal frequency vs profitability, Kelly Criterion |
| `PHASE_1_EURUSD_3TO1_ARCHITECTURE.md` | Full system architecture (14 sections, pseudocode, config) |

---

## Key Patterns Adopted From Code Analysis

### From Freqtrade (51.8k stars)
- FreqAI Template Method pattern for ML pipeline
- datasieve Pipeline (VarianceThreshold -> MinMaxScaler -> PCA)
- Walk-forward retraining
- Signal shift by 1 candle (anti-lookahead)
- Telegram bot integration

### From Microsoft Qlib (45.1k stars)
- Alpha158 factor engineering
- ZScoreNorm / CSZScoreNorm processors
- LightGBM as primary model
- Factor mining approach

### From VectorBT (7.7k stars)
- Numba JIT for speed
- Broadcasting for parameter sweeps
- WalkForwardAnalyzer

### From Backtrader (22.1k stars)
- Event-driven architecture
- Slippage modeling
- Dual-mode (_runonce + _runnext)

### From FinRL (15.5k stars)
- Turbulence Index (Mahalanobis distance)
- Risk-off when turbulence > 99th percentile

---

*Last updated: 2026-06-25*

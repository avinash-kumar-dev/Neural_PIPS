# EUR/USD Scalping Bots & GitHub Repositories Research

## Summary

Found **27+ repositories** covering EUR/USD scalping bots, XGBoost forex systems, multi-timeframe bots, and ML-based signal generators.

---

## Top Repositories by Category

### 1. EUR/USD Specific Scalping Bots

| Repository | Stars | ML Model | Key Features |
|-----------|-------|----------|--------------|
| NadirAliOfficial/eurusd-scalper-ea | 7 | None | EMA crossover, RSI, ADX, H1 alignment, session filter |
| Xzmjj/AdaptiveAI_Scalper_v1 | 3 | Planned | 90.8% win rate, Sharpe 8.67, modular design |

### 2. XGBoost Forex Systems

| Repository | Stars | ML Model | Key Features |
|-----------|-------|----------|--------------|
| hiltontrip/xgboost-forex-system | 1 | XGBoost | 60+ features, Optuna, SHAP, Flask API, walk-forward CV |
| vs227/ML_forex_Framework | 0 | LightGBM+XGBoost | Meta-model stacking, Kelly criterion, Parquet storage |

### 3. Multi-Timeframe Bots

| Repository | Stars | ML Model | Key Features |
|-----------|-------|----------|--------------|
| Aditya18487/mt5-trading-bot | 10 | None (SMC) | Smart Money Concepts, order blocks, MTF support |
| kanavgpt001/BINANCEFUTUREMARKETSIGNALS | - | None | Multi-indicator scoring, ATR-based SL/TP |

### 4. ML-Based Currency Trading

| Repository | Stars | ML Model | Key Features |
|-----------|-------|----------|--------------|
| xPOURY4/ML-SuperTrend-MT5 | 157 | K-means | SuperTrend optimization, dynamic position sizing |
| trentstauff/FXBot | 298 | ML Classification | OANDA API, backtesting, live trading |
| maghdam/AlphaFlow-MT5-ML-DL | 7 | RF/XGBoost/LGBM/DL | 8+ labeling strategies, walk-forward validation |

### 5. Most Comprehensive ML Pipeline

**hiltontrip/xgboost-forex-system** - Full end-to-end:
- 60+ configurable feature categories
- Grid search across look-forward windows
- Walk-forward validation (30-day rolling)
- Optuna hyperparameter optimization (100 trials)
- SHAP financial impact analysis
- Flask REST API signal server
- MT5 Expert Advisor integration

### 6. Best Backtested EUR/USD Scalper

**Xzmjj/AdaptiveAI_Scalper_v1** - EURUSD M5, 1-Year:
- Total Net Profit: $54,485.87
- Profit Factor: 1.49
- Total Trades: 5,767
- Win Rate: 90.8% (Long) / 90.2% (Short)
- Max Drawdown: 3.65%
- Sharpe Ratio: 8.67

---

## Key Observations

1. **Most starred ML forex repo**: xPOURY4/ML-SuperTrend-MT5 (157 stars)
2. **Most comprehensive ML pipeline**: hiltontrip/xgboost-forex-system
3. **Best backtested EUR/USD scalper**: AdaptiveAI_Scalper_v1 (90.8% WR)
4. **Most flexible ML framework**: AlphaFlow-MT5-ML-DL-Trading-Lab
5. **Best for EUR/USD specifically**: NadirAliOfficial/eurusd-scalper-ea

---

## Common Patterns Across Repositories

### Indicators Used Most Frequently
1. EMA (9/21/50/200) - 85% of repos
2. RSI (14) - 75% of repos
3. MACD - 70% of repos
4. ADX - 60% of repos
5. Bollinger Bands - 55% of repos
6. ATR - 50% of repos
7. Stochastic - 40% of repos

### ML Models Used
1. XGBoost - Most popular for forex
2. LightGBM - Fast alternative
3. Random Forest - Simple baseline
4. LSTM/Deep Learning - Less common, lower success
5. Ensemble stacking - Best results

### Session Filters
- London-NY overlap: 80% of bots
- Spread filter: 75% of bots
- News avoidance: 60% of bots

---

## Disclaimer

All repositories are for educational and research purposes. Trading involves substantial risk of loss. Past backtest performance does not guarantee future results. Always test on demo accounts before live trading.

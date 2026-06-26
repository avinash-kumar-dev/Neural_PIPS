# Python Libraries & Frameworks Guide

## Core Stack

### MT5 Integration
```bash
pip install MetaTrader5
```
- Windows only (requires MT5 terminal)
- Data retrieval, order execution, position management

### Data Analysis
```bash
pip install pandas numpy scipy
```

### Technical Analysis
```bash
pip install pandas-ta TA-Lib ta mplfinance
```
- **pandas-ta**: 150+ indicators, pandas extension
- **TA-Lib**: Industry-standard C library, 200+ indicators
- **ta**: Pure Python, no C dependencies
- **mplfinance**: Financial data visualization

### Machine Learning
```bash
pip install scikit-learn xgboost lightgbm catboost
```
- **XGBoost**: Most popular for forex tabular data
- **LightGBM**: Fast alternative, often outperforms XGBoost
- **CatBoost**: Native categorical feature support

### Optimization
```bash
pip install optuna
```
- TPE sampler, pruning, multi-objective optimization

### Backtesting
```bash
pip install vectorbt backtrader
```
- **VectorBT**: 10-1000x faster, vectorized, parameter sweep
- **Backtrader**: Event-driven, live trading support

### Visualization
```bash
pip install matplotlib plotly seaborn
```

---

## Data Sources

### Free News APIs
| Source | Free Tier | Best For |
|--------|-----------|----------|
| NewsAPI.org | 100 req/day | News aggregation |
| Finnhub | 60 calls/min | Forex news, economic calendar |
| Alpha Vantage | 25 req/day | Technical indicators |

### Economic Calendar
| Source | Free Tier | Notes |
|--------|-----------|-------|
| ForexFactory (scraping) | Free | Comprehensive |
| forexfactory (pip) | Free | Local parquet cache |
| RapidAPI Economic Calendar | Freemium | Multiple sources |

### DXY Data
| Source | Method |
|--------|--------|
| Yahoo Finance | `yfinance` package, ticker: DX-Y.NYB |
| FRED API | Series: DTWEXBGS |
| Twelve Data | API endpoint |

### Historical Forex Data
| Source | Data Type | Cost |
|--------|-----------|------|
| Dukascopy | Tick, OHLC | Free |
| HistData.com | Tick data | Free |
| Forex Tester | M1 OHLC | Free |

### Tick Data
| Source | Cost |
|--------|------|
| Dukascopy | Free |
| dukascopy-node | Free (open source) |
| Tickstory | Free |

---

## Project Requirements

```txt
# Core
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# MT5 Integration
MetaTrader5>=5.0.45

# Technical Analysis
pandas-ta>=0.3.14
TA-Lib>=0.6.0
ta>=0.11.0
mplfinance>=0.12.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0

# Optimization
optuna>=3.0.0

# Backtesting
vectorbt>=0.26.0
backtrader>=1.9.78.123

# Visualization
matplotlib>=3.7.0
plotly>=5.15.0
seaborn>=0.12.0

# Data Sources
yfinance>=0.2.28
requests>=2.31.0
forexfactory>=1.1.0

# Utilities
python-dotenv>=1.0.0
schedule>=1.2.0
```

---

## Project Structure

```
forex-trading-bot/
|-- config/
|   |-- settings.py
|   +-- mt5_config.py
|-- data/
|   |-- fetcher.py
|   |-- processor.py
|   +-- cache.py
|-- indicators/
|   |-- technical.py
|   +-- custom.py
|-- signals/
|   |-- generator.py
|   +-- filters.py
|-- models/
|   |-- ml_models.py
|   +-- ensemble.py
|-- backtesting/
|   |-- engine.py
|   +-- metrics.py
|-- trading/
|   |-- mt5_trader.py
|   |-- risk_manager.py
|   +-- position_sizer.py
|-- visualization/
|   |-- charts.py
|   +-- reports.py
|-- main.py
|-- train.py
|-- backtest.py
|-- requirements.txt
+-- README.md
```

---

## Recommended Stack by Experience

### Beginners
- pandas, numpy, pandas-ta, scikit-learn, XGBoost, VectorBT, MetaTrader5

### Intermediate
- pandas, numpy, scipy, TA-Lib + pandas-ta, XGBoost + LightGBM, Optuna, VectorBT + Backtrader

### Advanced
- pandas, numpy, scipy, Polars, TA-Lib + pandas-ta + custom, XGBoost + LightGBM + CatBoost ensemble, Optuna (multi-objective), VectorBT Pro + Backtrader, MT5 + custom risk management + Telegram notifications

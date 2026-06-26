# EUR/USD 3:1 TP:SL Signal Generator

A **EUR/USD scalping signal generator** that outputs LONG/SHORT/NO-TRADE alerts via Telegram and Email with ATR-based dynamic TP/SL (minimum 3:1 ratio) during London-NY overlap only.

**This is a signal generator, NOT an execution bot.** No MT5 order sending, no position management, no automation of trades.

---

## Features

- **60 raw features** → 40 scaled (VarianceThreshold + MinMaxScaler)
- **Triple-barrier labeling** with ATR-based dynamic TP (4.5x ATR) / SL (1.5x ATR)
- **8 sequential rule-based filters** (session, ADX, body ratio, volume, MTF alignment)
- **ML ensemble**: XGBoost + LightGBM → Ridge meta-learner
- **Walk-forward validation** (10 windows, 6-month train + 1-month test)
- **8-factor confidence scoring** with threshold gating (default: 78/100)
- **Live dashboard** with real-time signal tracking, equity curve, and activity log

## Performance

| Metric | Value |
|--------|-------|
| Win Rate | 92.2% (backtest) |
| Signals/Day | 3.0 |
| TP:SL Ratio | 3:1 |
| TP | 45 pips (dynamic ATR) |
| SL | 15 pips (dynamic ATR) |
| Session | 13:00-16:00 UTC (18:00-21:00 PKT) |

## Project Structure

```
tradding_app/
├── src/
│   ├── data_pipeline.py          # MT5 connection (RPyC), data fetching, Parquet storage
│   ├── feature_engineering.py    # 60 raw features across 4 tiers
│   ├── feature_pipeline.py       # VarianceThreshold + MinMaxScaler
│   ├── labels.py                 # Triple-barrier labeling with ATR-based TP/SL
│   ├── rule_filters.py           # 8 sequential rule gates
│   ├── model_trainer.py          # XGBoost + LightGBM → Ridge meta (walk-forward)
│   ├── confidence_scorer.py      # 8-factor weighted confidence score
│   ├── signal_engine.py          # Main signal generation pipeline
│   ├── live_feature_engineering.py  # Single-bar feature computation
│   ├── backtest.py               # Simple backtester
│   ├── metrics.py                # Performance monitoring
│   └── notifications.py          # Telegram + Email notifiers
├── scripts/
│   ├── live_paper_trading.py     # Live signal generation loop
│   ├── dashboard_server.py       # HTTP server for dashboard
│   └── threshold_comparison.py   # Threshold optimization
├── frontend/
│   └── index.html                # Dashboard UI
├── config/
│   └── config.yaml               # System parameters
├── models/                       # Trained models (.gitignored)
├── data/                         # Market data (.gitignored)
├── docker-compose.yml            # MT5 Docker container
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start MT5 Docker

```bash
docker compose up -d
```

Start the RPyC server inside the container:
```bash
docker exec mt5 bash -c "su - abc -c 'wine python -c \"import rpyc; from rpyc.utils.server import ThreadedServer; from rpyc.core import SlaveService; t = ThreadedServer(SlaveService, hostname=\\\"0.0.0.0\\\", port=8001, reuse_addr=True); t.start()\" &'"
```

### 3. Fetch Data & Train

```bash
# Fetch M5/M15/H1/H4 data from MT5
python scripts/test_step2.py

# Train models (walk-forward)
python -c "from src.model_trainer import EnsembleTrainer; ..."
```

### 4. Run Live Paper Trading

```bash
# Terminal 1: Dashboard
python scripts/dashboard_server.py
# Open http://localhost:8080

# Terminal 2: Signal generator (runs during 13:00-16:00 UTC)
python scripts/live_paper_trading.py
```

## Configuration

Key parameters in `config/config.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tp_atr_multiplier` | 4.5 | TP = 4.5x ATR(14) |
| `sl_atr_multiplier` | 1.5 | SL = 1.5x ATR(14) |
| `min_tp_pips` | 9 | Minimum TP in pips |
| `max_tp_pips` | 45 | Maximum TP in pips |
| `min_sl_pips` | 3 | Minimum SL in pips |
| `max_sl_pips` | 15 | Maximum SL in pips |
| `session_start` | 13 | Trading session start (UTC) |
| `session_end` | 16 | Trading session end (UTC) |
| `min_confidence` | 78 | Minimum confidence threshold |

## Tech Stack

- **Data**: MT5 Docker (RPyC), pandas, pyarrow
- **ML**: XGBoost, LightGBM, scikit-learn
- **Backtesting**: Custom vectorized backtester
- **Dashboard**: Vanilla HTML/JS, Chart.js
- **Notifications**: Telegram Bot API, SMTP email

## License

Private - All rights reserved.

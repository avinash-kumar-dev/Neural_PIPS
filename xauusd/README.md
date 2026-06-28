# XAUUSD Gold Rule-Based Signal Bot

A rule-based XAUUSD (gold) signal generator with multi-module voting, structural 2:1+ R:R enforcement, and honest backtesting.

## Architecture

```
Layer 1 (Structure)     → H4 BOS/CHoCH + EMA 50/200 filter
Layer 2 (Entry)         → Order Blocks + FVG (holds + fills) + RSI/MACD
Layer 3 (Confirmation)  → Volume above MA + strong-bodied candle
Layer 4 (Risk)          → Structural SL, 2:1 R:R enforcement, tiered exits
Strategy Modules        → Trend/Structure, Breakout-Retest, Liquidity Sweep
Voting Layer            → 2-of-3 module agreement required
```

## Quick Start

```bash
# Fetch XAUUSD data from MT5
python3 scripts/fetch_xauusd_data.py

# Run backtest
python3 scripts/backtest_xauusd.py

# Run combined multi-module backtest
python3 scripts/backtest_combined.py
```

## Backtest Results (H4, 2020-2026)

| Module | Trades | Win Rate | Avg R:R | Profit Factor |
|--------|--------|----------|---------|---------------|
| Trend Only | 100 | 41.0% | 2.24 | 1.56 |
| Combined (2/3 vote) | 8 | 62.5% | 2.26 | 3.76 |

## Key Files

```
xauusd/
├── config.yaml           # All parameters
├── structure/            # Layer 1: SMC structure detection
│   ├── swing_points.py   # Swing high/low detection
│   ├── bos.py            # Break of Structure
│   ├── choch.py          # Change of Character
│   ├── ema_filter.py     # EMA 50/200 cross-check
│   └── layer1.py         # Combined Layer 1
├── entry/                # Layer 2: Entry triggers
│   ├── order_blocks.py   # OB with displacement
│   ├── fvg.py            # FVG holds + fills
│   ├── indicator_triggers.py  # RSI + MACD
│   └── layer2.py         # Combined Layer 2
├── confirmation/         # Layer 3: Filters
│   └── layer3.py         # Volume + candle body
├── risk/                 # Layer 4: Risk management
│   └── layer4.py         # SL, R:R, exits, sizing
├── strategies/           # Multi-module strategies
│   ├── trend_structure.py
│   ├── breakout_retest.py
│   ├── liquidity_sweep.py
│   └── voting.py         # Module voting layer
├── backtest/             # Backtesting engine
│   └── engine.py
├── signals/              # Signal generation
│   ├── generator.py
│   └── logger.py         # Audit logging
└── data/                 # Data storage
    └── raw/              # Parquet files
```

## Constraints

- **Minimum 2:1 R:R** on every trade (hard floor, reject if not met)
- **No martingale/grid** — fixed 3% risk per trade
- **Every signal logged** with which conditions fired
- **Realistic backtest costs**: 0.15 pips spread + $7/lot commission + 2 pips slippage
- **No ML in Phase 1** — all rule-based, deterministic, inspectable

## Limitations

1. **Tick volume only** — retail CFD gold provides tick volume, not true order flow
2. **Backtest vs live** — expect 10-20% performance degradation
3. **SMC crowding** — mitigated by displacement + confluence requirements
4. **No DXY/US10Y feed** — macro correlation not implemented yet

## Next Steps (Phase 2)

- [ ] M15/M5 execution timeframe testing
- [ ] DXY/US10Y correlation features
- [ ] ML meta-labeling (secondary model for setup quality)
- [ ] News event filter (ForexFactory calendar)
- [ ] Walk-forward optimization
- [ ] Paper trading period (1+ month)

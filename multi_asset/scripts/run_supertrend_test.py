import sys
sys.path.insert(0, "/home/avinash-kumar/tradding_app")

import pandas as pd
import numpy as np
from pathlib import Path
from multi_asset.data.indicators import compute_basic_indicators, detect_swing_points, detect_bos
from multi_asset.backtest.engine import MultiAssetEngine, EngineConfig, compute_metrics
import pandas_ta as ta

RESULTS_FILE = "/home/avinash-kumar/tradding_app/multi_asset/RESULTS_LOG.md"
DATA_DIR = "/home/avinash-kumar/tradding_app/multi_asset/data/raw"


def log_result(name: str, metrics: dict, config_note: str = ""):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n## {name}\n")
        f.write(f"> Config: {config_note}\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total Trades | {metrics['total_trades']} |\n")
        f.write(f"| Win Rate | {metrics['win_rate']*100:.1f}% |\n")
        f.write(f"| Total PnL | {metrics['total_pnl_pips']:+.1f} pips |\n")
        f.write(f"| Profit Factor | {metrics['profit_factor']:.2f} |\n")
        f.write(f"| R:R | {metrics['rr']:.2f} |\n")
        f.write(f"| Max Drawdown | {metrics['max_drawdown_pips']:.1f} pips |\n")
        f.write(f"| Avg Win | {metrics['avg_win']:.1f} pips |\n")
        f.write(f"| Avg Loss | {metrics['avg_loss']:.1f} pips |\n")
        if metrics.get("instruments"):
            f.write(f"\n### Per Instrument\n\n")
            f.write(f"| Instrument | Trades | WR | PnL |\n|------------|--------|-----|-----|\n")
            for inst, stats in metrics["instruments"].items():
                f.write(f"| {inst} | {stats['trades']} | {stats['wr']*100:.0f}% | {stats['pnl']:+.1f} |\n")
        f.write("\n---\n")


def compute_supertrend_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
    if st is not None:
        df["st_line"] = st.iloc[:, 0]
        df["st_dir"] = st.iloc[:, 1]
    else:
        df["st_line"] = np.nan
        df["st_dir"] = 0

    st2 = ta.supertrend(df["high"], df["low"], df["close"], length=20, multiplier=4.0)
    if st2 is not None:
        df["st2_line"] = st2.iloc[:, 0]
        df["st2_dir"] = st2.iloc[:, 1]
    else:
        df["st2_line"] = np.nan
        df["st2_dir"] = 0

    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)

    df["rsi"] = ta.rsi(df["close"], length=14)

    df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14).iloc[:, 0]

    return df


def generate_supertrend_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_supertrend_signals(df)
    df = detect_swing_points(df, lookback=5)
    df = detect_bos(df)

    n = len(df)
    entry_long = np.zeros(n, dtype=bool)
    entry_short = np.zeros(n, dtype=bool)
    entry_sl_long = np.full(n, np.nan)
    entry_sl_short = np.full(n, np.nan)

    st_dir = df["st_dir"].values
    st2_dir = df["st2_dir"].values
    ema_50 = df["ema_50"].values
    ema_200 = df["ema_200"].values
    rsi = df["rsi"].values
    adx = df["adx"].values
    close = df["close"].values
    low = df["low"].values
    high = df["high"].values
    atr = df["atr"].values
    bos_bull = df["bos_bull"].values
    bos_bear = df["bos_bear"].values
    bullish_engulfing = df["bullish_engulfing"].values
    bearish_engulfing = df["bearish_engulfing"].values
    pin_bar_bull = df["pin_bar_bull"].values
    pin_bar_bear = df["pin_bar_bear"].values

    for i in range(1, n):
        if np.isnan(ema_200[i]) or np.isnan(atr[i]) or atr[i] == 0:
            continue

        adx_val = adx[i] if not np.isnan(adx[i]) else 0

        long_trend = (close[i] > ema_200[i]) and (st_dir[i] == 1 or st2_dir[i] == 1)
        short_trend = (close[i] < ema_200[i]) and (st_dir[i] == -1 or st2_dir[i] == -1)

        long_confirm = (bos_bull[i] or bullish_engulfing[i] or pin_bar_bull[i])
        short_confirm = (bos_bear[i] or bearish_engulfing[i] or pin_bar_bear[i])

        if long_trend and long_confirm and adx_val > 20:
            if rsi[i] < 70:
                entry_long[i] = True
                entry_sl_long[i] = low[i] - atr[i] * 1.5

        if short_trend and short_confirm and adx_val > 20:
            if rsi[i] > 30:
                entry_short[i] = True
                entry_sl_short[i] = high[i] + atr[i] * 1.5

    df["entry_long"] = entry_long
    df["entry_short"] = entry_short
    df["entry_sl_long"] = entry_sl_long
    df["entry_sl_short"] = entry_sl_short
    df["setup_long"] = "supertrend"
    df["setup_short"] = "supertrend"

    return df


def run_on_instrument(name: str, pip_value: float):
    path = Path(DATA_DIR) / f"{name}_M5.parquet"
    df = pd.read_parquet(path)

    df = compute_basic_indicators(df, atr_period=14)
    df = generate_supertrend_signals(df)

    long_count = df["entry_long"].sum()
    short_count = df["entry_short"].sum()

    if long_count == 0 and short_count == 0:
        return {"total_trades": 0, "wins": 0, "losses": 0, "be": 0, "win_rate": 0,
                "avg_win": 0, "avg_loss": 0, "rr": 0, "total_pnl_pips": 0,
                "profit_factor": 0, "max_drawdown_pips": 0, "instruments": {}}

    signals = df[["entry_long", "entry_short", "entry_sl_long", "entry_sl_short",
                   "setup_long", "setup_short"]].copy()
    signals["pip_value"] = pip_value

    engine = MultiAssetEngine(EngineConfig(
        spread_pips=1.0,
        slippage_pips=0.5,
        min_rr=2.0,
        breakeven_trigger_r=2.0,
        trailing_start_r=3.0,
        trailing_step_r=1.0,
    ))

    trades = engine.run({name: df}, {name: signals})
    metrics = compute_metrics(trades)
    metrics["long_signals"] = int(long_count)
    metrics["short_signals"] = int(short_count)
    return metrics


if __name__ == "__main__":
    with open(RESULTS_FILE, "a") as f:
        f.write("\n\n---\n\n# Supertrend + EMA + BOS/CHoCH — Iteration 1\n\n")

    instruments = [
        ("US30", 1.0),
        ("US100", 1.0),
        ("EURUSD", 0.0001),
        ("GBPUSD", 0.0001),
        ("USDJPY", 0.01),
    ]

    results = {}
    for name, pip_val in instruments:
        print(f"\nSupertrend on {name}...")
        try:
            m = run_on_instrument(name, pip_val)
            results[name] = m
            print(f"  Signals: {m.get('long_signals',0)}L/{m.get('short_signals',0)}S")
            print(f"  Trades: {m['total_trades']}, WR: {m['win_rate']*100:.0f}%, PnL: {m['total_pnl_pips']:+.1f}, PF: {m['profit_factor']:.2f}, R:R: {m['rr']:.2f}")
            log_result(f"Supertrend {name} M5 (Iteration 1)", m,
                       "ST(10,3)+ST(20,4)+EMA200+BOS+Engulfing, ADX>20, RSI filter, ATR SL 1.5x")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("SUMMARY — Supertrend Iteration 1")
    print(f"{'='*60}")
    print(f"{'Instrument':<12} {'Signals':>8} {'Trades':>7} {'WR':>6} {'PnL':>10} {'PF':>6} {'R:R':>6}")
    print("-" * 60)
    for name, m in results.items():
        sigs = f"{m.get('long_signals',0)}L/{m.get('short_signals',0)}S"
        print(f"{name:<12} {sigs:>8} {m['total_trades']:>7} {m['win_rate']*100:>5.0f}% {m['total_pnl_pips']:>+9.1f} {m['profit_factor']:>6.2f} {m['rr']:>6.2f}")

import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class SessionConfig:
    primary: str = "06:00-21:00"
    best: str = "13:00-16:00"

    def parse(self, s: str):
        start, end = s.split("-")
        return int(start.split(":")[0]), int(end.split(":")[0])

    def primary_hours(self):
        return self.parse(self.primary)

    def best_hours(self):
        return self.parse(self.best)


@dataclass
class InstrumentConfig:
    name: str = ""
    mt5_symbol: str = ""
    pip_value: float = 0.0001
    pip_name: str = "pip"
    spread_max: float = 1.0
    sessions: SessionConfig = field(default_factory=SessionConfig)
    risk_pct: float = 1.0
    min_sl_pips: float = 20.0
    max_sl_pips: float = 100.0
    atr_period: int = 14
    setups: list = field(default_factory=list)


@dataclass
class GeneralConfig:
    max_concurrent_trades: int = 3
    max_portfolio_heat_pct: float = 5.0
    daily_loss_limit_pct: float = 3.0
    default_rr: float = 2.0
    min_rr: float = 2.0
    max_bars: int = 300
    stale_exit_bars: int = 80
    breakeven_trigger_r: float = 2.0
    trailing_start_r: float = 3.0
    trailing_step_r: float = 1.0


@dataclass
class DataConfig:
    htf: str = "H1"
    bias: str = "M15"
    entry: str = "M5"
    max_bars: int = 100000
    chunk_size: int = 50000


def load_config(path: str = None) -> dict:
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    instruments = {}
    for name, cfg in raw.get("instruments", {}).items():
        sess = SessionConfig(
            primary=cfg.get("sessions", {}).get("primary", "06:00-21:00"),
            best=cfg.get("sessions", {}).get("best", "13:00-16:00"),
        )
        instruments[name] = InstrumentConfig(
            name=name,
            mt5_symbol=cfg["mt5_symbol"],
            pip_value=cfg.get("pip_value", 0.0001),
            pip_name=cfg.get("pip_name", "pip"),
            spread_max=cfg.get("spread_max", 1.0),
            sessions=sess,
            risk_pct=cfg.get("risk_pct", 1.0),
            min_sl_pips=cfg.get("min_sl_pips", 20.0),
            max_sl_pips=cfg.get("max_sl_pips", 100.0),
            atr_period=cfg.get("atr_period", 14),
            setups=cfg.get("setups", []),
        )

    gen = raw.get("general", {})
    general = GeneralConfig(
        max_concurrent_trades=gen.get("max_concurrent_trades", 3),
        max_portfolio_heat_pct=gen.get("max_portfolio_heat_pct", 5.0),
        daily_loss_limit_pct=gen.get("daily_loss_limit_pct", 3.0),
        default_rr=gen.get("default_rr", 2.0),
        min_rr=gen.get("min_rr", 2.0),
        max_bars=gen.get("max_bars", 300),
        stale_exit_bars=gen.get("stale_exit_bars", 80),
        breakeven_trigger_r=gen.get("breakeven_trigger_r", 2.0),
        trailing_start_r=gen.get("trailing_start_r", 3.0),
        trailing_step_r=gen.get("trailing_step_r", 1.0),
    )

    data_raw = raw.get("data", {})
    data = DataConfig(
        htf=data_raw.get("timeframes", {}).get("htf", "H1"),
        bias=data_raw.get("timeframes", {}).get("bias", "M15"),
        entry=data_raw.get("timeframes", {}).get("entry", "M5"),
        max_bars=data_raw.get("max_bars", 100000),
        chunk_size=data_raw.get("chunk_size", 50000),
    )

    return {
        "instruments": instruments,
        "general": general,
        "data": data,
    }

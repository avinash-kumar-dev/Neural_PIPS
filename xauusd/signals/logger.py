import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Signal:
    timestamp: str
    direction: str
    entry_price: float
    sl: float
    tp1: float
    tp2: float
    trigger: str
    module: str
    confidence: float
    conditions: dict


class SignalLogger:
    def __init__(self, log_dir: str = "xauusd/data"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.signals_file = self.log_dir / "signals_log.json"

    def log(self, signal: Signal):
        signals = []
        if self.signals_file.exists():
            with open(self.signals_file, "r") as f:
                signals = json.load(f)
        signals.append(asdict(signal))
        with open(self.signals_file, "w") as f:
            json.dump(signals, f, indent=2)

    def load(self) -> list[dict]:
        if self.signals_file.exists():
            with open(self.signals_file, "r") as f:
                return json.load(f)
        return []

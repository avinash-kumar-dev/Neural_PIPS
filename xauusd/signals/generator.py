import pandas as pd
import numpy as np
from xauusd.strategies.voting import compute_voting_signals
from xauusd.strategies.trend_structure import run_trend_strategy
from xauusd.risk.layer4 import check_risk_reward, PIP_VALUE
from xauusd.signals.logger import Signal, SignalLogger


class SignalGenerator:
    def __init__(self, min_agreement: int = 2, min_rr: float = 2.0):
        self.min_agreement = min_agreement
        self.min_rr = min_rr
        self.logger = SignalLogger()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = compute_voting_signals(df, min_agreement=self.min_agreement)
        return signals

    def check_new_signals(self, df: pd.DataFrame, signals: pd.DataFrame) -> list[Signal]:
        new_signals = []
        n = len(signals)

        for i in range(max(0, n - 5), n):
            row = signals.iloc[i]
            direction = 0
            entry = df.iloc[i]["close"]
            sl = np.nan
            trigger = ""

            if row.get("long_entry", False):
                direction = 1
                sl = row.get("long_sl", np.nan)
                trigger = row.get("long_trigger", "")
            elif row.get("short_entry", False):
                direction = -1
                sl = row.get("short_sl", np.nan)
                trigger = row.get("short_trigger", "")

            if direction != 0 and not pd.isna(sl):
                check = check_risk_reward(entry, sl, None, min_rr=self.min_rr)
                if check.valid:
                    if direction == 1:
                        tp2 = entry + check.tp2_pips * PIP_VALUE
                        tp1 = entry + check.tp1_pips * PIP_VALUE
                    else:
                        tp2 = entry - check.tp2_pips * PIP_VALUE
                        tp1 = entry - check.tp1_pips * PIP_VALUE

                    signal = Signal(
                        timestamp=str(df.iloc[i]["datetime"]),
                        direction="LONG" if direction == 1 else "SHORT",
                        entry_price=round(entry, 2),
                        sl=round(sl, 2),
                        tp1=round(tp1, 2),
                        tp2=round(tp2, 2),
                        trigger=trigger,
                        module="combined",
                        confidence=float(row.get("long_votes", 0) + row.get("short_votes", 0)) / 3.0,
                        conditions={
                            "layer1_bias": int(row.get("layer1_bias", 0)),
                            "volume_confirmed": bool(row.get("layer3_confirmed", False)),
                        },
                    )
                    new_signals.append(signal)
                    self.logger.log(signal)

        return new_signals

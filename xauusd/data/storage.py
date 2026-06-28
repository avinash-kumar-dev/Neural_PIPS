import pandas as pd
from pathlib import Path
from typing import Optional


class ParquetStorage:
    def __init__(self, base_dir: str = "xauusd/data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save(self, df: pd.DataFrame, name: str) -> Path:
        path = self.raw_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def load(self, name: str) -> Optional[pd.DataFrame]:
        path = self.raw_dir / f"{name}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def exists(self, name: str) -> bool:
        return (self.raw_dir / f"{name}.parquet").exists()

    def list_datasets(self) -> list[str]:
        return [f.stem for f in self.raw_dir.glob("*.parquet")]

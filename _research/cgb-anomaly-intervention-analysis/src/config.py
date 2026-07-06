from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = ["load_config"]
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    assets_dir: Path = project_root / "assets"
    seed: int = 42
    debug: bool = False

    # Model / training defaults (override with CLI or config file)
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

# GatedFusionFormer Experiment Suite

"""
GatedFusionFormer Experiment Suite - core package initializer.

提供輕量共用工具、常用匯出與懶加載引用，避免在匯入 package 時觸發重型依賴。
"""

from pathlib import Path
import logging

__all__ = [
    "__version__",
    "get_logger",
    "DEFAULT_CONFIG",
    "build_model",
    "ModelExporter",
]

__version__ = "0.1.0"

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger

# 指向預設 config 檔（若不存在只是路徑參考）
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

# 懶加載常用 API：如果 models 未安裝或導入失敗則保留為 None
try:
    from ..models.factory import build_model  # type: ignore
except Exception:
    build_model = None

try:
    from ..models.exporter import ModelExporter  # type: ignore
except Exception:
    ModelExporter = None
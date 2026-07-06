"""
utils.py — 共享工具函數

提供：
  set_seed       — 固定全局隨機種子
  load_rml_data  — 從 pickle 加載 RML2016.10a 並返回 (X, Y, Z)
  build_loader   — 建立 DataLoader
  load_model     — 實例化 GatedFusionFormer 並加載權重
"""

import random
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import CLASSES, MODEL_CFG, DEVICE
from .dataset import RMLDataset
from .model import GatedFusionFormer


def set_seed(seed: int = 29) -> None:
    """固定 Python / NumPy / PyTorch 的隨機種子，確保實驗可復現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_rml_data(pkl_path: str, classes=CLASSES):
    """從 RML2016.10a pickle 文件加載數據。

    Args:
        pkl_path : RML2016.10a_dict.pkl 路徑。
        classes  : 類別名稱列表，決定標籤編碼順序。

    Returns:
        X (np.ndarray): shape (N, 2, 128)
        Y (np.ndarray): shape (N,)  整數類別標籤
        Z (np.ndarray): shape (N,)  SNR 值 (dB)
    """
    warnings.filterwarnings(
        "ignore",
        message="Input data is complex, switching to return_onesided=False"
    )
    with open(pkl_path, "rb") as f:
        data_dict = pd.read_pickle(f)

    X, Y, Z = [], [], []
    for (mod, snr_val), samples in data_dict.items():
        if mod not in classes:
            continue
        n = len(samples)
        X.append(samples)
        Y.extend([classes.index(mod)] * n)
        Z.extend([snr_val] * n)

    return np.concatenate(X, axis=0), np.array(Y), np.array(Z)


def build_loader(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                 batch_size: int, shuffle: bool = False) -> DataLoader:
    """從 (X, Y, Z) 陣列建立 DataLoader。"""
    dataset = RMLDataset(X, Y, Z)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


def load_model(weights_path: str,
               device: torch.device = DEVICE,
               model_cfg: dict = MODEL_CFG) -> GatedFusionFormer:
    """實例化 GatedFusionFormer 並加載訓練好的權重。

    Args:
        weights_path : .pth 權重文件路徑。
        device       : 計算設備。
        model_cfg    : 模型超參數字典 (與 config.MODEL_CFG 格式一致)。

    Returns:
        已加載權重且切換到 eval 模式的模型。
    """
    model = GatedFusionFormer(**model_cfg)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def ensure_dir(path: str) -> str:
    """確保目錄存在，若不存在則建立，並返回路徑。"""
    os.makedirs(path, exist_ok=True)
    return path

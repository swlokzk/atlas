from typing import Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def build_dataloaders(X_train: np.ndarray, y_train: np.ndarray, batch_size: int, X_val=None, y_val=None):
    Xt = torch.from_numpy(X_train).float()
    yt = torch.from_numpy(y_train).float()
    if yt.ndim == 1:
        yt = yt.unsqueeze(-1)
    train_ds = TensorDataset(Xt, yt)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = None
    if X_val is not None and y_val is not None:
        Xv = torch.from_numpy(X_val).float()
        yv = torch.from_numpy(y_val).float()
        if yv.ndim == 1:
            yv = yv.unsqueeze(-1)
        val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

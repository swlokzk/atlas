"""
dataset.py — RML2016.10a 多模態數據集

將原始 IQ 信號轉換為模型所需的三種模態：
  - IQ 模態  : 原始兩路信號       shape (2, T)
  - STFT 模態 : 短時傅里葉時頻圖   shape (1, 32, T)
  - S-TD 模態 : 二階統計特徵       shape (2, T)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal


# STFT 超參數（須與訓練時一致）
_STFT_NPERSEG  = 31
_STFT_NOVERLAP = 30
_STFT_NFFT     = 128
_STFT_FREQ_BINS = 32   # 取前 32 個頻率 bin


class RMLDataset(Dataset):
    """PyTorch Dataset，包裝 RML2016.10a 的多模態特徵轉換。

    Args:
        data  (np.ndarray): shape (N, 2, T) 的 IQ 原始信號。
        label (np.ndarray): shape (N,) 的整數類別標籤。
        snr   (np.ndarray): shape (N,) 的 SNR 值 (dB)。
    """

    def __init__(self, data: np.ndarray, label: np.ndarray, snr: np.ndarray):
        self.data  = data
        self.label = label
        self.snr   = snr

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        iq_signal = self.data[idx]           # (2, T)
        I, Q = iq_signal[0], iq_signal[1]

        # S-TD (二階統計) 模態
        std_signal = np.vstack((I**2 - Q**2, 2 * I * Q))   # (2, T)

        # STFT 模態
        x_complex = I + 1j * Q
        _, _, stft_raw = signal.stft(
            x_complex, fs=1.0, window='blackman',
            nperseg=_STFT_NPERSEG, noverlap=_STFT_NOVERLAP, nfft=_STFT_NFFT
        )
        stft_signal = np.expand_dims(np.abs(stft_raw)[:_STFT_FREQ_BINS, :], axis=0)  # (1, 32, T)

        return (
            torch.from_numpy(iq_signal).float(),
            torch.from_numpy(stft_signal).float(),
            torch.from_numpy(std_signal).float(),
            torch.tensor(self.label[idx], dtype=torch.long),
            torch.tensor(self.snr[idx],   dtype=torch.float),
        )

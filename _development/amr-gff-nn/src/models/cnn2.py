import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

"""
1. 原始標準 CNN2 模型 (Baseline)
"""
class CNN2(nn.Module):
    """原始標準 CNN2 模型，用於基準測試 (Baseline)。"""
    def __init__(self, num_classes: int):
        super(CNN2, self).__init__()
        # 典型的雙層卷積結構
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 128, 128) # 假設輸入長度固定為 128
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

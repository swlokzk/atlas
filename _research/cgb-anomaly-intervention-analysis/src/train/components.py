from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.transformer import build_model


def get_training_components(processed_info: Dict[str, Any], params: Dict[str, Any], device: Optional[str] = None):
    device = device or params.get("DEVICE", "cpu")
    feature_dim = processed_info.get("feature_dim", 1)
    model = build_model(params, feature_dim, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.get("LEARNING_RATE", 1e-3))
    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "device": device,
    }

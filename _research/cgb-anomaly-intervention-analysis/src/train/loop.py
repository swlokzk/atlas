from typing import Any
import torch


def _compute_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, criterion) -> torch.Tensor:
    prediction = model(x)
    if isinstance(prediction, dict):
        reconstruction = prediction.get("reconstruction")
        if reconstruction is None:
            reconstruction = prediction.get("x_recon")
        if reconstruction is None:
            raise TypeError("Model returned a dict without a reconstruction tensor")
        return criterion(reconstruction, x)
    target = y.squeeze(-1)
    return criterion(prediction, target)


def train_epoch(model: torch.nn.Module, loader, criterion, optimizer, device: str = "cpu") -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = _compute_loss(model, x, y, criterion)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(1, count)


def evaluate_model_simple(model: torch.nn.Module, loader, criterion, device: str = "cpu") -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            loss = _compute_loss(model, x, y, criterion)
            total_loss += loss.item()
            count += 1
    return total_loss / max(1, count)

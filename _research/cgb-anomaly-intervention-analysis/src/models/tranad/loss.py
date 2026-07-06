"""Loss functions for TranAD-style anomaly detection."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class TranADLoss(nn.Module):
    """Combine reconstruction and adversarial objectives."""

    def __init__(self, adversarial_weight: float = 1.0, reconstruction_weight: float = 1.0):
        super().__init__()
        self.adversarial_weight = adversarial_weight
        self.reconstruction_weight = reconstruction_weight
        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.BCELoss()

    def generator_loss(
        self,
        x_true: torch.Tensor,
        x_recon: torch.Tensor,
        normal_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        recon_loss = self.reconstruction_loss(x_recon, x_true)
        normal_target = torch.ones_like(normal_probability)
        adv_loss = self.classification_loss(normal_probability, normal_target)
        total_loss = self.reconstruction_weight * recon_loss + self.adversarial_weight * adv_loss
        return total_loss, {
            "recon_loss": float(recon_loss.detach().item()),
            "adv_loss": float(adv_loss.detach().item()),
            "generator_loss": float(total_loss.detach().item()),
        }

    def discriminator_loss(
        self,
        normal_probability: torch.Tensor,
        negative_probability: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        normal_target = torch.ones_like(normal_probability)
        negative_target = torch.zeros_like(negative_probability)
        normal_loss = self.classification_loss(normal_probability, normal_target)
        negative_loss = self.classification_loss(negative_probability, negative_target)
        total_loss = 0.5 * (normal_loss + negative_loss)
        return total_loss, {
            "disc_normal_loss": float(normal_loss.detach().item()),
            "disc_negative_loss": float(negative_loss.detach().item()),
            "discriminator_loss": float(total_loss.detach().item()),
        }

    def anomaly_score(
        self,
        x_true: torch.Tensor,
        x_recon: torch.Tensor,
        normal_probability: torch.Tensor,
    ) -> torch.Tensor:
        recon_error = torch.mean(torch.abs(x_recon - x_true), dim=(1, 2))
        discriminator_penalty = 1.0 - normal_probability.squeeze(-1)
        return recon_error + discriminator_penalty


__all__ = ["TranADLoss"]
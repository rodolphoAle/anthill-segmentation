"""Focal Loss for semantic segmentation.

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Down-weights easy examples (confidently correct predictions) so training
focuses on hard, ambiguous cases — exactly what is needed when reddish
soil confuses the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Weighted Focal Loss for semantic segmentation.

    Args:
        gamma: Focusing parameter. ``0.0`` = standard CrossEntropyLoss.
               ``2.0`` is the standard value from Lin et al. (2017).
        weight: Per-class weights tensor (same semantics as
            ``CrossEntropyLoss.weight``).
        ignore_index: Label value to ignore (e.g. ``255`` for boundary
            pixels).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        valid = targets != self.ignore_index
        return focal_loss[valid].mean()

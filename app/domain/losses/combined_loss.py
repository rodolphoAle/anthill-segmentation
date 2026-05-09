"""Combined Tversky + Focal + Lovász loss for segmentation training.

Pure Tversky/Dice losses collapse to "predict all background" during
early training when the class imbalance is extreme.  This class anchors
training with Focal Loss (per-pixel CE gradients + class weights) and
adds the Tversky term to push Recall.  When ``lovasz_weight > 0``, a
Lovász Hinge term directly optimises IoU:

    total = tversky_weight * TverskyLoss
          + lovasz_weight  * LovaszHingeLoss
          + focal_weight   * FocalLoss

where ``focal_weight = 1 - tversky_weight - lovasz_weight``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.domain.losses.focal_loss import FocalLoss
from app.domain.losses.lovasz_loss import LovaszHingeLoss
from app.domain.losses.tversky_loss import TverskyLoss


class CombinedTverskyFocalLoss(nn.Module):
    """Weighted combination of Tversky, Focal, and (optionally) Lovász losses.

    Args:
        tversky_alpha: FP weight in Tversky denominator.
        tversky_beta: FN weight in Tversky denominator.
        tversky_weight: Fraction of total loss assigned to Tversky ``[0, 1]``.
        lovasz_weight: Fraction of total loss assigned to Lovász ``[0, 1]``.
            ``0.0`` disables Lovász.
        focal_gamma: Focusing exponent for the Focal component.
        class_weights: Per-class weights tensor for the Focal component.
        ignore_index: Label value excluded from all components.
    """

    def __init__(
        self,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_weight: float = 0.5,
        lovasz_weight: float = 0.0,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if tversky_weight + lovasz_weight > 1.0:
            raise ValueError(
                f"tversky_weight ({tversky_weight}) + lovasz_weight ({lovasz_weight}) "
                f"must be <= 1.0 (focal weight = remainder)"
            )
        self._tversky = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            ignore_index=ignore_index,
        )
        self._focal = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            ignore_index=ignore_index,
        )
        self._lovasz = (
            LovaszHingeLoss(ignore_index=ignore_index)
            if lovasz_weight > 0
            else None
        )
        self._tversky_weight = tversky_weight
        self._lovasz_weight = lovasz_weight
        self._focal_weight = 1.0 - tversky_weight - lovasz_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = (
            self._tversky_weight * self._tversky(inputs, targets)
            + self._focal_weight * self._focal(inputs, targets)
        )
        if self._lovasz is not None:
            total = total + self._lovasz_weight * self._lovasz(inputs, targets)
        return total

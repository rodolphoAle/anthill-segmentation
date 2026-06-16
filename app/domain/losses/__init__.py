"""Loss functions for UNet segmentation training.

This package provides modular, composable loss functions designed for
heavily imbalanced binary segmentation (background vs. anthill).

Exports:
    FocalLoss: Per-pixel CE with down-weighting of easy examples.
    TverskyLoss: Explicitly optimises Recall via asymmetric FP/FN weighting.
    LovaszHingeLoss: Direct IoU surrogate for boundary-precise segmentation.
    CombinedTverskyFocalLoss: Weighted combination of all three components.
"""

from app.domain.losses.focal_loss import FocalLoss
from app.domain.losses.tversky_loss import TverskyLoss
from app.domain.losses.lovasz_loss import LovaszHingeLoss
from app.domain.losses.combined_loss import CombinedTverskyFocalLoss

__all__ = [
    "FocalLoss",
    "TverskyLoss",
    "LovaszHingeLoss",
    "CombinedTverskyFocalLoss",
]

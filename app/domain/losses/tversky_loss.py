"""Tversky Loss — penalises FN more than FP when beta > alpha.

TL = 1 - TP / (TP + alpha*FP + beta*FN)

With ``alpha=0.3, beta=0.7``: FN is weighted 2.3x more than FP, which
directly optimises Recall at the cost of some Precision.  Ideal when
missing a detection is costlier than a false alarm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """Tversky Loss for binary semantic segmentation.

    Args:
        alpha: Weight for FP in the denominator (lower -> more FP-tolerant).
        beta: Weight for FN in the denominator (higher -> stronger Recall push).
        smooth: Laplace smoothing to avoid division by zero.
        ignore_index: Label value excluded from the loss computation.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        anthill_prob = probs[:, 1, :, :]

        valid_mask = targets != self.ignore_index
        target_bin = (targets == 1).float()

        anthill_prob = anthill_prob[valid_mask]
        target_bin = target_bin[valid_mask]

        tp = (anthill_prob * target_bin).sum()
        fp = (anthill_prob * (1.0 - target_bin)).sum()
        fn = ((1.0 - anthill_prob) * target_bin).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky_index

"""Binary Lovász Hinge Loss — direct surrogate for IoU (Jaccard index).

Standard segmentation losses (CE, Focal, Tversky) optimise per-pixel
classification and only improve IoU as a side-effect.  Lovász Hinge is a
*piecewise-linear convex surrogate* of the Jaccard loss on the simplex of
foreground errors (Berman et al., CVPR 2018).

For 2-class segmentation the logit input is converted to a single-channel
margin: ``logit_anthill - logit_background``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LovaszHingeLoss(nn.Module):
    """Binary Lovász Hinge Loss for semantic segmentation.

    Args:
        ignore_index: Label value excluded from the loss (e.g. ``255``).
    """

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Gradient of the Lovász extension w.r.t. sorted errors."""
        p = gt_sorted.numel()
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1.0 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p].clone() - jaccard[0:-1].clone()
        return jaccard

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        margin = inputs[:, 1] - inputs[:, 0]

        valid = targets != self.ignore_index
        margin_v = margin[valid]
        target_v = (targets[valid] == 1).float()

        if margin_v.numel() == 0:
            return inputs.sum() * 0.0

        signs = 2.0 * target_v - 1.0
        errors = 1.0 - margin_v * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = target_v[perm]
        grad = self._lovasz_grad(gt_sorted)
        return torch.dot(F.relu(errors_sorted), grad)

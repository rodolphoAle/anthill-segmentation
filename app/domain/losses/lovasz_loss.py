"""Lovász Hinge Loss para segmentação binária.

Objetivo:
Otimizar diretamente a métrica IoU (Intersection over Union),
muito utilizada em segmentação semântica.

Diferente da CrossEntropy e Focal Loss,
a Lovász atua diretamente sobre os erros relacionados ao IoU.

Para segmentação binária:
    margin = logit_formigueiro - logit_fundo
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LovaszHingeLoss(nn.Module):
    "Implementação da Lovász Hinge Loss."

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()

        # Pixels ignorados no cálculo da loss.
        self.ignore_index = ignore_index

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        "Calcula gradiente da extensão Lovász."

        # Quantidade de elementos válidos.
        p = gt_sorted.numel()

        # Total de pixels positivos.
        gts = gt_sorted.sum()

        # Interseção acumulada.
        intersection = gts - gt_sorted.cumsum(0)

        # União acumulada.
        union = gts + (1.0 - gt_sorted).cumsum(0)

        # Aproximação do erro IoU/Jaccard.
        jaccard = 1.0 - intersection / union

        # Ajuste incremental do gradiente.
        if p > 1:
            jaccard[1:p] = (
                jaccard[1:p].clone()
                - jaccard[0:-1].clone()
            )

        return jaccard

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        "Calcula Lovász Hinge Loss."

        # Converte logits binários em margem.
        margin = inputs[:, 1] - inputs[:, 0]

        # Remove pixels ignorados.
        valid = targets != self.ignore_index

        margin_v = margin[valid]

        # Converte alvo para binário.
        target_v = (targets[valid] == 1).float()

        # Evita erro caso não existam pixels válidos.
        if margin_v.numel() == 0:
            return inputs.sum() * 0.0

        # Define sinal correto da classificação.
        signs = 2.0 * target_v - 1.0

        # Calcula erro da margem.
        errors = 1.0 - margin_v * signs

        # Ordena erros do maior para o menor.
        errors_sorted, perm = torch.sort(
            errors,
            descending=True
        )

        # Reordena targets.
        gt_sorted = target_v[perm]

        # Calcula gradiente Lovász.
        grad = self._lovasz_grad(gt_sorted)

        # Calcula loss final.
        return torch.dot(
            F.relu(errors_sorted),
            grad
        )
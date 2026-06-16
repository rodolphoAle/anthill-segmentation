"""Tversky Loss para segmentação semântica binária.

Objetivo:
Controlar o equilíbrio entre falsos positivos (FP)
e falsos negativos (FN).

Fórmula:

    TL =
        1 - TP / (TP + alpha*FP + beta*FN)

Com:
    alpha = 0.3
    beta  = 0.7

A loss penaliza mais falsos negativos,
aumentando o Recall do modelo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    "Implementação da Tversky Loss."

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()

        # Peso dos falsos positivos.
        self.alpha = alpha

        # Peso dos falsos negativos.
        self.beta = beta

        # Evita divisão por zero.
        self.smooth = smooth

        # Pixels ignorados no cálculo.
        self.ignore_index = ignore_index

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        "Calcula Tversky Loss."

        # Converte logits em probabilidades.
        probs = F.softmax(inputs, dim=1)

        # Seleciona probabilidade da classe formigueiro.
        anthill_prob = probs[:, 1, :, :]

        # Máscara de pixels válidos.
        valid_mask = targets != self.ignore_index

        # Converte target para binário.
        target_bin = (targets == 1).float()

        # Remove pixels ignorados.
        anthill_prob = anthill_prob[valid_mask]
        target_bin = target_bin[valid_mask]

        # True Positives.
        tp = (anthill_prob * target_bin).sum()

        # False Positives.
        fp = (anthill_prob * (1.0 - target_bin)).sum()

        # False Negatives.
        fn = ((1.0 - anthill_prob) * target_bin).sum()

        # Calcula índice Tversky.
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Retorna loss final.
        return 1.0 - tversky_index
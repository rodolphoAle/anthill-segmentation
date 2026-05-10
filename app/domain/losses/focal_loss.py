"""Focal Loss para segmentação semântica.

Objetivo:
Reduzir o impacto de exemplos fáceis e aumentar o foco
em regiões difíceis de segmentar.

Muito útil em datasets desbalanceados, onde o modelo
tende a aprender mais o fundo do que o formigueiro.

Fórmula:

    FL(p_t) =
        -(1 - p_t)^gamma * log(p_t)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    "Implementação da Focal Loss."

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()

        # Controla o foco em exemplos difíceis.
        # gamma = 0 equivale à CrossEntropy padrão.
        self.gamma = gamma

        # Pesos por classe para balanceamento.
        self.weight = weight

        # Pixels ignorados no cálculo da loss.
        self.ignore_index = ignore_index

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calcula a Focal Loss."""

        # Calcula CrossEntropy por pixel.
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # Probabilidade da predição correta.
        pt = torch.exp(-ce_loss)

        # Aplica fator focal.
        # Penaliza mais exemplos difíceis.
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Remove pixels ignorados.
        valid = targets != self.ignore_index

        # Retorna média final da loss.
        return focal_loss[valid].mean()
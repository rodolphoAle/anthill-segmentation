"""Loss combinada para segmentação semântica utilizando
Tversky Loss + Focal Loss + Lovász Loss.

Objetivo:
Melhorar o treinamento em cenários com forte desbalanceamento
entre fundo e formigueiro.

Problema encontrado:
Quando apenas Dice/Tversky eram utilizadas, o modelo tendia
a prever apenas fundo durante as primeiras épocas, devido
à predominância da classe majoritária no dataset.

Estratégia adotada:
- Focal Loss:
    Mantém gradientes mais estáveis por pixel e ajuda
    no aprendizado inicial das regiões minoritárias.

- Tversky Loss:
    Penaliza falsos negativos e aumenta o recall
    da classe formigueiro.

- Lovász Loss:
    Otimiza diretamente a métrica IoU,
    importante em segmentação semântica.

Fórmula:

    total =
        tversky_weight * TverskyLoss
      + lovasz_weight  * LovaszLoss
      + focal_weight   * FocalLoss

Onde:

    focal_weight =
        1 - tversky_weight - lovasz_weight
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.domain.losses.focal_loss import FocalLoss
from app.domain.losses.lovasz_loss import LovaszHingeLoss
from app.domain.losses.tversky_loss import TverskyLoss


class CombinedTverskyFocalLoss(nn.Module):
    """Combinação ponderada de múltiplas funções de perda.

    Componentes:
    - Tversky Loss:
        Focada em melhorar Recall e reduzir falsos negativos.

    - Focal Loss:
        Ajuda no aprendizado em datasets desbalanceados,
        enfatizando exemplos difíceis.

    - Lovász Loss:
        Aproxima otimização direta da métrica IoU.

    Args:
        tversky_alpha:
            Peso aplicado aos falsos positivos (FP).

        tversky_beta:
            Peso aplicado aos falsos negativos (FN).

        tversky_weight:
            Peso da Tversky Loss na loss total.

        lovasz_weight:
            Peso da Lovász Loss na loss total.
            Valor 0 desabilita essa componente.

        focal_gamma:
            Expoente da Focal Loss.
            Valores maiores aumentam foco em erros difíceis.

        class_weights:
            Pesos por classe para lidar com desbalanceamento.

        ignore_index:
            Valor ignorado durante o cálculo da loss.
            Geralmente utilizado em pixels inválidos da máscara.
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

        # Garante que a soma dos pesos não ultrapasse 1.
        # O restante será atribuído automaticamente à Focal Loss.
        if tversky_weight + lovasz_weight > 1.0:
            raise ValueError(
                f"tversky_weight ({tversky_weight}) + lovasz_weight ({lovasz_weight}) "
                f"must be <= 1.0 (focal weight = remainder)"
            )

        # Inicializa componente Tversky.
        # Utilizada para melhorar Recall da classe minoritária.
        self._tversky = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            ignore_index=ignore_index,
        )

        # Inicializa componente Focal.
        # Auxilia no aprendizado de exemplos difíceis.
        self._focal = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            ignore_index=ignore_index,
        )

        # Inicializa componente Lovász apenas se habilitada.
        # Essa loss aproxima otimização direta do IoU.
        self._lovasz = (
            LovaszHingeLoss(ignore_index=ignore_index)
            if lovasz_weight > 0
            else None
        )

        # Pesos utilizados na composição final.
        self._tversky_weight = tversky_weight
        self._lovasz_weight = lovasz_weight

        # O peso da Focal é calculado automaticamente
        # como o restante da composição.
        self._focal_weight = 1.0 - tversky_weight - lovasz_weight

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calcula a loss total combinada.

        Args:
            inputs:
                Predições do modelo (logits).

            targets:
                Máscaras reais do dataset.

        Returns:
            Valor escalar da loss combinada.
        """ 

        # Combinação principal:
        # Tversky + Focal
        total = (
            self._tversky_weight * self._tversky(inputs, targets)
            + self._focal_weight * self._focal(inputs, targets)
        )

        # Adiciona Lovász caso esteja habilitada.
        if self._lovasz is not None:
            total = total + self._lovasz_weight * self._lovasz(inputs, targets)

        return total
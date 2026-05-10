"""Dataset streaming para segmentação semântica.

Objetivo:
Carregar imagens e máscaras diretamente de um armazenamento remoto
sem salvar arquivos localmente.

As imagens são baixadas sob demanda, processadas em memória
e descartadas após o uso.

Isso reduz:
- uso de disco;
- consumo permanente de armazenamento;
- necessidade de datasets locais grandes.
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from app.domain.mask_utils import decode_rgb_mask_to_int64


class StreamingSegmentationDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, str]]
):
    """Dataset de segmentação baseado em streaming remoto."""

    def __init__(
        self,
        pairs: list[tuple[dict[str, str], dict[str, str]]],
        download_fn: Callable[[str], io.BytesIO],
        augmentations: transforms.Compose | None = None,
    ) -> None:
        # Lista de pares:
        # (imagem RGB, máscara).
        self._pairs = pairs

        # Função responsável pelo download remoto.
        self._download_fn = download_fn

        # Transformações sincronizadas.
        self._augmentations = augmentations

    # Interface do Dataset

    def __len__(self) -> int:
        """Retorna quantidade de amostras."""
        return len(self._pairs)

    # Normalização padrão ImageNet.
    # Aplicada após augmentations.
    _normalize = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def __getitem__(
        self,
        index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Carrega uma amostra remotamente."""

        # Metadados da imagem e máscara.
        rgb_meta, label_meta = self._pairs[index]

        # Download remoto
        

        # Baixa imagem RGB.
        rgb_buffer = self._download_fn(rgb_meta["id"])

        # Baixa máscara.
        label_buffer = self._download_fn(label_meta["id"])

        # Decodificação das imagens

        # Converte imagem RGB.
        image = Image.open(rgb_buffer).convert("RGB")

        # Converte máscara RGB.
        # Necessário para preservar canal vermelho
        # usado na anotação do formigueiro.
        mask = Image.open(label_buffer).convert("RGB")

        
        # Augmentations sincronizadas
        

        # Aplica transformações em imagem e máscara.
        if self._augmentations:
            image, mask = self._augmentations(image, mask)


        # Conversão para tensor

        # Normaliza imagem.
        image_tensor: torch.Tensor = self._normalize(image)

        # Converte máscara RGB para classes.
        label = decode_rgb_mask_to_int64(mask)

        # Converte máscara para tensor.
        mask_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        # Retorna:
        # imagem, máscara e nome do arquivo.
        return image_tensor, mask_tensor, rgb_meta["name"]
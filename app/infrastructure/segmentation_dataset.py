"""Dataset PyTorch para segmentação semântica.

Responsável por:
- carregar imagens RGB e máscaras locais;
- aplicar augmentations;
- aplicar normalização;
- converter máscaras RGB em classes;
- gerar tensores usados no treinamento.

Este dataset trabalha apenas com arquivos locais.
Downloads remotos devem ser realizados antes da criação do dataset.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from app.domain.mask_utils import (
    compute_ignore_pixel_pct,
    decode_rgb_mask_to_int64,
    has_anthill_pixels,
)
from app.infrastructure.augmentations import (
    apply_anthill_duplicate,
    apply_copy_paste,
)


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Dataset de segmentação baseado em arquivos locais."""

    # Extensões válidas para imagens RGB.
    _IMAGE_EXTENSIONS: frozenset[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".tif"}
    )

    # Extensão esperada para máscaras.
    _LABEL_EXTENSION: str = ".png"

    # Normalização padrão ImageNet.
    # Aplicada apenas na imagem RGB.
    _normalize = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    def __init__(
        self,
        rgb_dir: str | Path,
        labels_dir: str | Path,
        augmentations: transforms.Compose | None = None,
        image_only_transforms: transforms.Compose | None = None,
        preload: bool = False,
        copy_paste: bool = False,
        copy_paste_prob: float = 0.5,
        max_ignore_pixel_pct: float = 1.0,
        anthill_duplicate: bool = False,
        anthill_duplicate_prob: float = 0.7,
        anthill_duplicate_max_copies: int = 2,
    ) -> None:
        # Diretórios do dataset.
        self._rgb_dir = Path(rgb_dir)
        self._labels_dir = Path(labels_dir)

        # Transformações aplicadas em imagem e máscara.
        self._augmentations = augmentations

        # Transformações aplicadas apenas na imagem.
        self._image_only_transforms = image_only_transforms

        # Limite máximo de pixels ignorados.
        self._max_ignore_pixel_pct = max_ignore_pixel_pct

        # Lista de pares (imagem, máscara).
        self._pairs: list[tuple[Path, Path]] = self._match_pairs()

        # Configuração de Copy-Paste augmentation.
        self._copy_paste = copy_paste
        self._copy_paste_prob = copy_paste_prob

        # Índices de imagens positivas.
        self._positive_indices: list[int] = []

        if copy_paste:
            self._positive_indices = self._build_positive_index()

        # Configuração de duplicação de formigueiros.
        self._anthill_duplicate = anthill_duplicate
        self._anthill_duplicate_prob = anthill_duplicate_prob

        # Quantidade máxima de cópias.
        self._anthill_duplicate_max_copies = max(
            1,
            anthill_duplicate_max_copies
        )

        # Cache em RAM.
        # Evita leituras constantes do disco.
        self._cache: list[tuple[Image.Image, Image.Image]] | None = (
            self._preload_to_ram() if preload else None
        )

    # Métodos internos
    

    def _preload_to_ram(self) -> list[tuple[Image.Image, Image.Image]]:
        """Carrega todas as imagens para memória RAM."""

        cache: list[tuple[Image.Image, Image.Image]] = []

        for rgb_path, label_path in self._pairs:

            image = Image.open(rgb_path).convert("RGB")
            image.load()

            mask = Image.open(label_path).convert("RGB")
            mask.load()

            cache.append((image, mask))

        return cache

    def _match_pairs(self) -> list[tuple[Path, Path]]:
        """Relaciona imagens RGB com máscaras pelo nome."""

        # Indexa máscaras pelo nome do arquivo.
        label_by_stem = {
            f.stem: f
            for f in self._labels_dir.iterdir()
            if f.suffix.lower() == self._LABEL_EXTENSION
        }

        pairs: list[tuple[Path, Path]] = []

        # Procura imagens RGB correspondentes.
        for rgb_file in sorted(
            f for f in self._rgb_dir.iterdir()
            if f.suffix.lower() in self._IMAGE_EXTENSIONS
        ):
            label = label_by_stem.get(rgb_file.stem)

            if label is not None:
                pairs.append((rgb_file, label))

        # Sem filtro de pixels ignorados.
        if self._max_ignore_pixel_pct >= 1.0:
            return pairs

        kept: list[tuple[Path, Path]] = []

        dropped = 0

        # Remove máscaras com excesso de pixels ignorados.
        for rgb_path, label_path in pairs:

            arr = np.array(
                Image.open(label_path).convert("RGB")
            )

            ignore_pct = compute_ignore_pixel_pct(arr)

            if ignore_pct <= self._max_ignore_pixel_pct:
                kept.append((rgb_path, label_path))
            else:
                dropped += 1

        logger.info(
            "Tile filter: dropped {}/{} tiles with >{:.0%} ignore pixels "
            "(kept {} useful tiles)",
            dropped,
            len(pairs),
            self._max_ignore_pixel_pct,
            len(kept),
        )

        return kept

    def _build_positive_index(self) -> list[int]:
        """Cria lista de imagens contendo formigueiro."""

        positives: list[int] = []

        for i in range(len(self._pairs)):

            if self.has_anthill(i):
                positives.append(i)

        return positives

    def _load_pair(
        self,
        index: int
    ) -> tuple[Image.Image, Image.Image]:
        """Carrega imagem e máscara."""

        # Usa cache em RAM.
        if self._cache is not None:

            image, mask = self._cache[index]

            return image.copy(), mask.copy()

        # Carrega do disco.
        rgb_path, label_path = self._pairs[index]

        image = Image.open(rgb_path).convert("RGB")

        mask = Image.open(label_path).convert("RGB")

        return image, mask

    # Interface do Dataset

    def __len__(self) -> int:
        """Retorna quantidade total de amostras."""
        return len(self._pairs)

    def has_anthill(self, index: int) -> bool:
        """Verifica se máscara contém formigueiro."""

        _, label_path = self._pairs[index]

        mask_arr = np.array(
            Image.open(label_path).convert("RGB")
        )

        return has_anthill_pixels(mask_arr)

    def __getitem__(
        self,
        index: int
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Retorna uma amostra do dataset."""

        rgb_path, label_path = self._pairs[index]

        # Carrega imagem e máscara.
        image, mask = self._load_pair(index)

        # Copy-Paste augmentation

        # Insere formigueiros em imagens negativas.
        if (
            self._copy_paste
            and random.random() < self._copy_paste_prob
        ):
            if not self.has_anthill(index):

                donor_idx = (
                    random.choice(self._positive_indices)
                    if self._positive_indices
                    else index
                )

                donor_rgb, donor_mask = self._load_pair(donor_idx)

                image, mask = apply_copy_paste(
                    image,
                    mask,
                    donor_rgb,
                    donor_mask,
                )

        # Duplicação de formigueiros
        

        # Duplica formigueiros na própria imagem.
        if (
            self._anthill_duplicate
            and random.random() < self._anthill_duplicate_prob
        ):
            if self.has_anthill(index):

                image, mask = apply_anthill_duplicate(
                    image,
                    mask,
                    self._anthill_duplicate_max_copies,
                )

        # Augmentations sincronizadas

        # Aplica transformações em imagem e máscara.
        if self._augmentations:
            image, mask = self._augmentations(image, mask)

        # Transformações apenas na imagem.
        if self._image_only_transforms:
            image = self._image_only_transforms(image)

        # Conversão para tensor

        # Normaliza imagem.
        image_tensor: torch.Tensor = self._normalize(image)

        # Converte máscara RGB em classes.
        label = decode_rgb_mask_to_int64(mask)

        # Converte máscara para tensor.
        mask_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return image_tensor, mask_tensor, rgb_path.name
"""PyTorch ``Dataset`` for paired RGB + label segmentation data.

This dataset works exclusively with **local** files.  Any remote
downloads (Google Drive, S3, …) must be completed *before* the
dataset is instantiated — that responsibility belongs to
:class:`~app.service.data_service.DataService`.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger  # pyright: ignore[reportMissingImports]
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
    """Local-file segmentation dataset.

    Args:
        rgb_dir: Directory containing input RGB images.
        labels_dir: Directory containing label masks (PNG).
        augmentations: Optional torchvision v2 transforms applied to
            both image and mask jointly.
        image_only_transforms: Optional transforms applied to the image
            only (e.g. ColorJitter).
        preload: Load all images into RAM at startup.
        copy_paste: Enable cross-tile copy-paste augmentation.
        copy_paste_prob: Probability of applying copy-paste to negative tiles.
        max_ignore_pixel_pct: Drop tiles with more than this fraction
            of ignore pixels.
        anthill_duplicate: Enable intra-tile anthill duplication.
        anthill_duplicate_prob: Probability of applying duplication to
            positive tiles.
        anthill_duplicate_max_copies: Maximum rotated copies per tile.
    """

    _IMAGE_EXTENSIONS: frozenset[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".tif"}
    )
    _LABEL_EXTENSION: str = ".png"

    # ImageNet normalisation applied only to the image tensor.
    _normalize = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        self._rgb_dir = Path(rgb_dir)
        self._labels_dir = Path(labels_dir)
        self._augmentations = augmentations
        # Applied to the image PIL object only (e.g. ColorJitter)  NOT the mask.
        self._image_only_transforms = image_only_transforms
        self._max_ignore_pixel_pct = max_ignore_pixel_pct
        self._pairs: list[tuple[Path, Path]] = self._match_pairs()

        # Copy-paste augmentation: builds index of positive tiles at init.
        self._copy_paste = copy_paste
        self._copy_paste_prob = copy_paste_prob
        self._positive_indices: list[int] = []
        if copy_paste:
            self._positive_indices = self._build_positive_index()

        # Anthill self-duplication augmentation: rotated copies within the same tile.
        self._anthill_duplicate = anthill_duplicate
        self._anthill_duplicate_prob = anthill_duplicate_prob
        self._anthill_duplicate_max_copies = max(1, anthill_duplicate_max_copies)

        # When preload=True all PIL images are loaded once into RAM so
        # __getitem__ never touches the disk again  eliminates I/O spikes
        # during training when data is already on a local drive.
        self._cache: list[tuple[Image.Image, Image.Image]] | None = (
            self._preload_to_ram() if preload else None
        )

    #  private 

    def _preload_to_ram(self) -> list[tuple[Image.Image, Image.Image]]:
        """Load every image pair into RAM as PIL objects.

        Called once at construction time so ``__getitem__`` never reads
        from disk again during training.  Augmentations are still applied
        lazily (they are random, so they must run per-sample call).
        """
        cache: list[tuple[Image.Image, Image.Image]] = []
        for rgb_path, label_path in self._pairs:
            image = Image.open(rgb_path).convert("RGB")
            image.load()  # force full decode now, not lazily
            mask = Image.open(label_path).convert("RGB")
            mask.load()
            cache.append((image, mask))
        return cache

    def _match_pairs(self) -> list[tuple[Path, Path]]:
        """Match each RGB image to its label mask by exact stem name.

        When ``max_ignore_pixel_pct < 1.0`` each label is scanned and tiles
        whose ignore-pixel fraction exceeds the threshold are dropped.
        """
        label_by_stem = {
            f.stem: f
            for f in self._labels_dir.iterdir()
            if f.suffix.lower() == self._LABEL_EXTENSION
        }
        pairs: list[tuple[Path, Path]] = []
        for rgb_file in sorted(
            f for f in self._rgb_dir.iterdir()
            if f.suffix.lower() in self._IMAGE_EXTENSIONS
        ):
            label = label_by_stem.get(rgb_file.stem)
            if label is not None:
                pairs.append((rgb_file, label))

        if self._max_ignore_pixel_pct >= 1.0:
            return pairs

        kept: list[tuple[Path, Path]] = []
        dropped = 0
        for rgb_path, label_path in pairs:
            arr = np.array(Image.open(label_path).convert("RGB"))
            ignore_pct = compute_ignore_pixel_pct(arr)
            if ignore_pct <= self._max_ignore_pixel_pct:
                kept.append((rgb_path, label_path))
            else:
                dropped += 1
        logger.info(
            "Tile filter: dropped {}/{} tiles with >{:.0%} ignore pixels "
            "(kept {} useful tiles)",
            dropped, len(pairs), self._max_ignore_pixel_pct, len(kept),
        )
        return kept

    def _build_positive_index(self) -> list[int]:
        """Scan all label masks and return indices of tiles with anthill pixels."""
        positives: list[int] = []
        for i in range(len(self._pairs)):
            if self.has_anthill(i):
                positives.append(i)
        return positives

    def _load_pair(self, index: int) -> tuple[Image.Image, Image.Image]:
        """Load an (image, mask) pair from cache or disk."""
        if self._cache is not None:
            image, mask = self._cache[index]
            return image.copy(), mask.copy()
        rgb_path, label_path = self._pairs[index]
        image = Image.open(rgb_path).convert("RGB")
        mask = Image.open(label_path).convert("RGB")
        return image, mask

    #  Dataset interface

    def __len__(self) -> int:
        return len(self._pairs)

    def has_anthill(self, index: int) -> bool:
        """Return True if the label mask at *index* contains anthill pixels."""
        _, label_path = self._pairs[index]
        mask_arr = np.array(Image.open(label_path).convert("RGB"))
        return has_anthill_pixels(mask_arr)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rgb_path, label_path = self._pairs[index]

        image, mask = self._load_pair(index)

        # Copy-paste augmentation: if this tile has no anthill pixels,
        # paste a random anthill from a positive tile with some probability.
        if self._copy_paste and random.random() < self._copy_paste_prob:
            if not self.has_anthill(index):
                donor_idx = random.choice(self._positive_indices) if self._positive_indices else index
                donor_rgb, donor_mask = self._load_pair(donor_idx)
                image, mask = apply_copy_paste(image, mask, donor_rgb, donor_mask)

        # Anthill self-duplication: if this tile already has anthills, paste
        # rotated copies of them onto empty regions of the same tile.
        if self._anthill_duplicate and random.random() < self._anthill_duplicate_prob:
            if self.has_anthill(index):
                image, mask = apply_anthill_duplicate(
                    image, mask, self._anthill_duplicate_max_copies,
                )

        if self._augmentations:
            image, mask = self._augmentations(image, mask)

        if self._image_only_transforms:
            image = self._image_only_transforms(image)

        image_tensor: torch.Tensor = self._normalize(image)

        label = decode_rgb_mask_to_int64(mask)
        mask_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, mask_tensor, rgb_path.name

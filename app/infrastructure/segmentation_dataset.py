"""PyTorch ``Dataset`` for paired RGB + label segmentation data.

This dataset works exclusively with **local** files.  Any remote
downloads (Google Drive, S3, …) must be completed *before* the
dataset is instantiated — that responsibility belongs to
:class:`~app.service.data_service.DataService`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Local-file segmentation dataset.

    Args:
        rgb_dir: Directory containing input RGB images.
        labels_dir: Directory containing label masks (PNG).
        augmentations: Optional torchvision v2 transforms applied to
            both image and mask jointly.
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
    ) -> None:
        self._rgb_dir = Path(rgb_dir)
        self._labels_dir = Path(labels_dir)
        self._augmentations = augmentations
        # Applied to the image PIL object only (e.g. ColorJitter) — NOT the mask.
        self._image_only_transforms = image_only_transforms
        self._pairs: list[tuple[Path, Path]] = self._match_pairs()
        # When preload=True all PIL images are loaded once into RAM so
        # __getitem__ never touches the disk again — eliminates I/O spikes
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
        """Match each RGB image to its label mask by exact stem name."""
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
        return pairs

    #  Dataset interface 

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rgb_path, label_path = self._pairs[index]

        if self._cache is not None:
            image, mask = self._cache[index]
            # PIL images are mutable; copy so augmentations don't corrupt cache
            image = image.copy()
            mask = mask.copy()
        else:
            image = Image.open(rgb_path).convert("RGB")
            # Open mask as RGB to read the colour-coded annotation
            mask = Image.open(label_path).convert("RGB")

        if self._augmentations:
            image, mask = self._augmentations(image, mask)

        if self._image_only_transforms:
            image = self._image_only_transforms(image)

        image_tensor: torch.Tensor = self._normalize(image)

        # Label convention (same as StreamingSegmentationDataset):
        #   Red   (R>150, G<100, B<100) → class 1 (anthill)
        #   Black (all channels < 50)   → class 0 (background)
        #   White (all channels > 200)  → ignore_index=255 (unlabelled)
        mask_arr = np.array(mask)
        r, g, b = mask_arr[:, :, 0], mask_arr[:, :, 1], mask_arr[:, :, 2]
        is_anthill = (r > 150) & (g < 100) & (b < 100)
        is_background = (r < 50) & (g < 50) & (b < 50)

        label = np.full(mask_arr.shape[:2], 255, dtype=np.int64)
        label[is_background] = 0
        label[is_anthill] = 1

        mask_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, mask_tensor, rgb_path.name

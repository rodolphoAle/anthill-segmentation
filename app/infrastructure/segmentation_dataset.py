"""PyTorch ``Dataset`` for paired RGB + label segmentation data.

This dataset works exclusively with **local** files.  Any remote
downloads (Google Drive, S3, …) must be completed *before* the
dataset is instantiated — that responsibility belongs to
:class:`~app.service.data_service.DataService`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
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

    def __init__(
        self,
        rgb_dir: str | Path,
        labels_dir: str | Path,
        augmentations: transforms.Compose | None = None,
    ) -> None:
        self._rgb_dir = Path(rgb_dir)
        self._labels_dir = Path(labels_dir)
        self._augmentations = augmentations
        self._pairs: list[tuple[Path, Path]] = self._match_pairs()

    # ── private ──────────────────────────────────────────────────────

    def _match_pairs(self) -> list[tuple[Path, Path]]:
        """Match each RGB image to its corresponding label mask."""
        rgb_files = sorted(
            f
            for f in self._rgb_dir.iterdir()
            if f.suffix.lower() in self._IMAGE_EXTENSIONS
        )
        label_files = sorted(
            f
            for f in self._labels_dir.iterdir()
            if f.suffix.lower() == self._LABEL_EXTENSION
        )

        pairs: list[tuple[Path, Path]] = []
        for rgb_file in rgb_files:
            prefix = "_".join(rgb_file.stem.split("_")[:4])
            matched = [lf for lf in label_files if lf.stem.startswith(prefix)]
            if matched:
                pairs.append((rgb_file, matched[0]))
        return pairs

    # ── Dataset interface ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_path, label_path = self._pairs[index]

        image = Image.open(rgb_path).convert("RGB")
        mask = Image.open(label_path)

        if self._augmentations:
            image, mask = self._augmentations(image, mask)

        mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)
        mask_tensor = torch.clamp(mask_tensor, 0, 1)
        # PILToTensor adds a channel dim [1, H, W] — CrossEntropyLoss needs [H, W]
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.squeeze(0)

        return image, mask_tensor

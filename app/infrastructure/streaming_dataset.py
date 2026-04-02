"""In-memory streaming dataset for segmentation — zero local disk writes.

Images and masks are downloaded on-demand via a synchronous callable
(e.g. ``GoogleDriveClient._sync_download_file``) and immediately
released from memory after the item has been consumed by the DataLoader.
Nothing is ever written to disk.
"""

from __future__ import annotations

import io
from typing import Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class StreamingSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Segmentation dataset backed purely by remote file IDs.

    Each call to ``__getitem__`` fetches the raw bytes for that pair,
    decodes them in memory, applies transforms, and returns tensors.
    The buffers are discarded immediately afterwards.

    Args:
        pairs: List of ``(rgb_meta, label_meta)`` tuples.  Each dict
            must have at least the keys ``"id"`` and ``"name"``.
        download_fn: **Synchronous** callable that accepts a file-id
            string and returns an ``io.BytesIO`` buffer.  It is called
            directly inside ``__getitem__`` so it must *not* contain any
            ``asyncio`` code.
        augmentations: Optional torchvision v2 transforms applied
            jointly to the image and mask.

    Note:
        When using ``DataLoader(num_workers > 0)`` each worker process
        receives a pickled copy of this dataset.  The ``download_fn``
        (typically a bound method on ``GoogleDriveClient``) is also
        pickled — the Drive service inside the client is lazily
        re-authenticated in every worker the first time it is needed.
        If pickling causes issues, set ``num_workers=0``.
    """

    def __init__(
        self,
        pairs: list[tuple[dict[str, str], dict[str, str]]],
        download_fn: Callable[[str], io.BytesIO],
        augmentations: transforms.Compose | None = None,
    ) -> None:
        self._pairs = pairs
        self._download_fn = download_fn
        self._augmentations = augmentations

    #  Dataset interface 

    def __len__(self) -> int:
        return len(self._pairs)

    # Image-only normalization (ImageNet stats) applied after augmentations.
    _normalize = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rgb_meta, label_meta = self._pairs[index]

        rgb_buffer = self._download_fn(rgb_meta["id"])
        label_buffer = self._download_fn(label_meta["id"])

        image = Image.open(rgb_buffer).convert("RGB")
        # Open mask as RGB to preserve the red channel (anthill annotation)
        mask = Image.open(label_buffer).convert("RGB")

        # Apply geometric augmentations jointly (flip, rotate, etc.)
        if self._augmentations:
            image, mask = self._augmentations(image, mask)

        # Normalise image to float32 with ImageNet stats — mask is NOT touched
        image_tensor: torch.Tensor = self._normalize(image)

        mask_arr = np.array(mask)  # (H, W, 3)
        # Label convention (RGB masks):
        #   Red   (R>150, G<100, B<100) → class 1 (anthill)
        #   Black (all channels <  50) → class 0 (background)
        #   White (all channels > 200) → ignore_index=255 (unlabelled)
        r, g, b = mask_arr[:, :, 0], mask_arr[:, :, 1], mask_arr[:, :, 2]
        is_anthill = (r > 150) & (g < 100) & (b < 100)
        is_background = (r < 50) & (g < 50) & (b < 50)

        label = np.full(mask_arr.shape[:2], 255, dtype=np.int64)  # default: ignore
        label[is_background] = 0
        label[is_anthill] = 1

        mask_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, mask_tensor, rgb_meta["name"]

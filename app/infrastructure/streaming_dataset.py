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
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


class StreamingSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_meta, label_meta = self._pairs[index]

        rgb_buffer = self._download_fn(rgb_meta["id"])
        label_buffer = self._download_fn(label_meta["id"])

        image = Image.open(rgb_buffer).convert("RGB")
        mask = Image.open(label_buffer)

        if self._augmentations:
            image, mask = self._augmentations(image, mask)

        mask_tensor = torch.tensor(np.array(mask), dtype=torch.long)
        mask_tensor = torch.clamp(mask_tensor, 0, 1)
        # PILToTensor adds a channel dim [1, H, W] — CrossEntropyLoss needs [H, W]
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.squeeze(0)

        return image, mask_tensor

"""Shared mask encoding/decoding utilities.

The project uses RGB-encoded label masks where colours map to classes:

    Red   (R>150, G<100, B<100)  ->  class 1 (anthill)
    Black (all channels < 50)    ->  class 0 (background)
    White (all channels > 200)   ->  ``255`` (ignore / unlabelled border)

These helpers centralise the conversion logic so every consumer
(datasets, services, evaluation scripts) uses the same thresholds.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


# Label constants
LABEL_BACKGROUND: int = 0
LABEL_ANTHILL: int = 1
LABEL_IGNORE: int = 255


def decode_rgb_mask(mask_image: Image.Image) -> np.ndarray:
    """Convert an RGB label image into a class-id array.

    Args:
        mask_image: PIL Image in any mode (will be converted to RGB).

    Returns:
        2-D ``uint8`` array with values ``0`` (background),
        ``1`` (anthill), or ``255`` (ignore).
    """
    mask_arr = np.array(mask_image.convert("RGB"))

    r = mask_arr[:, :, 0]
    g = mask_arr[:, :, 1]
    b = mask_arr[:, :, 2]

    is_anthill = (r > 150) & (g < 100) & (b < 100)
    is_background = (r < 50) & (g < 50) & (b < 50)

    label = np.full(mask_arr.shape[:2], LABEL_IGNORE, dtype=np.uint8)
    label[is_background] = LABEL_BACKGROUND
    label[is_anthill] = LABEL_ANTHILL

    return label


def decode_rgb_mask_to_int64(mask_image: Image.Image) -> np.ndarray:
    """Same as :func:`decode_rgb_mask` but returns ``int64`` for PyTorch."""
    mask_arr = np.array(mask_image.convert("RGB"))

    r = mask_arr[:, :, 0]
    g = mask_arr[:, :, 1]
    b = mask_arr[:, :, 2]

    is_anthill = (r > 150) & (g < 100) & (b < 100)
    is_background = (r < 50) & (g < 50) & (b < 50)

    label = np.full(mask_arr.shape[:2], LABEL_IGNORE, dtype=np.int64)
    label[is_background] = LABEL_BACKGROUND
    label[is_anthill] = LABEL_ANTHILL

    return label


def has_anthill_pixels(mask_arr: np.ndarray) -> bool:
    """Check whether an RGB mask array contains any anthill pixels.

    Args:
        mask_arr: ``(H, W, 3)`` uint8 RGB array.

    Returns:
        ``True`` if any pixel matches the anthill colour threshold.
    """
    r, g, b = mask_arr[:, :, 0], mask_arr[:, :, 1], mask_arr[:, :, 2]
    return bool(np.any((r > 150) & (g < 100) & (b < 100)))


def get_anthill_binary_mask(mask_arr: np.ndarray) -> np.ndarray:
    """Extract a binary anthill mask from an RGB mask array.

    Args:
        mask_arr: ``(H, W, 3)`` uint8 RGB array.

    Returns:
        ``(H, W)`` uint8 array with ``1`` for anthill pixels, ``0`` otherwise.
    """
    r, g, b = mask_arr[:, :, 0], mask_arr[:, :, 1], mask_arr[:, :, 2]
    return ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8)


def compute_ignore_pixel_pct(mask_arr: np.ndarray) -> float:
    """Return the fraction of ignore (white) pixels in an RGB mask.

    Args:
        mask_arr: ``(H, W, 3)`` uint8 RGB array.

    Returns:
        Float in ``[0, 1]`` representing the proportion of white pixels.
    """
    r, g, b = mask_arr[:, :, 0], mask_arr[:, :, 1], mask_arr[:, :, 2]
    return float(((r > 200) & (g > 200) & (b > 200)).mean())

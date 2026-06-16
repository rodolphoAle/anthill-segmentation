"""Data augmentation builders and augmentation strategies.

This module centralises all augmentation logic:

* **Transform builders**: ``create_train_transforms`` and
  ``create_image_only_transforms`` produce torchvision v2 transform
  pipelines from the global settings.
* **Copy-paste augmentation**: Paste random anthill regions from
  positive tiles onto negative tiles.
* **Anthill self-duplication**: Rotate and paste existing anthills
  within the same tile to multiply positive pixels.

The augmentation strategies are implemented as standalone functions
so they can be reused across datasets and preview scripts.
"""

from __future__ import annotations

import random

import numpy as np
import torchvision.transforms.v2 as transforms
from PIL import Image
from scipy import ndimage

from app.core.config import settings
from app.domain.mask_utils import get_anthill_binary_mask


# Donor size constraints for copy-paste / anthill-duplicate.
_MIN_DONOR_PX: int = 30
_MAX_DONOR_PX: int = 5000

# Gaussian alpha sigma for softening paste boundaries.
_PASTE_ALPHA_SIGMA: float = 1.5


# ---------------------------------------------------------------------------
#  Transform builders
# ---------------------------------------------------------------------------

def create_train_transforms() -> transforms.Compose | None:
    """Build geometric augmentations applied jointly to image AND mask.

    Returns ``None`` if all augmentations are disabled.
    """
    transform_list: list[transforms.Transform] = []

    if settings.aug_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if settings.aug_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip())

    if settings.aug_random_rotate_90:
        transform_list.append(
            transforms.RandomApply(
                [transforms.RandomChoice([
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270)),
                ])],
                p=0.5,
            )
        )

    if settings.aug_rotation_degrees > 0:
        transform_list.append(
            transforms.RandomRotation(settings.aug_rotation_degrees)
        )

    if settings.aug_elastic_transform:
        transform_list.append(
            transforms.RandomApply(
                [transforms.ElasticTransform(
                    alpha=settings.aug_elastic_alpha,
                    sigma=settings.aug_elastic_sigma,
                )],
                p=0.3,
            )
        )

    if not transform_list:
        return None

    return transforms.Compose(transform_list)


def create_image_only_transforms() -> transforms.Compose | None:
    """Photometric augmentations applied to the image tensor only (not mask).

    Returns ``None`` if colour jitter is disabled.
    """
    if not settings.aug_color_jitter:
        return None
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=settings.aug_color_jitter_brightness,
            contrast=settings.aug_color_jitter_contrast,
            saturation=settings.aug_color_jitter_saturation,
        )
    ])


# ---------------------------------------------------------------------------
#  Anthill region extraction (shared by copy-paste and anthill-duplicate)
# ---------------------------------------------------------------------------

def extract_anthill_region(
    source_rgb: Image.Image,
    source_mask: Image.Image,
    min_donor_px: int = _MIN_DONOR_PX,
    max_donor_px: int = _MAX_DONOR_PX,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract a random anthill connected component with its bounding box.

    Returns ``(rgb_crop, binary_mask_crop)`` as numpy arrays, or ``None``
    if no valid anthill region is found.
    """
    mask_arr = np.array(source_mask)
    anthill_binary = get_anthill_binary_mask(mask_arr)

    if anthill_binary.sum() == 0:
        return None

    labelled, num_features = ndimage.label(anthill_binary)
    if num_features == 0:
        return None

    valid_components = [
        cid for cid in range(1, num_features + 1)
        if min_donor_px <= int((labelled == cid).sum()) <= max_donor_px
    ]
    if not valid_components:
        return None

    component_id = random.choice(valid_components)
    component_mask = (labelled == component_id).astype(np.uint8)

    rows = np.any(component_mask, axis=1)
    cols = np.any(component_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    pad = 5
    h, w = mask_arr.shape[:2]
    rmin = max(0, rmin - pad)
    rmax = min(h - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w - 1, cmax + pad)

    rgb_arr = np.array(source_rgb)
    rgb_crop = rgb_arr[rmin:rmax + 1, cmin:cmax + 1]
    mask_crop = component_mask[rmin:rmax + 1, cmin:cmax + 1]

    return rgb_crop, mask_crop


def _random_transform_crop(
    rgb_crop: np.ndarray,
    mask_crop: np.ndarray,
    *,
    allow_identity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random flip + 90-degree rotation to a crop for diversity."""
    if random.random() < 0.5:
        rgb_crop = np.ascontiguousarray(np.fliplr(rgb_crop))
        mask_crop = np.ascontiguousarray(np.fliplr(mask_crop))
    if random.random() < 0.5:
        rgb_crop = np.ascontiguousarray(np.flipud(rgb_crop))
        mask_crop = np.ascontiguousarray(np.flipud(mask_crop))
    k_min = 0 if allow_identity else 1
    k = random.randint(k_min, 3)
    if k > 0:
        rgb_crop = np.ascontiguousarray(np.rot90(rgb_crop, k))
        mask_crop = np.ascontiguousarray(np.rot90(mask_crop, k))
    return rgb_crop, mask_crop


def _paste_with_alpha_blending(
    target_arr: np.ndarray,
    target_mask_arr: np.ndarray,
    rgb_crop: np.ndarray,
    mask_crop: np.ndarray,
    y: int,
    x: int,
    sigma: float = _PASTE_ALPHA_SIGMA,
) -> None:
    """Paste a crop onto a target with Gaussian-soft alpha blending.

    Modifies ``target_arr`` and ``target_mask_arr`` in place.
    """
    crop_h, crop_w = rgb_crop.shape[:2]

    alpha = ndimage.gaussian_filter(
        mask_crop.astype(np.float32), sigma=sigma,
    )
    alpha_max = float(alpha.max())
    if alpha_max > 0:
        alpha = np.clip(alpha / alpha_max, 0.0, 1.0)
    alpha_3c = alpha[..., None]

    target_patch = target_arr[y:y + crop_h, x:x + crop_w].astype(np.float32)
    donor_patch = rgb_crop.astype(np.float32)
    blended = alpha_3c * donor_patch + (1.0 - alpha_3c) * target_patch
    target_arr[y:y + crop_h, x:x + crop_w] = np.clip(
        blended, 0, 255
    ).astype(np.uint8)

    paste_region = mask_crop.astype(bool)
    target_mask_arr[y:y + crop_h, x:x + crop_w, 0] = np.where(
        paste_region, 255,
        target_mask_arr[y:y + crop_h, x:x + crop_w, 0],
    )
    target_mask_arr[y:y + crop_h, x:x + crop_w, 1] = np.where(
        paste_region, 0,
        target_mask_arr[y:y + crop_h, x:x + crop_w, 1],
    )
    target_mask_arr[y:y + crop_h, x:x + crop_w, 2] = np.where(
        paste_region, 0,
        target_mask_arr[y:y + crop_h, x:x + crop_w, 2],
    )


# ---------------------------------------------------------------------------
#  Copy-paste augmentation
# ---------------------------------------------------------------------------

def apply_copy_paste(
    target_rgb: Image.Image,
    target_mask: Image.Image,
    donor_rgb: Image.Image,
    donor_mask: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    """Paste a random anthill region from a donor tile onto the target.

    Only anthill pixels are pasted (alpha-masked), preserving the
    target background.

    Args:
        target_rgb: Target RGB image (typically a negative tile).
        target_mask: Target label mask.
        donor_rgb: Positive tile to extract the anthill from.
        donor_mask: Donor label mask.

    Returns:
        Tuple of ``(augmented_rgb, augmented_mask)``.
    """
    extraction = extract_anthill_region(donor_rgb, donor_mask)
    if extraction is None:
        return target_rgb, target_mask

    rgb_crop, mask_crop = extraction
    rgb_crop, mask_crop = _random_transform_crop(rgb_crop, mask_crop)

    crop_h, crop_w = rgb_crop.shape[:2]
    target_arr = np.array(target_rgb)
    target_mask_arr = np.array(target_mask)
    th, tw = target_arr.shape[:2]

    if crop_h >= th or crop_w >= tw:
        return target_rgb, target_mask

    y = random.randint(0, th - crop_h)
    x = random.randint(0, tw - crop_w)

    _paste_with_alpha_blending(
        target_arr, target_mask_arr, rgb_crop, mask_crop, y, x,
    )

    return Image.fromarray(target_arr), Image.fromarray(target_mask_arr)


# ---------------------------------------------------------------------------
#  Anthill self-duplication augmentation
# ---------------------------------------------------------------------------

def apply_anthill_duplicate(
    target_rgb: Image.Image,
    target_mask: Image.Image,
    max_copies: int = 2,
) -> tuple[Image.Image, Image.Image]:
    """Duplicate existing anthills within the same tile, rotated.

    Unlike copy-paste (which mixes anthills across tiles), this
    augmentation keeps the anthill in its native context: same lighting,
    soil type, and surrounding vegetation.

    Args:
        target_rgb: Positive tile RGB image.
        target_mask: Positive tile label mask.
        max_copies: Maximum number of rotated copies per call.

    Returns:
        Tuple of ``(augmented_rgb, augmented_mask)``.
    """
    target_arr = np.array(target_rgb)
    target_mask_arr = np.array(target_mask)
    th, tw = target_arr.shape[:2]

    anthill_binary = get_anthill_binary_mask(target_mask_arr)
    if anthill_binary.sum() == 0:
        return target_rgb, target_mask

    labelled, num_features = ndimage.label(anthill_binary)
    if num_features == 0:
        return target_rgb, target_mask

    valid_components = [
        cid for cid in range(1, num_features + 1)
        if _MIN_DONOR_PX <= int((labelled == cid).sum()) <= _MAX_DONOR_PX
    ]
    if not valid_components:
        return target_rgb, target_mask

    occupied = anthill_binary.astype(bool).copy()
    num_copies = random.randint(1, max(1, max_copies))

    for _ in range(num_copies):
        component_id = random.choice(valid_components)
        component_mask = (labelled == component_id).astype(np.uint8)

        rows = np.any(component_mask, axis=1)
        cols = np.any(component_mask, axis=0)
        if not rows.any() or not cols.any():
            continue
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        pad = 5
        rmin = max(0, rmin - pad)
        rmax = min(th - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(tw - 1, cmax + pad)

        rgb_crop = target_arr[rmin:rmax + 1, cmin:cmax + 1].copy()
        mask_crop = component_mask[rmin:rmax + 1, cmin:cmax + 1].copy()

        # Force non-zero rotation
        rgb_crop, mask_crop = _random_transform_crop(
            rgb_crop, mask_crop, allow_identity=False,
        )

        crop_h, crop_w = rgb_crop.shape[:2]
        if crop_h >= th or crop_w >= tw:
            continue

        # Try placements that don't overlap with existing anthills
        placement = None
        for _attempt in range(10):
            y = random.randint(0, th - crop_h)
            x = random.randint(0, tw - crop_w)
            placement_mask = mask_crop.astype(bool)
            if not (occupied[y:y + crop_h, x:x + crop_w] & placement_mask).any():
                placement = (y, x)
                break
        if placement is None:
            continue
        y, x = placement

        _paste_with_alpha_blending(
            target_arr, target_mask_arr, rgb_crop, mask_crop, y, x,
        )
        occupied[y:y + crop_h, x:x + crop_w] |= mask_crop.astype(bool)

    return Image.fromarray(target_arr), Image.fromarray(target_mask_arr)

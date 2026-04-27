"""PyTorch ``Dataset`` for paired RGB + label segmentation data.

This dataset works exclusively with **local** files.  Any remote
downloads (Google Drive, S3, …) must be completed *before* the
dataset is instantiated  that responsibility belongs to
:class:`~app.service.data_service.DataService`.

Copy-Paste Augmentation
-----------------------
When ``copy_paste=True`` the dataset builds an index of all tiles that
contain anthill pixels at construction time.  During ``__getitem__``,
if the sampled tile is **negative** (no anthill), a random anthill
region is cut from a positive tile and pasted onto it with probability
``copy_paste_prob``.  This creates genuinely new training examples
instead of merely repeating existing positive tiles.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from loguru import logger  # pyright: ignore[reportMissingImports]
from PIL import Image
from scipy import ndimage  # pyright: ignore[reportMissingImports]
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

    # Copy-paste donor size constraints.  Anthills smaller than 30px are noise
    # (single-pixel artefacts in labels); larger than 5000px are usually mask
    # errors that smear over crop rows or roads  pasting them teaches the
    # model the wrong shape prior.  5000px matches settings.max_anthill_region_px.
    _MIN_DONOR_PX: int = 30
    _MAX_DONOR_PX: int = 5000

    # Sigma (in pixels) of the Gaussian alpha used to soften paste boundaries.
    # ~1.5px feathers a 3-4px transition zone around the donor, breaking the
    # hard "cutout" silhouette that the model can otherwise learn as a shortcut.
    _PASTE_ALPHA_SIGMA: float = 1.5

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
        whose ignore-pixel fraction exceeds the threshold are dropped.  These
        tiles (typically 70-95% white border padding) provide negligible
        supervision signal but consume one full forward/backward pass per epoch.
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
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            ignore_pct = float(((r > 200) & (g > 200) & (b > 200)).mean())
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

    @staticmethod
    def _extract_anthill_region(
        source_rgb: Image.Image,
        source_mask: Image.Image,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract a random anthill connected component with its bounding box.

        Returns (rgb_crop, binary_mask_crop) as numpy arrays, or None if no
        valid anthill region is found.
        """
        mask_arr = np.array(source_mask)
        r, g, b = mask_arr[:, :, 0], mask_arr[:, :, 1], mask_arr[:, :, 2]
        anthill_binary = ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8)

        if anthill_binary.sum() == 0:
            return None

        labelled, num_features = ndimage.label(anthill_binary)
        if num_features == 0:
            return None

        # Filter components by size  reject noise (<30px) and oversized mask
        # errors (>5000px).  Pick a random component from the valid pool.
        valid_components = [
            cid for cid in range(1, num_features + 1)
            if SegmentationDataset._MIN_DONOR_PX
            <= int((labelled == cid).sum())
            <= SegmentationDataset._MAX_DONOR_PX
        ]
        if not valid_components:
            return None

        component_id = random.choice(valid_components)
        component_mask = (labelled == component_id).astype(np.uint8)

        # Get bounding box
        rows = np.any(component_mask, axis=1)
        cols = np.any(component_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add small padding (5px) for natural blending
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

    def _apply_copy_paste(
        self,
        target_rgb: Image.Image,
        target_mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        """Paste a random anthill region from a positive tile onto the target.

        The anthill is placed at a random valid position.  Only anthill
        pixels are pasted (alpha-masked), so the background is preserved.
        """
        if not self._positive_indices:
            return target_rgb, target_mask

        donor_idx = random.choice(self._positive_indices)
        donor_rgb, donor_mask = self._load_pair(donor_idx)

        extraction = self._extract_anthill_region(donor_rgb, donor_mask)
        if extraction is None:
            return target_rgb, target_mask

        rgb_crop, mask_crop = extraction

        # P3: random flip + 90° rotation of the donor for diversity.
        # The 600 positive tiles become ~4800 effective variants without I/O.
        if random.random() < 0.5:
            rgb_crop = np.ascontiguousarray(np.fliplr(rgb_crop))
            mask_crop = np.ascontiguousarray(np.fliplr(mask_crop))
        if random.random() < 0.5:
            rgb_crop = np.ascontiguousarray(np.flipud(rgb_crop))
            mask_crop = np.ascontiguousarray(np.flipud(mask_crop))
        k = random.randint(0, 3)
        if k > 0:
            rgb_crop = np.ascontiguousarray(np.rot90(rgb_crop, k))
            mask_crop = np.ascontiguousarray(np.rot90(mask_crop, k))

        crop_h, crop_w = rgb_crop.shape[:2]

        target_arr = np.array(target_rgb)
        target_mask_arr = np.array(target_mask)
        th, tw = target_arr.shape[:2]

        if crop_h >= th or crop_w >= tw:
            return target_rgb, target_mask

        # Random placement
        y = random.randint(0, th - crop_h)
        x = random.randint(0, tw - crop_w)

        # P1: Gaussian-soft alpha for RGB blending  eliminates the hard cutout
        # silhouette that lets the model learn "pasted boundary" as a shortcut.
        # The label mask stays binary (hard ground truth); only the visible
        # image is feathered so the supervision signal remains sharp.
        alpha = ndimage.gaussian_filter(
            mask_crop.astype(np.float32),
            sigma=SegmentationDataset._PASTE_ALPHA_SIGMA,
        )
        alpha_max = float(alpha.max())
        if alpha_max > 0:
            alpha = np.clip(alpha / alpha_max, 0.0, 1.0)
        alpha_3c = alpha[..., None]  # (H, W, 1) for broadcast

        target_patch = target_arr[y:y + crop_h, x:x + crop_w].astype(np.float32)
        donor_patch = rgb_crop.astype(np.float32)
        blended = alpha_3c * donor_patch + (1.0 - alpha_3c) * target_patch
        target_arr[y:y + crop_h, x:x + crop_w] = np.clip(blended, 0, 255).astype(np.uint8)

        # Label mask paint: hard binary, using the original component mask.
        paste_region = mask_crop.astype(bool)
        target_mask_arr[y:y + crop_h, x:x + crop_w, 0] = np.where(
            paste_region, 255, target_mask_arr[y:y + crop_h, x:x + crop_w, 0],
        )
        target_mask_arr[y:y + crop_h, x:x + crop_w, 1] = np.where(
            paste_region, 0, target_mask_arr[y:y + crop_h, x:x + crop_w, 1],
        )
        target_mask_arr[y:y + crop_h, x:x + crop_w, 2] = np.where(
            paste_region, 0, target_mask_arr[y:y + crop_h, x:x + crop_w, 2],
        )

        return Image.fromarray(target_arr), Image.fromarray(target_mask_arr)

    def _apply_anthill_duplicate(
        self,
        target_rgb: Image.Image,
        target_mask: Image.Image,
        max_copies: int = 2,
    ) -> tuple[Image.Image, Image.Image]:
        """Duplicate existing anthills within the same tile, rotated.

        Unlike copy-paste (which mixes anthills across tiles), this augmentation
        keeps the anthill in its native context: same lighting, soil type, and
        surrounding vegetation.  Each duplicate is rotated 90/180/270 (and may
        be flipped) so that the model sees the same anthill from new orientations.

        Steps:
            1. Detect anthill connected components in the current label.
            2. Filter by size (same limits as copy-paste: 30..5000 px).
            3. Sample 1..max_copies duplicates of randomly selected components.
            4. For each duplicate, rotate + optional flip, then place on a
               non-overlapping random position with Gaussian-soft alpha.
        """
        target_arr = np.array(target_rgb)
        target_mask_arr = np.array(target_mask)
        th, tw = target_arr.shape[:2]

        r = target_mask_arr[:, :, 0]
        g = target_mask_arr[:, :, 1]
        b = target_mask_arr[:, :, 2]
        anthill_binary = ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8)
        if anthill_binary.sum() == 0:
            return target_rgb, target_mask

        labelled, num_features = ndimage.label(anthill_binary)
        if num_features == 0:
            return target_rgb, target_mask

        valid_components = [
            cid for cid in range(1, num_features + 1)
            if SegmentationDataset._MIN_DONOR_PX
            <= int((labelled == cid).sum())
            <= SegmentationDataset._MAX_DONOR_PX
        ]
        if not valid_components:
            return target_rgb, target_mask

        # Track regions already occupied by anthills so duplicates do not stack.
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

            # Force a non-zero rotation so each duplicate looks different
            # from the original at the chosen position.
            k = random.randint(1, 3)  # 90, 180, 270 (skip 0)
            rgb_crop = np.ascontiguousarray(np.rot90(rgb_crop, k))
            mask_crop = np.ascontiguousarray(np.rot90(mask_crop, k))
            if random.random() < 0.5:
                rgb_crop = np.ascontiguousarray(np.fliplr(rgb_crop))
                mask_crop = np.ascontiguousarray(np.fliplr(mask_crop))

            crop_h, crop_w = rgb_crop.shape[:2]
            if crop_h >= th or crop_w >= tw:
                continue

            # Try a few placements; reject any that overlaps with occupied area.
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

            # Gaussian-soft alpha for RGB blending (same logic as copy-paste P1).
            alpha = ndimage.gaussian_filter(
                mask_crop.astype(np.float32),
                sigma=SegmentationDataset._PASTE_ALPHA_SIGMA,
            )
            alpha_max = float(alpha.max())
            if alpha_max > 0:
                alpha = np.clip(alpha / alpha_max, 0.0, 1.0)
            alpha_3c = alpha[..., None]

            target_patch = target_arr[y:y + crop_h, x:x + crop_w].astype(np.float32)
            donor_patch = rgb_crop.astype(np.float32)
            blended = alpha_3c * donor_patch + (1.0 - alpha_3c) * target_patch
            target_arr[y:y + crop_h, x:x + crop_w] = np.clip(blended, 0, 255).astype(np.uint8)

            paste_region = mask_crop.astype(bool)
            target_mask_arr[y:y + crop_h, x:x + crop_w, 0] = np.where(
                paste_region, 255, target_mask_arr[y:y + crop_h, x:x + crop_w, 0],
            )
            target_mask_arr[y:y + crop_h, x:x + crop_w, 1] = np.where(
                paste_region, 0, target_mask_arr[y:y + crop_h, x:x + crop_w, 1],
            )
            target_mask_arr[y:y + crop_h, x:x + crop_w, 2] = np.where(
                paste_region, 0, target_mask_arr[y:y + crop_h, x:x + crop_w, 2],
            )

            occupied[y:y + crop_h, x:x + crop_w] |= paste_region

        return Image.fromarray(target_arr), Image.fromarray(target_mask_arr)

    #  Dataset interface

    def __len__(self) -> int:
        return len(self._pairs)

    def has_anthill(self, index: int) -> bool:
        """Return True if the label mask at *index* contains any anthill pixels.

        Used by :func:`~app.service.data_service.build_oversampling_sampler`
        to assign higher sampling weights to positive examples.
        """
        _, label_path = self._pairs[index]
        mask_arr = np.array(Image.open(label_path).convert("RGB"))
        r, g, b = mask_arr[:, :, 0], mask_arr[:, :, 1], mask_arr[:, :, 2]
        return bool(np.any((r > 150) & (g < 100) & (b < 100)))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rgb_path, label_path = self._pairs[index]

        image, mask = self._load_pair(index)

        # Copy-paste augmentation: if this tile has no anthill pixels,
        # paste a random anthill from a positive tile with some probability.
        if self._copy_paste and random.random() < self._copy_paste_prob:
            if not self.has_anthill(index):
                image, mask = self._apply_copy_paste(image, mask)

        # Anthill self-duplication: if this tile already has anthills, paste
        # rotated copies of them onto empty regions of the same tile.  Keeps
        # the natural context (lighting, soil, vegetation) intact.
        if self._anthill_duplicate and random.random() < self._anthill_duplicate_prob:
            if self.has_anthill(index):
                image, mask = self._apply_anthill_duplicate(
                    image, mask, self._anthill_duplicate_max_copies,
                )

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

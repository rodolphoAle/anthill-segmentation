"""Visualize anthill augmentations BEFORE training.

Two modes are supported via ``--mode``:

* ``anthill-duplicate`` (default) — picks tiles that ALREADY contain anthills
  and shows the result of pasting rotated copies onto empty regions of the
  same tile (intra-tile duplication).
* ``copy-paste`` — picks tiles WITHOUT anthills and shows the result of
  pasting an anthill region from a different positive tile (cross-tile mix).

Each generated image is a side-by-side panel:
  - LEFT:   original tile (negative for copy-paste, positive for duplicate)
  - CENTER: tile after augmentation
  - RIGHT:  overlay highlighting where anthills (real + pasted) ended up

Usage::

    # Preview the active training augmentation (anthill-duplicate)
    python scripts/visualize_copy_paste.py

    # Force a specific mode
    python scripts/visualize_copy_paste.py --mode copy-paste
    python scripts/visualize_copy_paste.py --mode anthill-duplicate --max-copies 3

    # Common knobs
    python scripts/visualize_copy_paste.py --num-samples 20
    python scripts/visualize_copy_paste.py --output-dir output/aug_preview
    python scripts/visualize_copy_paste.py --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.infrastructure.augmentations import apply_anthill_duplicate, apply_copy_paste
from app.infrastructure.segmentation_dataset import SegmentationDataset
from app.core.config import settings


def _diff_mask(before: Image.Image, after: Image.Image) -> np.ndarray:
    """Return a boolean mask of pixels that are anthill in ``after`` but were
    not anthill in ``before`` — i.e. the regions newly added by augmentation.
    """
    before_arr = np.array(before)
    after_arr = np.array(after)

    def _is_anthill(arr: np.ndarray) -> np.ndarray:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        return (r > 150) & (g < 100) & (b < 100)

    return _is_anthill(after_arr) & ~_is_anthill(before_arr)


def _build_overlay(rgb_after: Image.Image, pasted_mask: np.ndarray) -> np.ndarray:
    """Highlight only the newly-pasted region in red on top of ``rgb_after``."""
    overlay = np.array(rgb_after).copy()
    if pasted_mask.any():
        overlay[pasted_mask, 0] = np.clip(
            overlay[pasted_mask, 0].astype(int) + 100, 0, 255
        ).astype(np.uint8)
        overlay[pasted_mask, 1] = (overlay[pasted_mask, 1] * 0.5).astype(np.uint8)
        overlay[pasted_mask, 2] = (overlay[pasted_mask, 2] * 0.5).astype(np.uint8)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview anthill augmentations on training tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="anthill-duplicate",
        choices=["copy-paste", "anthill-duplicate"],
        help="Which augmentation to preview.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of preview examples to generate.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/aug_preview",
        help="Directory to save the preview images.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--max-copies", type=int, default=2,
        help="(anthill-duplicate only) Max rotated copies pasted per tile.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = Path(settings.local_data_dir)
    train_rgb = base / settings.train_rgb_subdir
    train_labels = base / settings.train_labels_subdir

    # Build dataset with both flags enabled so the positive index is built once
    # and both augmentation methods are available.  Probabilities are 0.0 here
    # because we call the augmentation methods directly in this preview script.
    dataset = SegmentationDataset(
        rgb_dir=train_rgb,
        labels_dir=train_labels,
        augmentations=None,
        copy_paste=True,
        copy_paste_prob=0.0,
        anthill_duplicate=True,
        anthill_duplicate_prob=0.0,
        anthill_duplicate_max_copies=args.max_copies,
    )

    print(f"Dataset:        {len(dataset)} pairs")
    print(f"Positive tiles: {len(dataset._positive_indices)}")
    print(f"Negative tiles: {len(dataset) - len(dataset._positive_indices)}")
    print(f"Mode:           {args.mode}")

    # Tile pool, augmentation function, and labels depend on the mode.
    if args.mode == "copy-paste":
        all_indices = set(range(len(dataset)))
        positive_set = set(dataset._positive_indices)
        candidate_indices = sorted(all_indices - positive_set)
        tile_kind = "negative"
        title_after = "After Copy-Paste"
        title_overlay = "Overlay (red = pasted anthill)"
        filename_prefix = "copy_paste"

        def apply_aug(rgb: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
            donor_idx = random.choice(dataset._positive_indices)
            donor_rgb, donor_mask = dataset._load_pair(donor_idx)
            return apply_copy_paste(rgb, mask, donor_rgb, donor_mask)
    else:  # anthill-duplicate
        candidate_indices = sorted(dataset._positive_indices)
        tile_kind = "positive"
        title_after = f"After Anthill Duplicate (max {args.max_copies} cópias)"
        title_overlay = "Overlay (red = nova(s) cópia(s) rotacionada(s))"
        filename_prefix = "anthill_duplicate"

        def apply_aug(rgb: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
            return apply_anthill_duplicate(rgb, mask, args.max_copies)

    if not candidate_indices:
        print(f"ERROR: No {tile_kind} tiles found in the dataset.")
        return

    if args.mode == "copy-paste" and not dataset._positive_indices:
        print("ERROR: No positive tiles available as donors.")
        return

    num_samples = min(args.num_samples, len(candidate_indices))
    chosen = random.sample(candidate_indices, num_samples)

    print(f"\nGenerating {num_samples} '{args.mode}' previews...")

    for i, idx in enumerate(chosen):
        rgb_path, label_path = dataset._pairs[idx]

        original_rgb = Image.open(rgb_path).convert("RGB")
        original_mask = Image.open(label_path).convert("RGB")

        aug_rgb, aug_mask = apply_aug(original_rgb.copy(), original_mask.copy())

        # Highlight only what the augmentation ADDED relative to the original
        # mask (so existing anthills in positive tiles do not get re-painted).
        added_mask = _diff_mask(original_mask, aug_mask)
        overlay = _build_overlay(aug_rgb, added_mask)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(np.array(original_rgb))
        axes[0].set_title(f"Original ({tile_kind})\n{rgb_path.name}", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(np.array(aug_rgb))
        axes[1].set_title(title_after, fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(title_overlay, fontsize=10)
        axes[2].axis("off")

        plt.tight_layout()
        save_path = output_dir / f"{filename_prefix}_preview_{i+1:03d}.png"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i+1}/{num_samples}] Saved: {save_path}")

    # Combined grid for quick scanning
    print(f"\nGenerating combined grid...")
    cols = min(num_samples, 5)
    rows = min((num_samples + cols - 1) // cols, 4)
    shown = rows * cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for grid_idx in range(shown):
        if grid_idx >= num_samples:
            break
        idx = chosen[grid_idx]
        rgb_path, label_path = dataset._pairs[idx]

        original_rgb = Image.open(rgb_path).convert("RGB")
        original_mask = Image.open(label_path).convert("RGB")
        aug_rgb, _ = apply_aug(original_rgb.copy(), original_mask.copy())

        r_idx = grid_idx // cols
        c_idx = grid_idx % cols

        axes[r_idx, c_idx * 2].imshow(np.array(original_rgb))
        axes[r_idx, c_idx * 2].set_title("Original", fontsize=8)
        axes[r_idx, c_idx * 2].axis("off")

        axes[r_idx, c_idx * 2 + 1].imshow(np.array(aug_rgb))
        axes[r_idx, c_idx * 2 + 1].set_title(title_after.split(" (")[0], fontsize=8)
        axes[r_idx, c_idx * 2 + 1].axis("off")

    for grid_idx in range(shown, rows * cols):
        if grid_idx >= num_samples:
            r_idx = grid_idx // cols
            c_idx = grid_idx % cols
            axes[r_idx, c_idx * 2].axis("off")
            axes[r_idx, c_idx * 2 + 1].axis("off")

    plt.suptitle(
        f"{args.mode} preview (Left=Original, Right=Augmented)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    grid_path = output_dir / f"{filename_prefix}_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid saved: {grid_path}")

    print(f"\nAll previews saved to: {output_dir}")


if __name__ == "__main__":
    main()

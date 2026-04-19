"""Visualize copy-paste augmentation results BEFORE training.

Generates a grid of side-by-side images showing:
  - LEFT:  original negative tile (no anthill)
  - CENTER: the pasted anthill mask
  - RIGHT: the result after copy-paste augmentation

Usage::

    python scripts/visualize_copy_paste.py
    python scripts/visualize_copy_paste.py --num-samples 20
    python scripts/visualize_copy_paste.py --output-dir output/copy_paste_preview
    python scripts/visualize_copy_paste.py --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.infrastructure.segmentation_dataset import SegmentationDataset
from app.core.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview copy-paste augmentation on negative tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of copy-paste examples to generate.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/copy_paste_preview",
        help="Directory to save the preview images.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--copy-paste-prob", type=float, default=1.0,
        help="Probability of copy-paste (1.0 = always, for visualization).",
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

    # Build dataset with copy-paste enabled (prob=1.0 for visualization)
    dataset = SegmentationDataset(
        rgb_dir=train_rgb,
        labels_dir=train_labels,
        augmentations=None,
        copy_paste=True,
        copy_paste_prob=args.copy_paste_prob,
    )

    print(f"Dataset: {len(dataset)} pairs")
    print(f"Positive tiles: {len(dataset._positive_indices)}")
    print(f"Negative tiles: {len(dataset) - len(dataset._positive_indices)}")

    # Find negative indices
    all_indices = set(range(len(dataset)))
    positive_set = set(dataset._positive_indices)
    negative_indices = sorted(all_indices - positive_set)

    if not negative_indices:
        print("ERROR: No negative tiles found in the dataset.")
        return

    if not dataset._positive_indices:
        print("ERROR: No positive tiles found in the dataset.")
        return

    num_samples = min(args.num_samples, len(negative_indices))
    chosen_negatives = random.sample(negative_indices, num_samples)

    print(f"\nGenerating {num_samples} copy-paste previews...")

    for i, neg_idx in enumerate(chosen_negatives):
        rgb_path, label_path = dataset._pairs[neg_idx]

        # Load original (before copy-paste)
        original_rgb = Image.open(rgb_path).convert("RGB")
        original_mask = Image.open(label_path).convert("RGB")

        # Apply copy-paste (force it — prob already set to 1.0)
        aug_rgb, aug_mask = dataset._apply_copy_paste(
            original_rgb.copy(), original_mask.copy()
        )

        # Parse the augmented mask to show anthill region clearly
        aug_mask_arr = np.array(aug_mask)
        r, g, b = aug_mask_arr[:, :, 0], aug_mask_arr[:, :, 1], aug_mask_arr[:, :, 2]
        anthill_highlight = ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8)

        # Create overlay: augmented image with semi-transparent red on anthill
        overlay = np.array(aug_rgb).copy()
        overlay[anthill_highlight == 1, 0] = np.clip(
            overlay[anthill_highlight == 1, 0].astype(int) + 100, 0, 255
        ).astype(np.uint8)
        overlay[anthill_highlight == 1, 1] = (
            overlay[anthill_highlight == 1, 1] * 0.5
        ).astype(np.uint8)
        overlay[anthill_highlight == 1, 2] = (
            overlay[anthill_highlight == 1, 2] * 0.5
        ).astype(np.uint8)

        # Plot: original | result | overlay with mask
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(np.array(original_rgb))
        axes[0].set_title(f"Original (negative)\n{rgb_path.name}", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(np.array(aug_rgb))
        axes[1].set_title("After Copy-Paste", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (red = pasted anthill)", fontsize=10)
        axes[2].axis("off")

        plt.tight_layout()
        save_path = output_dir / f"copy_paste_preview_{i+1:03d}.png"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i+1}/{num_samples}] Saved: {save_path}")

    # Also generate a combined grid for quick overview
    print(f"\nGenerating combined grid...")
    cols = min(num_samples, 5)
    rows = min((num_samples + cols - 1) // cols, 4)
    shown = rows * cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for idx in range(shown):
        if idx >= num_samples:
            break
        neg_idx = chosen_negatives[idx]
        rgb_path, label_path = dataset._pairs[neg_idx]

        original_rgb = Image.open(rgb_path).convert("RGB")
        original_mask = Image.open(label_path).convert("RGB")
        aug_rgb, aug_mask = dataset._apply_copy_paste(
            original_rgb.copy(), original_mask.copy()
        )

        r_idx = idx // cols
        c_idx = idx % cols

        axes[r_idx, c_idx * 2].imshow(np.array(original_rgb))
        axes[r_idx, c_idx * 2].set_title("Original", fontsize=8)
        axes[r_idx, c_idx * 2].axis("off")

        axes[r_idx, c_idx * 2 + 1].imshow(np.array(aug_rgb))
        axes[r_idx, c_idx * 2 + 1].set_title("Copy-Paste", fontsize=8)
        axes[r_idx, c_idx * 2 + 1].axis("off")

    # Hide empty subplots
    for idx in range(shown, rows * cols):
        r_idx = idx // cols
        c_idx = idx % cols
        axes[r_idx, c_idx * 2].axis("off")
        axes[r_idx, c_idx * 2 + 1].axis("off")

    plt.suptitle(
        "Copy-Paste Augmentation Preview (Left=Original, Right=Augmented)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    grid_path = output_dir / "copy_paste_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid saved: {grid_path}")

    print(f"\nAll previews saved to: {output_dir}")


if __name__ == "__main__":
    main()

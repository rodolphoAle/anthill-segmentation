"""Matplotlib-based visualization helpers for image segmentation.

Provides convenience functions for side-by-side comparison of RGB images,
ground-truth masks, and model predictions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from PIL import Image


def show_image_and_mask(
    image: np.ndarray | Image.Image,
    mask: np.ndarray | Image.Image,
    pred_mask: np.ndarray | Image.Image | None = None,
    *,
    figsize: tuple[int, int] = (12, 4),
    save_path: str | None = None,
) -> None:
    """Display image, ground-truth mask, and optional prediction side-by-side.

    Args:
        image: RGB input image.
        mask: Ground-truth segmentation mask.
        pred_mask: Optional predicted mask.
        figsize: Matplotlib figure size.
        save_path: If provided, save the figure instead of showing it.
    """
    n_cols = 3 if pred_mask is not None else 2

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    axes[0].set_title("Image")
    axes[0].imshow(image)
    axes[0].axis("off")

    axes[1].set_title("Ground Truth")
    axes[1].imshow(mask, cmap="gray")
    axes[1].axis("off")

    if pred_mask is not None:
        axes[2].set_title("Prediction")
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def show_augmentation_preview(
    original_rgb: np.ndarray,
    augmented_rgb: np.ndarray,
    overlay: np.ndarray,
    *,
    title_before: str = "Original",
    title_after: str = "After Augmentation",
    title_overlay: str = "Overlay (red = new regions)",
    figsize: tuple[int, int] = (18, 6),
    save_path: str | None = None,
) -> None:
    """Display before/after/overlay panels for augmentation preview.

    Args:
        original_rgb: Original tile RGB array.
        augmented_rgb: Tile after augmentation.
        overlay: RGB array highlighting newly pasted regions.
        title_before: Title for the left panel.
        title_after: Title for the centre panel.
        title_overlay: Title for the right panel.
        figsize: Matplotlib figure size.
        save_path: If provided, save the figure instead of showing it.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].set_title(title_before)
    axes[0].imshow(original_rgb)
    axes[0].axis("off")

    axes[1].set_title(title_after)
    axes[1].imshow(augmented_rgb)
    axes[1].axis("off")

    axes[2].set_title(title_overlay)
    axes[2].imshow(overlay)
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

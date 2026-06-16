"""Quick dataset sanity-check — validates shapes, normalisation, class
distribution and gradient flow on a small batch.

Can be invoked directly::

    python scripts/validate_dataset.py --data-dir ./data/training

Or via the unified CLI::

    python -m app.main validate-dataset --data-dir ./data/training
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.infrastructure.segmentation_dataset import SegmentationDataset  # noqa: E402


#  Validation logic 


def _validate(rgb_dir: str, labels_dir: str, batch_size: int) -> bool:
    """Run all checks and return True if every check passes."""

    print("\n" + "=" * 70)
    print("  DATASET VALIDATION")
    print("=" * 70)

    #  Create dataset 
    print(f"\nCreating dataset  rgb={rgb_dir}  labels={labels_dir}")
    dataset = SegmentationDataset(rgb_dir=rgb_dir, labels_dir=labels_dir)
    print(f"Dataset contains {len(dataset)} image/mask pairs")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Loading first batch (batch_size={batch_size})…")
    batch = next(iter(loader))
    images, masks, names = batch
    print(f"Files: {names}")

    ok = True

    #  1. Image shape 
    print("\n" + "-" * 70)
    print("  CHECK 1: Image shape")
    print("-" * 70)

    h, w = images.shape[2], images.shape[3]
    print(f"  Shape: {list(images.shape)}  (batch, channels, H, W)")

    if images.shape[1] == 3:
        print("  PASS — 3 channels")
    else:
        print(f"  FAIL — expected 3 channels, got {images.shape[1]}")
        ok = False

    #  2. ImageNet normalisation 
    print("\n" + "-" * 70)
    print("  CHECK 2: ImageNet normalisation")
    print("-" * 70)

    img_min = images.min().item()
    img_max = images.max().item()
    img_mean = images.mean().item()
    img_std = images.std().item()

    print(f"  Min:  {img_min:+.3f}")
    print(f"  Max:  {img_max:+.3f}")
    print(f"  Mean: {img_mean:+.3f}")
    print(f"  Std:  {img_std:.3f}")

    if -3.5 < img_min and img_max < 3.5:
        print("  PASS — values in expected normalised range")
    else:
        print("  FAIL — values outside [-3.5, 3.5]")
        ok = False

    #  3. Mask shape 
    print("\n" + "-" * 70)
    print("  CHECK 3: Mask shape")
    print("-" * 70)

    expected_mask = torch.Size([batch_size, h, w])
    print(f"  Expected: {list(expected_mask)}")
    print(f"  Got:      {list(masks.shape)}")

    if masks.shape == expected_mask:
        print("  PASS")
    else:
        print("  FAIL — shape mismatch")
        ok = False

    #  4. Class distribution 
    print("\n" + "-" * 70)
    print("  CHECK 4: Class distribution")
    print("-" * 70)

    total_bg = total_ant = total_ign = 0

    for i in range(batch_size):
        mask = masks[i]
        bg = int((mask == 0).sum())
        ant = int((mask == 1).sum())
        ign = int((mask == 255).sum())
        total_pixels = mask.numel()

        total_bg += bg
        total_ant += ant
        total_ign += ign

        print(f"\n  Image {i} ({names[i]}):")
        print(f"    Background (0):  {bg:>8d}  ({100 * bg / total_pixels:5.1f}%)")
        print(f"    Anthill    (1):  {ant:>8d}  ({100 * ant / total_pixels:5.1f}%)")
        print(f"    Ignore   (255):  {ign:>8d}  ({100 * ign / total_pixels:5.1f}%)")

    total_all = total_bg + total_ant + total_ign
    print(f"\n  Totals across batch:")
    print(f"    Background:  {total_bg:>8d}  ({100 * total_bg / total_all:5.1f}%)")
    print(f"    Anthill:     {total_ant:>8d}  ({100 * total_ant / total_all:5.1f}%)")
    print(f"    Ignore:      {total_ign:>8d}  ({100 * total_ign / total_all:5.1f}%)")

    if total_ant > 0:
        print(f"  PASS — {100 * total_ant / total_all:.2f}% anthill pixels found")
    else:
        print("  WARNING — 0 anthill pixels in this batch (may be normal for some tiles)")

    #  5. Gradient flow 
    print("\n" + "-" * 70)
    print("  CHECK 5: Gradient flow (backward pass)")
    print("-" * 70)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")
        masks_f = masks.float().to(device)
        masks_f.requires_grad_(True)
        loss = masks_f.mean()
        loss.backward()
        print("  PASS — backward() succeeded")
    except Exception as exc:
        print(f"  FAIL — {exc}")
        ok = False

    #  Summary 
    print("\n" + "=" * 70)
    if ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — review output above")
    print("=" * 70)

    return ok


#  Parser / CLI 


def build_parser(
    subparsers: argparse._SubParsersAction | None = None,
) -> argparse.ArgumentParser:
    """Build the validate-dataset argument parser."""
    kwargs: dict = dict(
        description="Quick sanity-check on the local segmentation dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if subparsers is not None:
        parser = subparsers.add_parser("validate-dataset", **kwargs)
    else:
        parser = argparse.ArgumentParser(**kwargs)

    default_rgb = str(
        Path(settings.local_data_dir) / settings.train_rgb_subdir
    )
    default_labels = str(
        Path(settings.local_data_dir) / settings.train_labels_subdir
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help="Root data directory containing rgb/ and labels/ sub-folders. "
             "If provided, --rgb-dir and --labels-dir are derived automatically.",
    )
    parser.add_argument(
        "--rgb-dir",
        default=default_rgb,
        metavar="DIR",
        help=f"RGB images directory (default: {default_rgb})",
    )
    parser.add_argument(
        "--labels-dir",
        default=default_labels,
        metavar="DIR",
        help=f"Label masks directory (default: {default_labels})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        metavar="N",
        help="Number of images to load for validation",
    )

    return parser


def run(args: argparse.Namespace) -> None:
    """Execute dataset validation."""
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        rgb_dir = str(data_dir / "rgb")
        labels_dir = str(data_dir / "labels")
    else:
        rgb_dir = args.rgb_dir
        labels_dir = args.labels_dir

    success = _validate(rgb_dir, labels_dir, args.batch_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    parser = build_parser()
    _args = parser.parse_args()
    run(_args)

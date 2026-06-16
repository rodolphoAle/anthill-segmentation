"""Training CLI — parse args, apply env overrides, run training pipeline.

Can be invoked directly::

    python scripts/run_training.py --epochs 50 --lr 0.0005

Or via the unified CLI::

    python -m app.main train --epochs 50 --lr 0.0005
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser(subparsers: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Build the training argument parser.

    If *subparsers* is provided, the parser is registered as a sub-command
    named ``train``.  Otherwise a standalone parser is returned.
    """
    kwargs: dict = dict(
        description="Train the UNet segmentation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if subparsers is not None:
        parser = subparsers.add_parser("train", **kwargs)
    else:
        parser = argparse.ArgumentParser(**kwargs)

    #  Basic training 
    parser.add_argument("--epochs", type=int, metavar="N",
                        help="Number of training epochs (UNET_NUM_EPOCHS)")
    parser.add_argument("--lr", type=float, metavar="LR",
                        help="Learning rate (UNET_LEARNING_RATE)")
    parser.add_argument("--batch-size", type=int, metavar="N",
                        help="Batch size (UNET_BATCH_SIZE)")
    parser.add_argument("--workers", type=int, metavar="N",
                        help="DataLoader worker processes (UNET_NUM_WORKERS)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        help="Compute device (UNET_DEVICE)")
    parser.add_argument("--data-mode", choices=["local", "online"],
                        help="Dataset source (UNET_DATA_MODE)")

    #  Loss & optimisation 
    parser.add_argument("--class-weight-anthill", type=float, metavar="W",
                        help="Loss weight for anthill class (UNET_CLASS_WEIGHT_ANTHILL)")
    parser.add_argument("--class-weight-bg", type=float, metavar="W",
                        help="Loss weight for background class (UNET_CLASS_WEIGHT_BACKGROUND)")
    parser.add_argument("--grad-clip", type=float, metavar="N",
                        help="Max gradient L2 norm (UNET_GRAD_CLIP_MAX_NORM)")
    parser.add_argument("--scheduler-factor", type=float, metavar="F",
                        help="LR reduction factor on plateau (UNET_SCHEDULER_FACTOR)")
    parser.add_argument("--scheduler-patience", type=int, metavar="N",
                        help="Epochs before reducing LR (UNET_SCHEDULER_PATIENCE)")

    #  Augmentations 
    parser.add_argument("--aug-no-hflip", action="store_true",
                        help="Disable horizontal flip (UNET_AUG_HORIZONTAL_FLIP=false)")
    parser.add_argument("--aug-vertical-flip", action="store_true",
                        help="Enable vertical flip (UNET_AUG_VERTICAL_FLIP=true)")
    parser.add_argument("--aug-rotation", type=int, metavar="DEG",
                        help="Max rotation angle in degrees; 0=off (UNET_AUG_ROTATION_DEGREES)")
    parser.add_argument("--aug-color-jitter", action="store_true",
                        help="Enable ColorJitter augmentation (UNET_AUG_COLOR_JITTER=true)")

    return parser


def apply_overrides(args: argparse.Namespace) -> None:
    """Push parsed CLI values into ``os.environ`` for Pydantic to pick up."""
    scalar_mapping = {
        "epochs":              "UNET_NUM_EPOCHS",
        "lr":                  "UNET_LEARNING_RATE",
        "batch_size":          "UNET_BATCH_SIZE",
        "workers":             "UNET_NUM_WORKERS",
        "device":              "UNET_DEVICE",
        "data_mode":           "UNET_DATA_MODE",
        "class_weight_anthill": "UNET_CLASS_WEIGHT_ANTHILL",
        "class_weight_bg":     "UNET_CLASS_WEIGHT_BACKGROUND",
        "grad_clip":           "UNET_GRAD_CLIP_MAX_NORM",
        "scheduler_factor":    "UNET_SCHEDULER_FACTOR",
        "scheduler_patience":  "UNET_SCHEDULER_PATIENCE",
        "aug_rotation":        "UNET_AUG_ROTATION_DEGREES",
    }
    for attr, env_key in scalar_mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env_key] = str(value)

    if getattr(args, "aug_no_hflip", False):
        os.environ["UNET_AUG_HORIZONTAL_FLIP"] = "false"
    if getattr(args, "aug_vertical_flip", False):
        os.environ["UNET_AUG_VERTICAL_FLIP"] = "true"
    if getattr(args, "aug_color_jitter", False):
        os.environ["UNET_AUG_COLOR_JITTER"] = "true"

    os.environ.setdefault("UNET_PIPELINE_MODE", "train")


def run(args: argparse.Namespace) -> None:
    """Execute training with the given arguments."""
    apply_overrides(args)

    from app.main import pipeline_main  # noqa: E402

    asyncio.run(pipeline_main())


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)

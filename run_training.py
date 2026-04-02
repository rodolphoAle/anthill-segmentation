"""Entry point for model training.

Usage (inside the container)::

    python run_training.py
    python run_training.py --epochs 50 --lr 0.0005
    python run_training.py --batch-size 4 --class-weight-anthill 8.0
    python run_training.py --aug-vertical-flip --aug-color-jitter
    python run_training.py --help   # full list of flags

All flags are optional and override the value from .env / config.py defaults
for THIS RUN ONLY — no container restart required.

Parameter reference (see app/core/config.py for full documentation):
  --epochs N               Total training epochs
  --lr LR                  Adam learning rate
  --batch-size N           Images per gradient update
  --workers N              DataLoader subprocesses
  --device auto|cpu|cuda   Compute device
  --data-mode local|online Dataset source
  --class-weight-anthill W CrossEntropyLoss weight for anthill class
  --class-weight-bg W      CrossEntropyLoss weight for background class
  --grad-clip N            Max gradient L2 norm
  --scheduler-factor F     LR reduction multiplier on plateau
  --scheduler-patience N   Epochs before reducing LR
  --aug-no-hflip           Disable horizontal flip augmentation
  --aug-vertical-flip      Enable vertical flip augmentation
  --aug-rotation N         Max rotation angle in degrees (0 = off)
  --aug-color-jitter       Enable ColorJitter (brightness/contrast/saturation)
"""

import argparse
import asyncio
import os


def _apply_cli_overrides() -> None:
    """Parse CLI flags and push provided values into os.environ.

    Must run *before* importing app.main so that Pydantic BaseSettings
    picks them up at construction time.
    """
    parser = argparse.ArgumentParser(
        description="Train the UNet segmentation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Basic training ────────────────────────────────────────────────────────
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

    # ── Loss & optimisation ───────────────────────────────────────────────────
    parser.add_argument("--class-weight-anthill", type=float, metavar="W",
                        help="CrossEntropyLoss weight for anthill class "
                             "(UNET_CLASS_WEIGHT_ANTHILL)")
    parser.add_argument("--class-weight-bg", type=float, metavar="W",
                        help="CrossEntropyLoss weight for background class "
                             "(UNET_CLASS_WEIGHT_BACKGROUND)")
    parser.add_argument("--grad-clip", type=float, metavar="N",
                        help="Max gradient L2 norm (UNET_GRAD_CLIP_MAX_NORM)")
    parser.add_argument("--scheduler-factor", type=float, metavar="F",
                        help="LR reduction factor on plateau "
                             "(UNET_SCHEDULER_FACTOR)")
    parser.add_argument("--scheduler-patience", type=int, metavar="N",
                        help="Epochs before reducing LR "
                             "(UNET_SCHEDULER_PATIENCE)")

    # ── Augmentations ─────────────────────────────────────────────────────────
    parser.add_argument("--aug-no-hflip", action="store_true",
                        help="Disable horizontal flip (UNET_AUG_HORIZONTAL_FLIP=false)")
    parser.add_argument("--aug-vertical-flip", action="store_true",
                        help="Enable vertical flip (UNET_AUG_VERTICAL_FLIP=true)")
    parser.add_argument("--aug-rotation", type=int, metavar="DEG",
                        help="Max rotation angle in degrees; 0=off "
                             "(UNET_AUG_ROTATION_DEGREES)")
    parser.add_argument("--aug-color-jitter", action="store_true",
                        help="Enable ColorJitter augmentation "
                             "(UNET_AUG_COLOR_JITTER=true)")

    args = parser.parse_args()

    # Scalar overrides
    scalar_mapping = {
        "epochs":              "UNET_NUM_EPOCHS",
        "lr":                  "UNET_LEARNING_RATE",
        "batch_size":          "UNET_BATCH_SIZE",
        "workers":             "UNET_NUM_WORKERS",
        "device":              "UNET_DEVICE",
        "data_mode":           "UNET_DATA_MODE",
        "class_weight_anthill":"UNET_CLASS_WEIGHT_ANTHILL",
        "class_weight_bg":     "UNET_CLASS_WEIGHT_BACKGROUND",
        "grad_clip":           "UNET_GRAD_CLIP_MAX_NORM",
        "scheduler_factor":    "UNET_SCHEDULER_FACTOR",
        "scheduler_patience":  "UNET_SCHEDULER_PATIENCE",
        "aug_rotation":        "UNET_AUG_ROTATION_DEGREES",
    }
    for attr, env_key in scalar_mapping.items():
        value = getattr(args, attr)
        if value is not None:
            os.environ[env_key] = str(value)

    # Boolean flag overrides
    if args.aug_no_hflip:
        os.environ["UNET_AUG_HORIZONTAL_FLIP"] = "false"
    if args.aug_vertical_flip:
        os.environ["UNET_AUG_VERTICAL_FLIP"] = "true"
    if args.aug_color_jitter:
        os.environ["UNET_AUG_COLOR_JITTER"] = "true"


_apply_cli_overrides()
os.environ.setdefault("UNET_PIPELINE_MODE", "train")

from app.main import main  # noqa: E402

asyncio.run(main())

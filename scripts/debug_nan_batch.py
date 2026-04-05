"""Debug NaN/Inf in training: load a single batch and print tensor stats.

Usage:
  # debug by sample filename (one of the names printed in logs)
  python scripts/debug_nan_batch.py --name "Gleba 01_25172_35656.jpg"

  # debug by batch index (0-based)
  python scripts/debug_nan_batch.py --batch 10

The script uses the same DataService and TrainingService model code,
loads the model weights from `u_net.pth` by default, and runs a single
forward+loss pass printing min/max/mean/std and NaN/Inf checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image

# When this script is executed as a top-level file (python scripts/debug_nan_batch.py)
# Python puts the script's directory on sys.path which prevents imports like
# `from app.core.config import settings` from finding the project package. Ensure
# the repository root is on sys.path so `app` can be imported whether the script
# is run directly or as a module.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.service.data_service import DataService, create_train_transforms
from app.infrastructure.segmentation_dataset import SegmentationDataset
from app.domain.protocols import StorageClientProtocol


def print_stats(name: str, t: torch.Tensor) -> None:
    try:
        vals = t.detach().cpu()
        print(f"-- {name}: dtype={vals.dtype} shape={tuple(vals.shape)}")
        if vals.numel() == 0:
            return
        print("   min=", float(torch.min(vals)))
        print("   max=", float(torch.max(vals)))
        print("   mean=", float(torch.mean(vals.float())))
        print("   std=", float(torch.std(vals.float())))
        print("   anyNaN=", bool(torch.isnan(vals).any()))
        print("   anyInf=", bool(torch.isinf(vals).any()))
    except Exception as exc:
        print("   (failed to print stats)", exc)


def load_model(path: str | None, device: torch.device):
    # import here to avoid top-level import issues when running as script
    from app.domain.unet import UNet

    model = UNet(n_channels=settings.n_channels, n_classes=settings.n_classes)
    if path and Path(path).exists():
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {path}")
    else:
        print("No weights file found, using random init")
    model.to(device)
    model.eval()
    return model


def find_pair_by_name(dataset: SegmentationDataset, name: str):
    for idx in range(len(dataset)):
        _, _, fname = dataset[idx]
        if fname == name:
            return idx
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, help="Batch index (0-based)")
    parser.add_argument("--name", type=str, help="Image filename to locate in dataset")
    parser.add_argument("--weights", type=str, default=str(Path(settings.model_save_path)), help="Model weights path")
    parser.add_argument("--split", choices=["train","val"], default="train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base = Path(settings.local_data_dir)
    if args.split == "train":
        rgb = base / settings.train_rgb_subdir
        labels = base / settings.train_labels_subdir
    else:
        rgb = base / settings.val_rgb_subdir
        labels = base / settings.val_labels_subdir

    ds = SegmentationDataset(rgb_dir=rgb, labels_dir=labels, augmentations=create_train_transforms(), preload=False)
    print(f"Dataset size: {len(ds)}")

    if args.name:
        idx = find_pair_by_name(ds, args.name)
        if idx is None:
            print("Name not found in dataset:" , args.name)
            sys.exit(1)
    elif args.batch is not None:
        idx = args.batch
        if idx < 0 or idx >= len(ds):
            print("Batch index out of range")
            sys.exit(1)
    else:
        print("Provide --name or --batch")
        sys.exit(1)

    model = load_model(args.weights, device)

    # get single sample and make a batch of size 1
    image, mask, fname = ds[idx]
    print("Sample:", fname)
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    print_stats("inputs", image)
    print_stats("labels", mask)

    with torch.no_grad():
        try:
            outputs = model(image)
            print_stats("outputs", outputs)
        except Exception as exc:
            print("Model forward failed:", exc)
            raise

    # compute loss
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0,5.0], device=device), ignore_index=255)
    try:
        loss = criterion(outputs, mask)
        print("loss:", float(loss))
        print("loss is nan:", torch.isnan(loss).any().item())
    except Exception as exc:
        print("Loss computation failed:", exc)


if __name__ == "__main__":
    main()

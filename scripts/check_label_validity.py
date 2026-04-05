"""Check how many label masks contain zero valid pixels according to
the dataset mapping (background/anthill).

Run:
  python scripts/check_label_validity.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def has_valid_pixels(path: Path) -> bool:
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        is_anthill = (r > 150) & (g < 100) & (b < 100)
        is_background = (r < 50) & (g < 50) & (b < 50)
        valid = is_anthill | is_background
        return bool(valid.any())
    except Exception:
        return False


def scan_dir(dirpath: Path, sample_max: int = 10) -> None:
    files = sorted(p for p in dirpath.iterdir() if p.is_file())
    total = len(files)
    no_valid = []
    for p in files:
        if not has_valid_pixels(p):
            no_valid.append(p.name)

    print(f"Scanned {total} files in {dirpath}")
    print(f"Masks with NO valid pixels: {len(no_valid)} ({len(no_valid)/total:.2%})")
    if no_valid:
        print("Sample filenames:")
        for name in no_valid[:sample_max]:
            print(" -", name)


if __name__ == "__main__":
    base = Path("data")
    train_labels = base / "training" / "labels" / "labels"
    val_labels = base / "validation" / "labels" / "labels"

    if train_labels.exists():
        scan_dir(train_labels)
    else:
        print("Training labels folder not found:", train_labels)

    if val_labels.exists():
        scan_dir(val_labels)
    else:
        print("Validation labels folder not found:", val_labels)

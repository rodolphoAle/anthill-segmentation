"""Evaluation CLI — compare predictions vs ground-truth labels.

Can be invoked directly::

    python scripts/run_evaluate.py --pred-dir output/validation_results

Or via the unified CLI::

    python -m app.main evaluate --pred-dir output/validation_results
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from loguru import logger  # type: ignore

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.domain.mask_utils import (  # noqa: E402
    LABEL_ANTHILL,
    LABEL_IGNORE,
    decode_rgb_mask,
)
from app.domain.metrics import EvaluationMetrics  # noqa: E402
from app.infrastructure.google_drive_client import GoogleDriveClient  # noqa: E402
from app.service.data_service import DataService  # noqa: E402

#  Constants 

RGB_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

#  I/O helpers 


def load_label(path: Path) -> np.ndarray:
    """Load a ground-truth label PNG using the shared decoder."""
    return decode_rgb_mask(Image.open(path))


def load_pred_mask(path: Optional[Path], *, shape: tuple[int, int]) -> np.ndarray:
    """Load a predicted mask PNG.

    Returns a binary 2-D uint8 array (0 = background, 1 = anthill).
    If *path* is None or does not exist an all-zero mask is returned.
    """
    if path is None or not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def find_rgb(stem: str, rgb_dir: Path) -> Optional[Path]:
    """Return the first matching RGB file for *stem* in *rgb_dir*."""
    for ext in RGB_EXTENSIONS:
        candidate = rgb_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def image_has_anthill(label: np.ndarray, min_px: int = 1) -> bool:
    """Return True when *label* has at least *min_px* anthill pixels."""
    return int((label == LABEL_ANTHILL).sum()) >= max(min_px, 1)


def save_case(
    stem: str,
    dest_dir: Path,
    gt_label: np.ndarray,
    pred_mask: np.ndarray,
    gt_rgb_path: Optional[Path],
    pred_rgb_path: Optional[Path],
) -> None:
    """Save diagnostics for a single FP or FN image into *dest_dir*."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    label_vis = np.zeros_like(gt_label)
    label_vis[gt_label == LABEL_ANTHILL] = 255
    label_vis[gt_label == LABEL_IGNORE] = 128
    Image.fromarray(label_vis, mode="L").save(dest_dir / f"{stem}_gt_label.png")

    if gt_rgb_path and gt_rgb_path.exists():
        Image.open(gt_rgb_path).save(dest_dir / f"{stem}_gt_rgb.png")

    pred_vis = (pred_mask * 255).astype(np.uint8)
    Image.fromarray(pred_vis, mode="L").save(dest_dir / f"{stem}_pred_mask.png")

    if pred_rgb_path and pred_rgb_path.exists():
        Image.open(pred_rgb_path).save(dest_dir / f"{stem}_pred_rgb.png")
    elif gt_rgb_path and gt_rgb_path.exists():
        Image.open(gt_rgb_path).save(dest_dir / f"{stem}_pred_rgb.png")


#  Parser 


def build_parser(subparsers: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Build the evaluation argument parser.

    If *subparsers* is provided, the parser is registered as a sub-command
    named ``evaluate``.  Otherwise a standalone parser is returned.
    """
    _default_labels = str(Path(settings.local_data_dir) / settings.val_labels_subdir)
    _default_rgb = str(Path(settings.local_data_dir) / settings.val_rgb_subdir)

    kwargs: dict = dict(
        description="Evaluate anthill detection quality vs ground-truth labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if subparsers is not None:
        parser = subparsers.add_parser("evaluate", **kwargs)
    else:
        parser = argparse.ArgumentParser(**kwargs)

    parser.add_argument(
        "--pred-dir", required=(subparsers is None),
        help="Directory with *_mask.png and *_rgb.png from validation",
    )
    parser.add_argument(
        "--labels-dir", default=_default_labels,
        help=f"Ground-truth label directory (default: {_default_labels})",
    )
    parser.add_argument(
        "--rgb-dir", default=_default_rgb,
        help=f"Validation RGB directory (default: {_default_rgb})",
    )
    parser.add_argument(
        "--save-dir", default="evaluation_output",
        help="Output directory for FP/FN cases and metrics.txt",
    )
    parser.add_argument(
        "--min-px", type=int, default=1,
        help="Min GT anthill pixels to classify image as positive",
    )

    return parser


#  Execution 


async def _run_evaluate(args: argparse.Namespace) -> None:
    setup_logging()

    pred_dir = Path(args.pred_dir)
    save_dir = Path(args.save_dir)

    logger.info("Evaluation started — mode={}", settings.data_mode)

    if settings.data_mode == "online":
        logger.info("Online mode — downloading validation data from Google Drive…")
        drive_client = GoogleDriveClient()
        data_service = DataService(storage_client=drive_client)
        labels_dir, rgb_dir = await data_service.download_validation_from_drive()
    else:
        labels_dir = Path(args.labels_dir)
        rgb_dir = Path(args.rgb_dir)

    fn_dir = save_dir / "somente_original"
    fp_dir = save_dir / "somente_resultado"

    if not pred_dir.exists():
        logger.error("pred-dir '{}' does not exist.", pred_dir)
        sys.exit(1)
    if not labels_dir.exists():
        logger.error("labels-dir '{}' does not exist.", labels_dir)
        sys.exit(1)

    label_files = sorted(labels_dir.glob("*.png"))
    if not label_files:
        logger.error("No *.png files found in '{}'.", labels_dir)
        sys.exit(1)

    logger.info("Labels dir:  {}  ({} files)", labels_dir, len(label_files))
    logger.info("Pred dir:    {}", pred_dir)
    logger.info("RGB dir:     {}", rgb_dir)
    logger.info("Save dir:    {}", save_dir)
    logger.info("Min GT px:   {}", args.min_px)

    metrics = EvaluationMetrics()

    for idx, label_path in enumerate(label_files):
        stem = label_path.stem
        gt_label = load_label(label_path)
        h, w = gt_label.shape

        pred_mask_path = pred_dir / f"{stem}_mask.png"
        pred_rgb_path = pred_dir / f"{stem}_rgb.png"
        gt_rgb_path = find_rgb(stem, rgb_dir)

        pred_mask = load_pred_mask(pred_mask_path, shape=(h, w))
        gt_pos = image_has_anthill(gt_label, args.min_px)
        pred_pos = bool(pred_mask.any())

        metrics.update_image_level(gt_pos, pred_pos)
        metrics.update_pixel_level(gt_label, pred_mask)

        if gt_pos and not pred_pos:
            save_case(stem, fn_dir, gt_label, pred_mask, gt_rgb_path, pred_rgb_path)
        elif pred_pos and not gt_pos:
            save_case(stem, fp_dir, gt_label, pred_mask, gt_rgb_path, pred_rgb_path)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(label_files):
            logger.info(
                "  [{:>5}/{}]  TP={}  FP={}  FN={}  TN={}",
                idx + 1, len(label_files),
                metrics.tp, metrics.fp, metrics.fn, metrics.tn,
            )

    report = metrics.report()
    logger.info(report)

    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / "metrics.txt"
    report_path.write_text(report, encoding="utf-8")

    logger.info("Report saved  →  {}", report_path)
    logger.info("FN cases      →  {}  ({} images)", fn_dir, metrics.fn)
    logger.info("FP cases      →  {}  ({} images)", fp_dir, metrics.fp)


def run(args: argparse.Namespace) -> None:
    """Execute evaluation with the given arguments."""
    asyncio.run(_run_evaluate(args))


if __name__ == "__main__":
    parser = build_parser()
    _args = parser.parse_args()
    run(_args)

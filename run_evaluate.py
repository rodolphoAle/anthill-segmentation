"""
Anthill detection evaluation script.

Compares predicted masks produced by run_validation.py against ground-truth
labels stored locally.  Computes both image-level detection metrics
(TP / FP / FN / TN / Precision / Recall / F1) and pixel-level metrics
(Pixel Accuracy / per-class IoU / per-class Dice / mIoU / Mean Dice).

Output structure inside --save-dir:
    somente_original/   — FN cases: anthill in GT label, not in prediction
    somente_resultado/  — FP cases: anthill in prediction, not in GT label
    metrics.txt         — full metrics report

Usage
-----
    python scripts/evaluate_detections.py \\
        --pred-dir  validation_results_run3 \\
        --save-dir  evaluation_run3

Optional flags:
    --labels-dir  data/validation/labels/labels  (default)
    --rgb-dir     data/validation/rgb/rgb         (default)
    --min-px      1   minimum GT anthill pixels to classify image as positive
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from loguru import logger  # type: ignore

# Ensure the repo root (parent of scripts/) is on sys.path so that `app`
# is importable when the script is invoked as `python scripts/evaluate_detections.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.core.logging_config import setup_logging  # noqa: E402
from app.infrastructure.google_drive_client import GoogleDriveClient  # noqa: E402
from app.service.data_service import DataService  # noqa: E402

#  constants 

LABEL_BACKGROUND: int = 0
LABEL_ANTHILL: int = 1
LABEL_IGNORE: int = 255
RGB_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

#  I/O helpers 


def load_label(path: Path) -> np.ndarray:
    """Load a ground-truth label PNG.

    Labels are colour-coded RGB images:
        Red   (R>150, G<100, B<100) → 1  (anthill)
        Black (all channels < 50)   → 0  (background)
        White (all channels > 200)  → 255 (ignore / border)

    Returns a 2-D uint8 array with values 0, 1, or 255.
    """
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    label = np.full(arr.shape[:2], 255, dtype=np.uint8)
    label[(r < 50) & (g < 50) & (b < 50)] = 0       # black → background
    label[(r > 150) & (g < 100) & (b < 100)] = 1     # red   → anthill
    return label


def load_pred_mask(path: Optional[Path], *, shape: tuple[int, int]) -> np.ndarray:
    """Load a predicted mask PNG saved by run_validation.py.

    The service saves masks as ``(pred_mask * 255).astype(uint8)``, so values
    are 0 or 255.  Returns a binary 2-D uint8 array (0 = background, 1 = anthill).
    If *path* is None or does not exist an all-zero mask of *shape* is returned.
    """
    if path is None or not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def find_rgb(stem: str, rgb_dir: Path) -> Optional[Path]:
    """Return the first matching RGB file for *stem* in *rgb_dir*, or None."""
    for ext in RGB_EXTENSIONS:
        candidate = rgb_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def image_has_anthill(label: np.ndarray, min_px: int = 1) -> bool:
    """Return True when *label* contains at least *min_px* anthill pixels."""
    return int((label == LABEL_ANTHILL).sum()) >= max(min_px, 1)


#  case saving 


def save_case(
    stem: str,
    dest_dir: Path,
    gt_label: np.ndarray,
    pred_mask: np.ndarray,
    gt_rgb_path: Optional[Path],
    pred_rgb_path: Optional[Path],
) -> None:
    """Save diagnostics for a single FP or FN image into *dest_dir*.

    Four files are written per case:
        {stem}_gt_label.png   — ground-truth label (anthill=white, ignore=grey)
        {stem}_gt_rgb.png     — original RGB image
        {stem}_pred_mask.png  — predicted mask (anthill=white)
        {stem}_pred_rgb.png   — RGB saved by the validation service (or copy of GT RGB)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # --- GT label: visualise 0→black, 1→white, 255→grey -----------------------
    label_vis = np.zeros_like(gt_label)
    label_vis[gt_label == LABEL_ANTHILL] = 255
    label_vis[gt_label == LABEL_IGNORE] = 128
    Image.fromarray(label_vis, mode="L").save(dest_dir / f"{stem}_gt_label.png")

    # --- GT RGB ----------------------------------------------------------------
    if gt_rgb_path and gt_rgb_path.exists():
        Image.open(gt_rgb_path).save(dest_dir / f"{stem}_gt_rgb.png")

    # --- Predicted mask --------------------------------------------------------
    pred_vis = (pred_mask * 255).astype(np.uint8)
    Image.fromarray(pred_vis, mode="L").save(dest_dir / f"{stem}_pred_mask.png")

    # --- Predicted RGB (falls back to GT RGB if the service did not save one) --
    if pred_rgb_path and pred_rgb_path.exists():
        Image.open(pred_rgb_path).save(dest_dir / f"{stem}_pred_rgb.png")
    elif gt_rgb_path and gt_rgb_path.exists():
        Image.open(gt_rgb_path).save(dest_dir / f"{stem}_pred_rgb.png")


#  metrics accumulator 


@dataclass
class Metrics:
    """Accumulates image-level and pixel-level evaluation statistics."""

    total_images: int = 0

    # image-level
    gt_positive: int = 0
    pred_positive: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    # pixel-level: per-class [background, anthill]
    px_inter: list[int] = field(default_factory=lambda: [0, 0])
    px_union: list[int] = field(default_factory=lambda: [0, 0])
    px_pred_sum: list[int] = field(default_factory=lambda: [0, 0])
    px_gt_sum: list[int] = field(default_factory=lambda: [0, 0])
    px_correct: int = 0
    px_total: int = 0

    def update_image_level(self, gt_pos: bool, pred_pos: bool) -> None:
        self.total_images += 1
        if gt_pos:
            self.gt_positive += 1
        if pred_pos:
            self.pred_positive += 1
        if gt_pos and pred_pos:
            self.tp += 1
        elif not gt_pos and pred_pos:
            self.fp += 1
        elif gt_pos and not pred_pos:
            self.fn += 1
        else:
            self.tn += 1

    def update_pixel_level(self, gt: np.ndarray, pred: np.ndarray) -> None:
        """Accumulate pixel statistics.

        Parameters
        ----------
        gt:   uint8 label array with values 0/1/255.
        pred: binary uint8 mask array with values 0/1.
        """
        valid = gt != LABEL_IGNORE
        gt_v = gt[valid].astype(np.int32)
        pred_v = pred[valid].astype(np.int32)

        self.px_correct += int((gt_v == pred_v).sum())
        self.px_total += int(valid.sum())

        for cls in (LABEL_BACKGROUND, LABEL_ANTHILL):
            gt_cls = gt_v == cls
            pred_cls = pred_v == cls
            self.px_inter[cls] += int((gt_cls & pred_cls).sum())
            self.px_union[cls] += int((gt_cls | pred_cls).sum())
            self.px_pred_sum[cls] += int(pred_cls.sum())
            self.px_gt_sum[cls] += int(gt_cls.sum())

    def report(self) -> str:  # noqa: PLR0912
        """Build and return the full metrics report string."""
        sep = "=" * 62
        lines: list[str] = [sep, "  ANTHILL DETECTION — EVALUATION METRICS", sep, ""]

        lines += [
            f"  Total images evaluated:        {self.total_images}",
            f"  GT positives (label has FQ):   {self.gt_positive}",
            f"  GT negatives (label empty):    {self.total_images - self.gt_positive}",
            f"  Pred positives (detected):     {self.pred_positive}",
            f"  Pred negatives (not detected): {self.total_images - self.pred_positive}",
            "",
            "-" * 62,
            "  Image-level Detection",
            "-" * 62,
            f"  TP (correct detection):    {self.tp:>6}",
            f"  FP (false alarm):          {self.fp:>6}",
            f"  FN (missed anthill):       {self.fn:>6}",
            f"  TN (correct rejection):    {self.tn:>6}",
            "",
        ]

        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        lines += [
            f"  Precision:                 {precision:.4f}  ({precision * 100:.1f}%)",
            f"  Recall / Detection Rate:   {recall:.4f}  ({recall * 100:.1f}%)",
            f"  F1 Score:                  {f1:.4f}  ({f1 * 100:.1f}%)",
            "",
            "-" * 62,
            "  Pixel-level Segmentation",
            "-" * 62,
        ]

        px_acc = self.px_correct / self.px_total if self.px_total > 0 else 0.0
        lines.append(
            f"  Pixel Accuracy:                {px_acc:.4f}  ({px_acc * 100:.1f}%)"
        )
        lines.append("")

        cls_names = ["Background", "Anthill   "]
        ious: list[float] = []
        dices: list[float] = []

        for cls in (LABEL_BACKGROUND, LABEL_ANTHILL):
            inter = self.px_inter[cls]
            union = self.px_union[cls]
            pred_s = self.px_pred_sum[cls]
            gt_s = self.px_gt_sum[cls]

            iou = inter / union if union > 0 else 1.0
            dice_denom = pred_s + gt_s
            dice = 2 * inter / dice_denom if dice_denom > 0 else 1.0

            ious.append(iou)
            dices.append(dice)

            lines += [
                f"  [{cls_names[cls]}]  IoU:  {iou:.4f}  ({iou * 100:.1f}%)",
                f"  [{cls_names[cls]}]  Dice: {dice:.4f}  ({dice * 100:.1f}%)",
                "",
            ]

        mean_iou = sum(ious) / len(ious)
        mean_dice = sum(dices) / len(dices)

        lines += [
            f"  Mean IoU:                      {mean_iou:.4f}  ({mean_iou * 100:.1f}%)",
            f"  Mean Dice:                     {mean_dice:.4f}  ({mean_dice * 100:.1f}%)",
            "",
            sep,
            "",
            "-" * 62,
            "  Glossário de Métricas",
            "-" * 62,
            "  [Nível de imagem — detecção]",
            "  TP  True Positive  — imagem COM formigueiro na label e modelo",
            "                       DETECTOU  → acerto correto",
            "  FP  False Positive — imagem SEM formigueiro na label, mas",
            "                       modelo 'detectou' → alarme falso",
            "  FN  False Negative — imagem COM formigueiro na label, mas",
            "                       modelo NÃO detectou → formigueiro perdido",
            "  TN  True Negative  — imagem SEM formigueiro na label e modelo",
            "                       NÃO detectou nada → rejeição correta",
            "",
            "  Precision  = TP / (TP + FP)",
            "               Dos que o modelo disse 'tem formigueiro',",
            "               quantos realmente tinham?",
            "",
            "  Recall     = TP / (TP + FN)",
            "               Dos que REALMENTE tinham formigueiro,",
            "               quantos o modelo encontrou?",
            "",
            "  F1 Score   = 2 × Precision × Recall / (Precision + Recall)",
            "               Média harmônica entre Precision e Recall.",
            "",
            "  [Nível de pixel — segmentação]",
            "  Pixel Accuracy  Proporção de pixels classificados corretamente",
            "                  (ignora pixels de borda marcados como 255).",
            "",
            "  IoU   Intersection over Union",
            "        Sobreposição pixel a pixel: área da interseção ÷ área",
            "        da união entre máscara prevista e label ground-truth.",
            "",
            "  Dice  Dice Coefficient (F1 de pixels)",
            "        Medida de sobreposição similar ao IoU, mais sensível",
            "        a regiões pequenas: 2×interseção / (pred + label).",
            "",
            "  mIoU  Mean IoU  — média do IoU das duas classes",
            "                   (fundo + formigueiro).",
            "",
            "  [Pastas de saída]",
            "  somente_original/   → FN: formigueiro na label, não na predição",
            "  somente_resultado/  → FP: formigueiro na predição, não na label",
            "",
            sep,
        ]

        return "\n".join(lines)


#  main 


def _parse_args() -> argparse.Namespace:
    _default_labels = str(
        Path(settings.local_data_dir) / settings.val_labels_subdir
    )
    _default_rgb = str(
        Path(settings.local_data_dir) / settings.val_rgb_subdir
    )
    parser = argparse.ArgumentParser(
        description="Evaluate anthill detection quality vs ground-truth labels."
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory that contains *_mask.png and *_rgb.png files produced by run_validation.py",
    )
    parser.add_argument(
        "--labels-dir",
        default=_default_labels,
        help=f"Ground-truth label directory (default: {_default_labels}). Ignored in online mode.",
    )
    parser.add_argument(
        "--rgb-dir",
        default=_default_rgb,
        help=f"Validation RGB image directory (default: {_default_rgb}). Ignored in online mode.",
    )
    parser.add_argument(
        "--save-dir",
        default="evaluation_output",
        help="Root directory to write FP/FN cases and metrics.txt (default: evaluation_output)",
    )
    parser.add_argument(
        "--min-px",
        type=int,
        default=1,
        help="Minimum GT anthill pixels required to classify an image as positive (default: 1)",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    setup_logging()

    pred_dir = Path(args.pred_dir)
    save_dir = Path(args.save_dir)
    logger.info(f"Evaluation started with settings: {args}\n mode = {settings.data_mode}\n")
    if settings.data_mode == "online":
        logger.info("Online mode — downloading validation data from Google Drive…")
        drive_client = GoogleDriveClient()
        data_service = DataService(storage_client=drive_client)
        labels_dir, rgb_dir = await data_service.download_validation_from_drive()
        logger.info(f"Download complete → labels: {labels_dir}  rgb: {rgb_dir}\n")
    else:
        labels_dir = Path(args.labels_dir)
        rgb_dir = Path(args.rgb_dir)

    fn_dir = save_dir / "somente_original"   # FN: anthill in label only
    fp_dir = save_dir / "somente_resultado"  # FP: anthill in prediction only

    if not pred_dir.exists():
        logger.error(f"pred-dir '{pred_dir}' does not exist.")
        sys.exit(1)
    if not labels_dir.exists():
        logger.error(f"labels-dir '{labels_dir}' does not exist.")
        sys.exit(1)

    label_files = sorted(labels_dir.glob("*.png"))
    if not label_files:
        logger.error(f"no *.png files found in '{labels_dir}'.")
        sys.exit(1)

    logger.info(f"Labels dir:  {labels_dir}  ({len(label_files)} files)")
    logger.info(f"Pred dir:    {pred_dir}")
    logger.info(f"RGB dir:     {rgb_dir}")
    logger.info(f"Save dir:    {save_dir}")
    logger.info(f"Min GT px:   {args.min_px}")

    metrics = Metrics()

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
            # FN: anthill missed by model — save to somente_original
            save_case(stem, fn_dir, gt_label, pred_mask, gt_rgb_path, pred_rgb_path)
        elif pred_pos and not gt_pos:
            # FP: false detection — save to somente_resultado
            save_case(stem, fp_dir, gt_label, pred_mask, gt_rgb_path, pred_rgb_path)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(label_files):
            logger.info(
                f"  [{idx + 1:>5}/{len(label_files)}]  "
                f"TP={metrics.tp}  FP={metrics.fp}  "
                f"FN={metrics.fn}  TN={metrics.tn}"
            )

    report = metrics.report()
    logger.info(report)

    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / "metrics.txt"
    report_path.write_text(report, encoding="utf-8")

    logger.info(f"Report saved  →  {report_path}")
    logger.info(f"FN cases      →  {fn_dir}  ({metrics.fn} images)")
    logger.info(f"FP cases      →  {fp_dir}  ({metrics.fp} images)")


if __name__ == "__main__":
    asyncio.run(main())

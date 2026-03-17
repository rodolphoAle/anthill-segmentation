"""Async service for streaming model validation and metrics computation.

How it works
-----------
1. File metadata (IDs) for ``validacao/rgb`` and ``validacao/labels``
   are fetched from Google Drive — no bulk download.
2. For every matched RGB/label pair:
   a. Both files are downloaded as in-memory ``BytesIO`` buffers.
   b. Inference is run on the RGB image.
   c. Pixel accuracy, IoU, and Dice metrics are accumulated.
   d. If the predicted mask contains at least one anthill pixel
      (class = 1), the original image *and* the predicted mask are
      saved to ``output_dir`` — these are the only files written to disk.
3. Aggregated metrics are returned in a :class:`ValidationMetrics`
   dataclass.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from loguru import logger  # pyright: ignore[reportMissingImports]
from PIL import Image

from app.core.config import settings
from app.core.exceptions import DatasetNotFoundError, FolderNotFoundError
from app.domain.protocols import StorageClientProtocol


#  Result dataclass 

@dataclass
class ValidationMetrics:
    """Aggregated metrics produced by a full validation run."""

    total_images: int = 0
    anthill_detections: int = 0
    pixel_accuracy: float = 0.0
    mean_iou: float = 0.0
    mean_dice: float = 0.0
    per_image_iou: list[float] = field(default_factory=list)
    per_image_dice: list[float] = field(default_factory=list)


#  Service 

class ValidationService:
    """Streams validation images from Drive and evaluates the model.

    Image bytes are downloaded one pair at a time and released from
    memory immediately after inference.  Only images where an anthill
    is detected are written to disk (in ``output_dir``).

    Args:
        model: Trained ``nn.Module`` with weights already loaded.
        device: ``torch.device`` used for inference.
        storage_client: Storage backend implementing
            :class:`~app.domain.protocols.StorageClientProtocol`.
        output_dir: Folder where anthill detection images are saved.
            Defaults to ``settings.validation_output_dir``.
    """

    _ANTHILL_CLASS: int = 1

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        storage_client: StorageClientProtocol,
        output_dir: str | Path | None = None,
    ) -> None:
        self._model = model
        self._device = device
        self._storage = storage_client
        self._output_dir = Path(output_dir or settings.validation_output_dir)
        self._transform = transforms.Compose([transforms.PILToTensor()])

    #  folder navigation 

    async def _resolve_subfolder_id(
        self, parent_id: str, *names: str
    ) -> str:
        current_id = parent_id
        for name in names:
            folder_id = await self._storage.get_folder_id(name, current_id)
            if folder_id is None:
                raise FolderNotFoundError(
                    f"Folder '{name}' not found under parent '{current_id}'"
                )
            current_id = folder_id
        return current_id

    #  sync helpers (run inside thread pool) 

    def _predict_sync(self, image: Image.Image) -> np.ndarray:
        tensor = (
            self._transform(image).unsqueeze(0).float().to(self._device)
        )
        self._model.eval()
        with torch.no_grad():
            output = self._model(tensor)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        return prediction

    def _save_anthill_result_sync(
        self,
        image: Image.Image,
        pred_mask: np.ndarray,
        filename_stem: str,
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        image.save(self._output_dir / f"{filename_stem}_rgb.png")
        mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        mask_img.save(self._output_dir / f"{filename_stem}_mask.png")

    #  metric helpers (pure functions) 

    @staticmethod
    def _iou(pred: np.ndarray, target: np.ndarray, cls: int) -> float:
        p = pred == cls
        t = target == cls
        inter = int((p & t).sum())
        union = int((p | t).sum())
        if union == 0:
            return 1.0 if p.sum() == 0 else 0.0
        return inter / union

    @staticmethod
    def _dice(pred: np.ndarray, target: np.ndarray, cls: int) -> float:
        p = pred == cls
        t = target == cls
        numerator = 2 * int((p & t).sum())
        denominator = int(p.sum()) + int(t.sum())
        if denominator == 0:
            return 1.0 if p.sum() == 0 else 0.0
        return numerator / denominator

    #  pair matching 

    @staticmethod
    def _match_pairs(
        rgb_files: list[dict[str, str]],
        label_files: list[dict[str, str]],
    ) -> list[tuple[dict[str, str], dict[str, str]]]:
        pairs: list[tuple[dict[str, str], dict[str, str]]] = []
        for rgb in sorted(rgb_files, key=lambda f: f["name"]):
            prefix = "_".join(Path(rgb["name"]).stem.split("_")[:4])
            matched = [
                lf
                for lf in label_files
                if Path(lf["name"]).stem.startswith(prefix)
            ]
            if matched:
                pairs.append((rgb, matched[0]))
        return pairs

    #  public async API 

    async def run(self, base_folder_id: str) -> ValidationMetrics:
        """Stream-validate all pairs in ``validacao/`` and return metrics.

        Args:
            base_folder_id: Root Drive folder that contains the
                ``validacao/rgb`` and ``validacao/labels`` sub-folders.

        Returns:
            Fully populated :class:`ValidationMetrics`.

        Raises:
            DatasetNotFoundError: When no matched pairs are found.
            FolderNotFoundError: When a required sub-folder is missing.
        """
        rgb_folder_id = await self._resolve_subfolder_id(
            base_folder_id, "validacao", "rgb"
        )
        labels_folder_id = await self._resolve_subfolder_id(
            base_folder_id, "validacao", "labels"
        )

        img_exts = [".png", ".jpg", ".jpeg", ".tif"]
        rgb_files = await self._storage.list_files(rgb_folder_id, img_exts)
        label_files = await self._storage.list_files(labels_folder_id, [".png"])

        pairs = self._match_pairs(rgb_files, label_files)
        if not pairs:
            raise DatasetNotFoundError(
                "No validation image-mask pairs found on Drive"
            )

        logger.info(
            "Validation started — {} pair(s) (streaming, no disk writes unless anthill detected)",
            len(pairs),
        )

        metrics = ValidationMetrics(total_images=len(pairs))
        total_correct = 0
        total_pixels = 0

        for idx, (rgb_meta, label_meta) in enumerate(pairs, start=1):
            # Download both files into memory buffers
            rgb_buffer = await self._storage.download_file(rgb_meta["id"])
            label_buffer = await self._storage.download_file(label_meta["id"])

            assert isinstance(rgb_buffer, io.BytesIO), (
                "download_file must return BytesIO when no destination_path is given"
            )
            assert isinstance(label_buffer, io.BytesIO)

            rgb_buffer.seek(0)
            label_buffer.seek(0)

            image = Image.open(rgb_buffer).convert("RGB")
            gt_mask = np.clip(np.array(Image.open(label_buffer)), 0, 1)

            # Inference (offloaded to thread pool — keeps event loop free)
            pred_mask: np.ndarray = await asyncio.to_thread(
                self._predict_sync, image
            )

            # Pixel accuracy accumulation
            total_correct += int((pred_mask == gt_mask).sum())
            total_pixels += int(gt_mask.size)

            # Per-image mean IoU and Dice over both classes
            iou = float(np.mean([self._iou(pred_mask, gt_mask, c) for c in (0, 1)]))
            dice = float(np.mean([self._dice(pred_mask, gt_mask, c) for c in (0, 1)]))
            metrics.per_image_iou.append(iou)
            metrics.per_image_dice.append(dice)

            # % of pixels the model classified as anthill
            anthill_pct = float((pred_mask == self._ANTHILL_CLASS).mean() * 100)

            saved = False
            if anthill_pct >= settings.anthill_save_threshold:
                metrics.anthill_detections += 1
                stem = Path(rgb_meta["name"]).stem
                await asyncio.to_thread(
                    self._save_anthill_result_sync, image, pred_mask, stem
                )
                saved = True

            logger.info(
                "[{}/{}] {:.1f}% anthill{}  '{}'",
                idx,
                len(pairs),
                anthill_pct,
                " ✓ saved" if saved else "",
                rgb_meta["name"],
            )

        metrics.pixel_accuracy = total_correct / max(total_pixels, 1)
        metrics.mean_iou = float(np.mean(metrics.per_image_iou))
        metrics.mean_dice = float(np.mean(metrics.per_image_dice))

        logger.info(
            "Validation complete — "
            "PixelAcc={:.4f}  mIoU={:.4f}  MeanDice={:.4f}  "
            "anthill detections={} (saved to '{}')",
            metrics.pixel_accuracy,
            metrics.mean_iou,
            metrics.mean_dice,
            metrics.anthill_detections,
            self._output_dir,
        )
        return metrics

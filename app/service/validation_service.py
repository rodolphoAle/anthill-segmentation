"""Async service for streaming model validation and metrics computation."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from loguru import logger
from PIL import Image
from scipy.ndimage import label as ndimage_label

from app.core.config import settings
from app.core.exceptions import DatasetNotFoundError, FolderNotFoundError
from app.domain.mask_utils import decode_rgb_mask
from app.domain.metrics import ValidationMetrics
from app.domain.protocols import StorageClientProtocol


class ValidationService:
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
        self._transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    async def _resolve_subfolder_id(self, parent_id: str, *names: str) -> str:
        current_id = parent_id

        for name in names:
            folder_id = await self._storage.get_folder_id(name, current_id)

            if folder_id is None:
                raise FolderNotFoundError(
                    f"Folder '{name}' not found under parent '{current_id}'"
                )

            current_id = folder_id

        return current_id

    @staticmethod
    def _decode_gt_mask(mask_image: Image.Image) -> np.ndarray:
        """Decode RGB label mask into class ids using shared utility."""
        return decode_rgb_mask(mask_image)

    def _predict_sync(self, image: Image.Image) -> np.ndarray:
        tensor = self._transform(image).unsqueeze(0).to(self._device)

        self._model.eval()

        with torch.no_grad():
            output = self._model(tensor)
            probs = F.softmax(output, dim=1)
            anthill_prob = probs[0, self._ANTHILL_CLASS].cpu().numpy()

            prediction = (
                anthill_prob >= settings.anthill_confidence_threshold
            ).astype(np.uint8)

        if settings.use_region_filter:
            return self._filter_small_regions(prediction)

        return prediction

    @staticmethod
    def _filter_small_regions(mask: np.ndarray) -> np.ndarray:
        min_px = settings.min_anthill_region_px
        max_px = settings.max_anthill_region_px

        if min_px <= 1 and max_px <= 0:
            return mask

        labeled, num_features = ndimage_label(mask)

        if num_features == 0:
            return mask

        filtered = np.zeros_like(mask)

        for region_id in range(1, num_features + 1):
            size = int((labeled == region_id).sum())

            too_small = min_px > 1 and size < min_px
            too_large = max_px > 0 and size > max_px

            if not too_small and not too_large:
                filtered[labeled == region_id] = 1

        return filtered

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

    async def run(self, base_folder_id: str) -> ValidationMetrics:
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
            "Validation started  {} pair(s)",
            len(pairs),
        )

        metrics = ValidationMetrics(total_images=len(pairs))

        total_correct = 0
        total_pixels = 0

        for idx, (rgb_meta, label_meta) in enumerate(pairs, start=1):
            rgb_buffer = await self._storage.download_file(rgb_meta["id"])
            label_buffer = await self._storage.download_file(label_meta["id"])

            assert isinstance(rgb_buffer, io.BytesIO)
            assert isinstance(label_buffer, io.BytesIO)

            rgb_buffer.seek(0)
            label_buffer.seek(0)

            image = Image.open(rgb_buffer).convert("RGB")
            gt_mask = self._decode_gt_mask(Image.open(label_buffer))

            pred_mask: np.ndarray = await asyncio.to_thread(
                self._predict_sync,
                image,
            )

            valid_mask = gt_mask != 255

            if valid_mask.sum() == 0:
                logger.warning(
                    "[{}/{}] skipping '{}': all pixels are ignore_index",
                    idx,
                    len(pairs),
                    rgb_meta["name"],
                )
                continue

            pred_valid = pred_mask[valid_mask]
            gt_valid = gt_mask[valid_mask]

            total_correct += int((pred_valid == gt_valid).sum())
            total_pixels += int(valid_mask.sum())

            iou = float(
                np.mean([
                    self._iou(pred_valid, gt_valid, cls)
                    for cls in (0, 1)
                ])
            )

            dice = float(
                np.mean([
                    self._dice(pred_valid, gt_valid, cls)
                    for cls in (0, 1)
                ])
            )

            metrics.per_image_iou.append(iou)
            metrics.per_image_dice.append(dice)

            anthill_pct = float((pred_mask == self._ANTHILL_CLASS).mean() * 100)

            if anthill_pct >= settings.anthill_save_threshold:
                metrics.anthill_detections += 1
                stem = Path(rgb_meta["name"]).stem

                await asyncio.to_thread(
                    self._save_anthill_result_sync,
                    image,
                    pred_mask,
                    stem,
                )

        metrics.pixel_accuracy = total_correct / max(total_pixels, 1)
        metrics.mean_iou = float(np.mean(metrics.per_image_iou))
        metrics.mean_dice = float(np.mean(metrics.per_image_dice))

        logger.info(
            "Validation complete  "
            "PixelAcc={:.4f}  mIoU={:.4f}  MeanDice={:.4f}  "
            "anthill detections={} (saved to '{}')",
            metrics.pixel_accuracy,
            metrics.mean_iou,
            metrics.mean_dice,
            metrics.anthill_detections,
            self._output_dir,
        )

        return metrics
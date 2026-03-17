"""Async service for running segmentation inference on single images."""

from __future__ import annotations

import asyncio
import io

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from loguru import logger
from PIL import Image


class PredictionService:
    """Runs U-Net inference and returns segmentation masks.

    Args:
        model: A trained ``nn.Module`` (shared reference with
            :class:`~app.service.training_service.TrainingService`).
        device: ``torch.device`` used for computation.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self._model = model
        self._device = device
        self._transform: transforms.Compose = transforms.Compose([
            transforms.PILToTensor(),
        ])

    #  private (sync, thread-pool) 

    def _sync_predict(self, image_bytes: bytes) -> np.ndarray:
        """Run inference synchronously on raw image bytes."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = (
            self._transform(image)
            .unsqueeze(0)
            .float()
            .to(self._device)
        )

        self._model.eval()
        with torch.no_grad():
            output = self._model(tensor)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        return prediction

    #  public async API 

    async def predict(self, image_bytes: bytes) -> np.ndarray:
        """Return a segmentation mask (numpy array) for *image_bytes*."""
        logger.info("Running segmentation prediction")
        result: np.ndarray = await asyncio.to_thread(
            self._sync_predict, image_bytes,
        )
        logger.info("Prediction complete — mask shape {}", result.shape)
        return result

    async def predict_to_png(self, image_bytes: bytes) -> bytes:
        """Predict and encode the result as PNG bytes."""
        mask_array = await self.predict(image_bytes)
        mask_image = Image.fromarray(
            (mask_array * 255).astype(np.uint8),
        )
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()

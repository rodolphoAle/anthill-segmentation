from __future__ import annotations

import asyncio
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from loguru import logger
from PIL import Image
from scipy.ndimage import label as ndimage_label
from app.core.config import settings


class PredictionService:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self._model = model
        self._device = device
        self._transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _sync_predict(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._transform(image).unsqueeze(0).to(self._device)

        self._model.eval()
        with torch.no_grad():
            output = self._model(tensor)
            probs = F.softmax(output, dim=1)
            anthill_prob = probs[0, 1]
            
            # Apply confidence threshold from settings
            threshold = settings.anthill_confidence_threshold
            prediction = (anthill_prob >= threshold).cpu().numpy().astype(np.uint8)
            
            # Apply connected-component filter to remove isolated points and oversized blobs
            if settings.use_region_filter:
                prediction = self._filter_regions(prediction)

        return prediction
    
    def _filter_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove anthill regions outside the configured size range.
        
        Removes connected components with fewer pixels than min_anthill_region_px
        (noise fragments) or more pixels than max_anthill_region_px (false positives).
        """
        if mask.sum() == 0:  
            return mask
        
        labeled_array, num_features = ndimage_label(mask)        
        
        for region_id in range(1, num_features + 1):
            region_size = (labeled_array == region_id).sum()
            
            # Remove if too small
            if region_size < settings.min_anthill_region_px:
                mask[labeled_array == region_id] = 0
            
            # Remove if too large (only if filter is enabled, i.e., max_anthill_region_px > 0)
            elif settings.max_anthill_region_px > 0 and region_size > settings.max_anthill_region_px:
                mask[labeled_array == region_id] = 0
        
        return mask

    async def predict(self, image_bytes: bytes) -> np.ndarray:
        logger.info("Running segmentation prediction")
        result = await asyncio.to_thread(self._sync_predict, image_bytes)
        logger.info("Prediction complete  mask shape {}", result.shape)
        return result

    async def predict_to_png(self, image_bytes: bytes) -> bytes:
        mask_array = await self.predict(image_bytes)
        mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))

        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()
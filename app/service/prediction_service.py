"""Serviço de predição para segmentação de formigueiros.

Responsável por:
- receber uma imagem em bytes;
- aplicar o mesmo pré-processamento usado no treinamento;
- executar inferência com a U-Net;
- gerar uma máscara binária de formigueiro;
- aplicar filtro por tamanho de região;
- retornar a máscara como array ou PNG.
"""

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
    """Serviço responsável pela inferência do modelo."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self._model = model
        self._device = device

        # Mesmo pré-processamento usado no treinamento.
        self._transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _sync_predict(self, image_bytes: bytes) -> np.ndarray:
        """Executa a predição de forma síncrona."""

        # Converte bytes em imagem RGB.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Converte imagem para tensor e adiciona dimensão de batch.
        tensor = self._transform(image).unsqueeze(0).to(self._device)

        self._model.eval()

        with torch.no_grad():
            # Executa inferência.
            output = self._model(tensor)

            # Converte logits em probabilidades.
            probs = F.softmax(output, dim=1)

            # Seleciona probabilidade da classe formigueiro.
            anthill_prob = probs[0, 1]

            # Aplica threshold de confiança.
            threshold = settings.anthill_confidence_threshold

            prediction = (
                anthill_prob >= threshold
            ).cpu().numpy().astype(np.uint8)

            # Remove regiões muito pequenas ou muito grandes.
            if settings.use_region_filter:
                prediction = self._filter_regions(prediction)

        return prediction

    def _filter_regions(self, mask: np.ndarray) -> np.ndarray:
        """Filtra regiões detectadas fora do tamanho esperado."""

        # Se não houver pixels positivos, retorna a máscara original.
        if mask.sum() == 0:
            return mask

        # Identifica componentes conectados na máscara.
        labeled_array, num_features = ndimage_label(mask)

        for region_id in range(1, num_features + 1):
            region_size = (labeled_array == region_id).sum()

            # Remove ruídos pequenos.
            if region_size < settings.min_anthill_region_px:
                mask[labeled_array == region_id] = 0

            # Remove regiões grandes demais, geralmente falsos positivos.
            elif (
                settings.max_anthill_region_px > 0
                and region_size > settings.max_anthill_region_px
            ):
                mask[labeled_array == region_id] = 0

        return mask

    async def predict(self, image_bytes: bytes) -> np.ndarray:
        """Executa predição assíncrona e retorna máscara binária."""

        logger.info("Running segmentation prediction")

        # Executa inferência em thread separada para não bloquear o loop async.
        result = await asyncio.to_thread(
            self._sync_predict,
            image_bytes,
        )

        logger.info("Prediction complete  mask shape {}", result.shape)

        return result

    async def predict_to_png(self, image_bytes: bytes) -> bytes:
        """Executa predição e retorna a máscara em formato PNG."""

        # Gera máscara binária.
        mask_array = await self.predict(image_bytes)

        # Converte 0/1 para 0/255 para visualização.
        mask_image = Image.fromarray(
            (mask_array * 255).astype(np.uint8)
        )

        # Salva PNG em memória.
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)

        return buffer.getvalue()
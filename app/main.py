"""Async CLI entry point for the UNet segmentation pipeline.

Run locally::

    python -m app.main

Or via Docker (see ``docker-compose.yml``)::

    docker compose up

Pipeline modes (set via ``UNET_PIPELINE_MODE`` env-var or ``.env`` file):

* ``train``    — streams training + validation images from Drive,
                 trains the model, and saves weights to disk.
* ``validate`` — loads saved weights, streams validation images from
                 Drive, computes metrics (pixel accuracy, mIoU, Dice),
                 and saves anthill-detected images to
                 ``UNET_VALIDATION_OUTPUT_DIR``.

In both modes **images are never bulk-downloaded to disk** — each pair
is fetched into memory, processed, and released before the next one
is fetched.
"""

from __future__ import annotations

import asyncio

import torch
from loguru import logger  # pyright: ignore[reportMissingImports]

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.domain.unet import UNet
from app.infrastructure.google_drive_client import GoogleDriveClient
from app.service.data_service import DataService
from app.service.training_service import TrainingService
from app.service.validation_service import ValidationService


async def _run_train(
    data_service: DataService,
    training_service: TrainingService,
) -> None:
    """Stream-train the model and save the resulting weights."""
    if settings.data_mode == "online":
        logger.info("Fetching file metadata from Google Drive (train mode)…")
        train_loader, val_loader = (
            await data_service.create_streaming_dataloaders_from_drive(
                base_folder_id=settings.base_folder_id,
            )
        )
    else:
        logger.info(
            "Loading local dataset from '{}' (offline mode)…",
            settings.local_data_dir,
        )
        train_loader, val_loader = await data_service.create_local_dataloaders()

    logger.info("Starting training…")
    await training_service.start_training(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=settings.num_epochs,
        learning_rate=settings.learning_rate,
    )

    state = training_service.state
    logger.info(
        "Training complete — final loss: {:.4f} | val loss: {:.4f}",
        state.current_loss,
        state.val_loss or 0.0,
    )

    saved_path = await training_service.save_model(settings.model_save_path)
    logger.info("Model saved to {}", saved_path)


async def _run_validate(
    drive_client: GoogleDriveClient,
    training_service: TrainingService,
) -> None:
    """Load saved weights and run streaming validation with metric reporting."""
    weights_path = settings.best_model_params_path
    logger.info("Loading model weights from '{}'…", weights_path)
    state_dict = await asyncio.to_thread(
        torch.load, weights_path, map_location=training_service.device
    )
    training_service.model.load_state_dict(state_dict)
    logger.info("Weights loaded successfully.")

    validation_service = ValidationService(
        model=training_service.model,
        device=training_service.device,
        storage_client=drive_client,
        output_dir=settings.validation_output_dir,
    )

    metrics = await validation_service.run(
        base_folder_id=settings.base_folder_id,
    )

    logger.info(
        "Results — PixelAcc={:.4f}  mIoU={:.4f}  MeanDice={:.4f}  "
        "Anthill detections={}",
        metrics.pixel_accuracy,
        metrics.mean_iou,
        metrics.mean_dice,
        metrics.anthill_detections,
    )


async def main() -> None:
    """Dispatch to the pipeline defined by ``settings.pipeline_mode``."""
    setup_logging(debug=settings.debug)
    logger.info(
        "Starting {} (mode={}, pipeline={})",
        settings.app_name,
        settings.data_mode,
        settings.pipeline_mode,
    )

    #  Build shared services (Dependency Injection) 
    drive_client = GoogleDriveClient(
        credentials_path=settings.google_credentials_path,
    )
    data_service = DataService(storage_client=drive_client)

    model = UNet(
        n_channels=settings.n_channels,
        n_classes=settings.n_classes,
    )
    training_service = TrainingService(model=model)
    logger.info("Using device: {}", training_service.device)

    #  Dispatch 
    if settings.pipeline_mode == "validate":
        await _run_validate(drive_client, training_service)
    else:
        await _run_train(data_service, training_service)


if __name__ == "__main__":
    asyncio.run(main())


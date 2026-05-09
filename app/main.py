"""Unified CLI entry point for the UNet anthill segmentation pipeline.

Usage::

    python -m app.main train     [--epochs N] [--lr LR] [--batch-size N] ...
    python -m app.main validate  [--device cpu] [--output-dir DIR] ...
    python -m app.main evaluate  --pred-dir DIR [--save-dir DIR] ...

Each sub-command accepts ``--help`` for a full list of flags.

All CLI flags are optional and override the corresponding ``UNET_*``
env-var / ``.env`` value for the current run only.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import torch
from loguru import logger  # pyright: ignore[reportMissingImports]

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.domain.unet import UNet
from app.infrastructure.google_drive_client import GoogleDriveClient
from app.service.data_service import DataService
from app.service.training_service import TrainingService
from app.service.validation_service import ValidationService


#  Pipeline functions (called by both the CLI and legacy env-var mode) 


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


async def pipeline_main() -> None:
    """Dispatch to the pipeline defined by ``settings.pipeline_mode``.

    Called by sub-command ``run()`` functions after they have pushed
    their CLI overrides into ``os.environ``.
    """
    setup_logging(debug=settings.debug)
    logger.info(
        "Starting {} (mode={}, pipeline={})",
        settings.app_name,
        settings.data_mode,
        settings.pipeline_mode,
    )

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

    if settings.pipeline_mode == "validate":
        await _run_validate(drive_client, training_service)
    else:
        await _run_train(data_service, training_service)


#  Unified CLI 


def cli() -> None:
    """Build the unified CLI with ``train``, ``validate``, and ``evaluate``
    sub-commands and dispatch to the appropriate runner."""
    from scripts.run_training import build_parser as train_parser, run as run_train
    from scripts.run_validation import build_parser as val_parser, run as run_validate
    from scripts.run_evaluate import build_parser as eval_parser, run as run_evaluate
    from scripts.validate_dataset import build_parser as vds_parser, run as run_vds

    parser = argparse.ArgumentParser(
        prog="unet-pipeline",
        description="UNet anthill segmentation — training, validation & evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser(subparsers)
    val_parser(subparsers)
    eval_parser(subparsers)
    vds_parser(subparsers)

    args = parser.parse_args()

    dispatch = {
        "train": run_train,
        "validate": run_validate,
        "evaluate": run_evaluate,
        "validate-dataset": run_vds,
    }
    dispatch[args.command](args)


#  Entry points 

# Keep backward-compat alias so old ``from app.main import main`` still works.
main = pipeline_main

if __name__ == "__main__":
    # If called with sub-commands → unified CLI
    # If called without args → legacy env-var dispatch
    if len(sys.argv) > 1 and sys.argv[1] in ("train", "validate", "evaluate", "validate-dataset"):
        cli()
    else:
        asyncio.run(pipeline_main())


if __name__ == "__main__":
    asyncio.run(main())


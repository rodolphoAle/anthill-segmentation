"""Async service for model training and evaluation.

CPU / GPU-bound work is offloaded via ``asyncio.to_thread`` so the
main event loop stays responsive while heavy computation runs in
the default thread-pool executor.
"""

from __future__ import annotations

import asyncio
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.exceptions import TrainingAlreadyInProgressError


#  Training state 

class TrainingStatus(str, Enum):
    """Finite-state representation of a training job."""

    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingState:
    """Mutable snapshot of the current training run."""

    def __init__(self) -> None:
        self.status: TrainingStatus = TrainingStatus.IDLE
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.current_loss: float = 0.0
        self.val_loss: float | None = None
        self.error_message: str | None = None


#  Service 

class TrainingService:
    """Orchestrates model training, evaluation, and persistence.

    A single instance manages **one** model and tracks training
    progress through :attr:`state`.

    Args:
        model: PyTorch ``nn.Module`` to train.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._device = self._resolve_device()
        self._model.to(self._device)
        self._state = TrainingState()

    #  helpers 

    @staticmethod
    def _resolve_device() -> torch.device:
        """Determine the computing device from settings or auto-detect."""
        device_setting: str = getattr(settings, "device", "auto")
        if device_setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_setting)

    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> nn.Module:
        return self._model

    #  synchronous loops (run inside thread pool) 

    def _train_loop(
        self,
        train_loader: DataLoader[tuple],
        val_loader: DataLoader[tuple],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        num_epochs: int,
    ) -> None:
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self._state.current_epoch = epoch + 1

            # --- train ---
            self._model.train()
            epoch_loss = 0.0
            batch_count = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)

                optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / max(batch_count, 1)
            self._state.current_loss = avg_loss

            # --- validate ---
            val_loss = self._evaluate_loop_sync(val_loader, criterion)
            self._state.val_loss = val_loss
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch {}/{} — loss: {:.4f} | val_loss: {:.4f} | lr: {:.2e}",
                epoch + 1,
                num_epochs,
                avg_loss,
                val_loss,
                current_lr,
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self._model.state_dict(),
                    settings.best_model_params_path,
                )
                logger.info(
                    "New best val_loss={:.4f} — checkpoint saved to '{}'",
                    best_val_loss,
                    settings.best_model_params_path,
                )

    def _evaluate_loop_sync(
        self,
        val_loader: DataLoader[tuple],
        criterion: nn.Module,
    ) -> float:
        self._model.eval()
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                outputs = self._model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                batch_count += 1
        return total_loss / max(batch_count, 1)

    def _evaluate_loop(
        self,
        val_loader: DataLoader[tuple],
        criterion: nn.Module,
    ) -> float:
        return self._evaluate_loop_sync(val_loader, criterion)

    #  async public API 

    async def start_training(
        self,
        train_loader: DataLoader[tuple],
        val_loader: DataLoader[tuple],
        num_epochs: int | None = None,
        learning_rate: float | None = None,
    ) -> None:
        """Run a full training + validation cycle asynchronously.

        Raises:
            TrainingAlreadyInProgressError: If a job is already running.
        """
        if self._state.status == TrainingStatus.TRAINING:
            raise TrainingAlreadyInProgressError(
                "A training job is already in progress"
            )

        epochs = num_epochs or settings.num_epochs
        lr = learning_rate or settings.learning_rate

        self._state = TrainingState()
        self._state.status = TrainingStatus.PREPARING
        self._state.total_epochs = epochs

        # Weight class 1 (anthill) heavily to counter class imbalance.
        # Assumes anthill pixels are ~5-10x rarer than background.
        class_weights = torch.tensor([1.0, 10.0], device=self._device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        try:
            self._state.status = TrainingStatus.TRAINING
            logger.info(
                "Training started — {} epochs on {}",
                epochs,
                self._device,
            )
            await asyncio.to_thread(
                self._train_loop,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                epochs,
            )

            logger.info("Training done. Running final validation…")
            val_loss = await asyncio.to_thread(
                self._evaluate_loop, val_loader, criterion,
            )
            self._state.val_loss = val_loss
            self._state.status = TrainingStatus.COMPLETED
            logger.info("Final validation loss: {:.4f}", val_loss)

        except Exception as exc:
            self._state.status = TrainingStatus.FAILED
            self._state.error_message = str(exc)
            logger.error("Training failed: {}", exc)
            raise

    async def save_model(self, path: str | None = None) -> str:
        """Persist model weights to disk."""
        save_path = path or settings.model_save_path
        await asyncio.to_thread(
            torch.save, self._model.state_dict(), save_path,
        )
        logger.info("Model saved → {}", save_path)
        return save_path

    async def load_model(self, path: str | None = None) -> None:
        """Load model weights from disk."""
        load_path = path or settings.model_save_path
        state_dict = await asyncio.to_thread(
            torch.load, load_path, map_location=self._device,
        )
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        logger.info("Model loaded ← {}", load_path)

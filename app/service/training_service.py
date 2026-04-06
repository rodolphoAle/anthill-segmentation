"""Async service for model training and evaluation.

CPU / GPU-bound work is offloaded via ``asyncio.to_thread`` so the
main event loop stays responsive while heavy computation runs in
the default thread-pool executor.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger  # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.exceptions import TrainingAlreadyInProgressError


#  Focal Loss 

class FocalLoss(nn.Module):
    """Weighted Focal Loss for semantic segmentation.

    Focal Loss down-weights easy examples (confidently correct predictions)
    so training focuses on hard, ambiguous cases — exactly what we need
    when reddish soil confuses the model.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. 0.0 = standard CrossEntropyLoss.
               2.0 is the standard value from the original paper.
        weight: Per-class weights tensor (same as CrossEntropyLoss weight).
        ignore_index: Label value to ignore (e.g. 255 for boundary pixels).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        valid = targets != self.ignore_index
        return focal_loss[valid].mean()


#  Tversky Loss 

class TverskyLoss(nn.Module):
    """Tversky Loss — penalises FN more than FP when beta > alpha.

    TL = 1 - TP / (TP + alpha*FP + beta*FN)

    With alpha=0.3, beta=0.7: FN is weighted 2.3× more than FP, which
    directly optimises Recall at the cost of some Precision.  Ideal when
    missing a detection is costlier than a false alarm.

    Args:
        alpha: Weight for FP in the denominator (lower → more FP-tolerant).
        beta:  Weight for FN in the denominator (higher → stronger Recall push).
        smooth: Laplace smoothing to avoid division by zero.
        ignore_index: Label value excluded from the loss computation (e.g. 255).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (N, C, H, W) logits — C=2 (background, anthill)
        probs = F.softmax(inputs, dim=1)          # (N, 2, H, W)
        anthill_prob = probs[:, 1, :, :]          # (N, H, W)

        # Build binary target and valid mask
        valid_mask = targets != self.ignore_index  # (N, H, W)
        target_bin = (targets == 1).float()        # (N, H, W) — 1=anthill

        anthill_prob = anthill_prob[valid_mask]
        target_bin   = target_bin[valid_mask]

        tp = (anthill_prob * target_bin).sum()
        fp = (anthill_prob * (1.0 - target_bin)).sum()
        fn = ((1.0 - anthill_prob) * target_bin).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky_index


#  Combined Tversky + Focal Loss 

class CombinedTverskyFocalLoss(nn.Module):
    """Combines Tversky Loss and Focal Loss to prevent mode collapse.

    Pure Tversky/Dice losses collapse to "predict all background" during
    early training when the class imbalance is extreme: the alpha*FP term
    dominates, the gradient drives anthill_prob → 0 everywhere, and
    val_loss freezes at a constant value while LR decays to zero.

    This class anchors training with Focal Loss (per-pixel cross-entropy
    gradients + class weights) and adds the Tversky term to push Recall:

        total = tversky_weight * TverskyLoss
              + (1 - tversky_weight) * FocalLoss

    The Focal component prevents collapse; the Tversky component ensures
    FN are penalised more than FP once the model is stable.

    Args:
        tversky_alpha:  FP weight in Tversky denominator (lower → FP-tolerant).
        tversky_beta:   FN weight in Tversky denominator (higher → Recall push).
        tversky_weight: Fraction of the total loss assigned to Tversky [0, 1].
        focal_gamma:    Focusing exponent for the Focal component.
        class_weights:  Per-class weights tensor for the Focal component.
        ignore_index:   Label value excluded from both components (e.g. 255).
    """

    def __init__(
        self,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_weight: float = 0.5,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self._tversky = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            ignore_index=ignore_index,
        )
        self._focal = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            ignore_index=ignore_index,
        )
        self._tversky_weight = tversky_weight
        self._focal_weight = 1.0 - tversky_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self._tversky_weight * self._tversky(inputs, targets)
            + self._focal_weight * self._focal(inputs, targets)
        )

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
        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True

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
        scaler: torch.amp.GradScaler,
    ) -> None:
        best_val_loss = float("inf")
        # AMP disabled: this UNet has no BatchNorm, so FP16 activations in the
        # 1024-channel bottleneck can overflow during backward, producing
        # Inf gradients that clip_grad_norm converts to NaN (Inf * 0 = NaN),
        # corrupting model weights after just a few updates.  Pure FP32 is safe.
        use_amp = False
        total_batches = len(train_loader)
        log_every = 50  # log every 50 batches regardless of dataset size

        for epoch in range(num_epochs):
            self._state.current_epoch = epoch + 1
            epoch_start = time.monotonic()

            # --- train ---
            self._model.train()
            epoch_loss = 0.0
            batch_count = 0
            last_log_time = time.monotonic()

            for batch_idx, (inputs, labels, names) in enumerate(train_loader):
                fetch_done = time.monotonic()
                inputs = inputs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)

                # Skip batches with corrupted data (e.g. failed SSL download)
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    logger.warning(
                        "Epoch {}/{} batch {} — skipping: NaN/Inf in inputs ({})",
                        epoch + 1, num_epochs, batch_idx + 1, names[0],
                    )
                    last_log_time = time.monotonic()
                    continue

                # Skip batches where every pixel is the ignore class (can happen
                # after random rotation crops out all valid-label regions).
                if (labels == 255).all():
                    logger.warning(
                        "Epoch {}/{} batch {} — skipping: all pixels are ignore_index ({})",
                        epoch + 1, num_epochs, batch_idx + 1, names[0],
                    )
                    last_log_time = time.monotonic()
                    continue

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=self._device.type, enabled=use_amp):
                    outputs = self._model(inputs)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.warning(
                        "Epoch {}/{} batch {} — skipping: NaN/Inf in model outputs ({})",
                        epoch + 1, num_epochs, batch_idx + 1, names[0],
                    )
                    optimizer.zero_grad()
                    last_log_time = time.monotonic()
                    continue

                loss = criterion(outputs, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        "Epoch {}/{} batch {} — skipping: NaN/Inf loss ({})",
                        epoch + 1, num_epochs, batch_idx + 1, names[0],
                    )
                    optimizer.zero_grad()
                    last_log_time = time.monotonic()
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    max_norm=settings.grad_clip_max_norm,
                )
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                batch_count += 1
                gpu_done = time.monotonic()

                if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
                    elapsed_epoch = gpu_done - epoch_start
                    since_last = gpu_done - last_log_time
                    fetch_ms = (fetch_done - last_log_time) * 1000
                    gpu_ms = (gpu_done - fetch_done) * 1000
                    progress_pct = (batch_idx + 1) / total_batches * 100
                    sample_names = ", ".join(names[:2])
                    logger.info(
                        "Epoch {}/{} [{:.1f}%] batch {}/{} | loss: {:.4f} | "
                        "fetch: {:.0f}ms  gpu: {:.0f}ms  step: {:.1f}s | {}",
                        epoch + 1,
                        num_epochs,
                        progress_pct,
                        batch_idx + 1,
                        total_batches,
                        epoch_loss / batch_count,
                        fetch_ms,
                        gpu_ms,
                        since_last,
                        sample_names,
                    )
                    last_log_time = gpu_done

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
        # AMP disabled for the same reason as _train_loop: no BatchNorm means
        # FP16 activations overflow in the bottleneck, producing NaN outputs.
        # Inference has no backward pass so there is no speed benefit worth
        # the instability risk.
        self._model.eval()
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for inputs, labels, _names in val_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                outputs = self._model(inputs)
                loss = criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                total_loss += loss.item()
                batch_count += 1
        if batch_count == 0:
            logger.warning("Validation loop: all batches produced NaN/Inf loss — val_loss reported as inf")
            return float("inf")
        return total_loss / batch_count

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

        class_weights = torch.tensor(
            [settings.class_weight_background, settings.class_weight_anthill],
            device=self._device,
        )
        if settings.tversky_alpha > 0 and settings.tversky_beta > 0:
            criterion: nn.Module = CombinedTverskyFocalLoss(
                tversky_alpha=settings.tversky_alpha,
                tversky_beta=settings.tversky_beta,
                tversky_weight=settings.tversky_loss_weight,
                focal_gamma=settings.focal_loss_gamma if settings.focal_loss_gamma > 0 else 2.0,
                class_weights=class_weights,
                ignore_index=255,
            )
            logger.info(
                "Using Combined Tversky+Focal Loss "
                "(tversky_weight={}, alpha={}, beta={}, focal_gamma={}, class_weights=[{}, {}])",
                settings.tversky_loss_weight,
                settings.tversky_alpha,
                settings.tversky_beta,
                settings.focal_loss_gamma if settings.focal_loss_gamma > 0 else 2.0,
                settings.class_weight_background,
                settings.class_weight_anthill,
            )
        elif settings.focal_loss_gamma > 0:
            criterion = FocalLoss(
                gamma=settings.focal_loss_gamma,
                weight=class_weights,
                ignore_index=255,
            )
            logger.info(
                "Using Focal Loss (gamma={}, weights=[{}, {}])",
                settings.focal_loss_gamma,
                settings.class_weight_background,
                settings.class_weight_anthill,
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=settings.scheduler_factor,
            patience=settings.scheduler_patience,
        )
        # GradScaler disabled: AMP is off in _train_loop, so the scaler is a
        # passthrough kept only to avoid restructuring the training loop.
        scaler = torch.amp.GradScaler("cuda", enabled=False)

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
                scaler,
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

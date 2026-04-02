"""Centralized application configuration via Pydantic BaseSettings.

All tunable parameters live here as typed fields with documented defaults.
Values can be overridden in three ways (highest priority wins):

  1. CLI flag         --  python run_training.py --epochs 50
  2. Environment var  --  UNET_NUM_EPOCHS=50 python run_training.py
  3. .env file        --  set in the .env file next to this repo
  4. Default below    --  the value defined in this class

The ``settings`` singleton is constructed once at import time and must be
treated as **read-only** throughout the application lifecycle.

Where each group is consumed
----------------------------
* General / Google Drive / Model / Data / Pipeline
      → app/main.py, app/service/data_service.py
* Training — basic
      → app/service/training_service.py  (start_training, _train_loop)
* Training — loss & optimisation
      → app/service/training_service.py  (start_training)
* Training — augmentations
      → app/service/data_service.py      (create_train_transforms)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings  # pyright: ignore[reportMissingImports]


class Settings(BaseSettings):
    """Immutable application settings loaded from environment / .env."""

    # ── General ───────────────────────────────────────────────────────────────
    app_name: str = "UNet Segmentation Pipeline"
    debug: bool = False

    # ── Google Drive ──────────────────────────────────────────────────────────
    # Path to the service-account JSON key used to authenticate with Drive.
    google_credentials_path: str = "credentials.json"
    # ID of the root Drive folder that contains "treino/" and "validacao/".
    base_folder_id: str = "1slS6V7OWBaBny7v94K3Vx9eGHp7lph91"

    # ── Model ─────────────────────────────────────────────────────────────────
    # Path where the last-epoch weights are written at the end of training.
    model_save_path: str = "u_net.pth"
    # Path where the best val_loss checkpoint is saved during training.
    best_model_params_path: str = "best_model_params.pth"
    # Number of input image channels (3 = RGB).
    n_channels: int = 3
    # Number of output segmentation classes (2 = background + anthill).
    n_classes: int = 2

    # ── Training — basic ──────────────────────────────────────────────────────

    # batch_size: how many image-mask pairs are processed together in one
    # gradient update.
    #   ↑ larger  → smoother loss curve, more stable gradients, more VRAM
    #   ↓ smaller → noisier gradients (can help escape local minima), less VRAM
    #   Applied in: DataLoader(batch_size=...) inside data_service.py
    batch_size: int = 6

    # learning_rate: step size for the Adam optimiser.
    #   ↑ larger  → faster convergence but risk of overshooting / loss oscillation
    #   ↓ smaller → slower but stable; use if val_loss bounces between epochs
    #   Applied in: optim.Adam(lr=learning_rate) inside training_service.py
    learning_rate: float = 0.001

    # num_epochs: total passes over the full training dataset.
    #   ↑ larger  → more learning time; monitor val_loss to detect overfitting
    #   ↓ smaller → faster experiment cycle
    #   Applied in: _train_loop outer loop inside training_service.py
    num_epochs: int = 80

    # num_workers: subprocess workers that load batches from disk in parallel.
    #   0   → main process only (use for debugging DataLoader issues)
    #   1-4 → parallel I/O; recommended when data_mode="local"
    #   Applied in: DataLoader(num_workers=...) inside data_service.py
    num_workers: int = 4

    # device: compute backend.
    #   "auto" → CUDA if available, otherwise CPU
    #   "cuda" → force GPU (fails if no CUDA)
    #   "cpu"  → force CPU (very slow for training, useful for debugging)
    #   Applied in: TrainingService._resolve_device() inside training_service.py
    device: str = "cuda"

    # ── Training — loss & optimisation ────────────────────────────────────────

    # class_weight_background: CrossEntropyLoss weight for the background class.
    # Usually kept at 1.0 as the reference weight.
    #   Applied in: nn.CrossEntropyLoss(weight=[bg, anthill]) in training_service.py
    class_weight_background: float = 1.0

    # class_weight_anthill: CrossEntropyLoss weight for the anthill class.
    # The anthill class is rare (heavily imbalanced), so it needs a higher weight.
    #   ↑ larger  → model penalised more for missing anthill pixels
    #               → better recall, but may produce more false positives
    #   ↓ smaller → safer against false positives, but may miss small anthills
    #   Applied in: nn.CrossEntropyLoss(weight=[bg, anthill]) in training_service.py
    class_weight_anthill: float = 2.5

    # grad_clip_max_norm: maximum L2 norm allowed for the full gradient vector.
    # Prevents exploding gradients (critical for UNets without BatchNorm).
    #   ↑ larger  → allows bigger gradient steps; faster early phases
    #   ↓ smaller → more conservative updates; safer but potentially slower
    #   Applied in: clip_grad_norm_(params, max_norm=...) in training_service.py
    grad_clip_max_norm: float = 1.0

    # scheduler_factor: multiplier applied to the LR when a plateau is detected.
    # new_lr = current_lr * scheduler_factor
    #   ↑ closer to 1.0 (e.g. 0.9) → gentle reduction, stays near original LR
    #   ↓ closer to 0.0 (e.g. 0.1) → aggressive cut, effectively stops learning
    #   Applied in: ReduceLROnPlateau(factor=...) in training_service.py
    scheduler_factor: float = 0.5

    # scheduler_patience: epochs with no val_loss improvement before reducing LR.
    #   ↑ larger  → gives more time before reducing LR; good for noisy datasets
    #   ↓ smaller → reacts faster to plateaus; risk of reducing LR too early
    #   Applied in: ReduceLROnPlateau(patience=...) in training_service.py
    scheduler_patience: int = 6

    # ── Training — data augmentations ─────────────────────────────────────────
    # All augmentations are applied jointly to the image AND the mask so spatial
    # correspondence is preserved.  Applied in: create_train_transforms() in
    # data_service.py.  Augmentations do NOT affect the validation set.

    # aug_horizontal_flip: randomly mirror image left↔right with p=0.5.
    #   True  → strongly recommended for aerial imagery (no fixed orientation)
    aug_horizontal_flip: bool = True

    # aug_vertical_flip: randomly mirror image top↔bottom with p=0.5.
    #   True  → also recommended for aerial imagery
    aug_vertical_flip: bool = True

    # aug_rotation_degrees: max angle for RandomRotation in degrees.
    #   ↑ larger (→ 90) → more orientational variety; can push label pixels
    #                      outside the tile at large angles (handled by ignore)
    #   0               → disables rotation augmentation
    aug_rotation_degrees: int = 30

    # aug_color_jitter: apply random brightness / contrast / saturation shifts.
    # Does NOT affect the mask (applied to image only).
    #   True  → useful when images come from different flights, sensors, or seasons
    aug_color_jitter: bool = False

    # aug_color_jitter_brightness: max brightness deviation as fraction of original.
    #   0.0 → no change  |  0.5 → ±50%  |  keep below 0.3 for aerial imagery
    aug_color_jitter_brightness: float = 0.2

    # aug_color_jitter_contrast: max contrast deviation as fraction of original.
    #   0.0 → no change  |  0.5 → ±50%  |  keep below 0.3 for aerial imagery
    aug_color_jitter_contrast: float = 0.2

    # aug_color_jitter_saturation: max saturation deviation as fraction of original.
    #   0.0 → no change  |  keep low (0.1) to avoid unnatural colours
    aug_color_jitter_saturation: float = 0.1

    # ── Data ──────────────────────────────────────────────────────────────────
    # data_mode: where images come from.
    #   "local"  → reads from local_data_dir on disk (fast, no internet needed)
    #   "online" → streams from Google Drive on-demand (no local storage needed)
    data_mode: str = "local"
    local_data_dir: str = "data"
    # Subdirectory paths relative to local_data_dir (local mode only).
    train_rgb_subdir: str = "training/rgb/rgb"
    train_labels_subdir: str = "training/labels/labels"
    val_rgb_subdir: str = "validation/rgb/rgb"
    val_labels_subdir: str = "validation/labels/labels"
    # preload_dataset: load ALL images into RAM at startup.
    #   True  → eliminates disk I/O during training; only viable if dataset fits in RAM
    #   False → reads from disk per batch (default, works for any dataset size)
    preload_dataset: bool = False

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # pipeline_mode: what to run when the container starts.
    #   "train"    → train model, save weights to model_save_path
    #   "validate" → load weights, run metrics, save detections to validation_output_dir
    pipeline_mode: str = "train"
    validation_output_dir: str = "validation_results"
    # anthill_save_threshold: minimum % of pixels predicted as anthill to save
    # the image to validation_output_dir.
    #   ↑ larger  → only saves tiles with large anthill regions (fewer, more confident)
    #   ↓ smaller → saves tiles with even tiny anthill detections (more, noisier)
    anthill_save_threshold: float = 0.1

    model_config = {"env_file": ".env", "env_prefix": "UNET_"}


settings = Settings()

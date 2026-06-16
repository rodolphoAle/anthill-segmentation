"""Centralized application configuration via Pydantic BaseSettings.

Override priority (highest wins):
  1. CLI flag  →  2. Environment variable (UNET_*)  →  3. .env file  →  4. Default

The ``settings`` singleton is constructed once at import time and is read-only.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings  # pyright: ignore[reportMissingImports]


class Settings(BaseSettings):
    """Immutable application settings loaded from environment / .env."""

    #  General 
    app_name: str = "UNet Segmentation Pipeline"
    debug: bool = False

    #  Google Drive  (only used when data_mode="online"; set via UNET_BASE_FOLDER_ID)
    google_credentials_path: str = "credentials.json"
    base_folder_id: str = ""

    #  Model 
    model_save_path: str = "u_net.pth"
    best_model_params_path: str = "best_model_params.pth"
    n_channels: int = 3
    n_classes: int = 2

    #  Training — basic 
    batch_size: int = 2
    learning_rate: float = 0.001
    num_epochs: int = 20
    num_workers: int = 4
    device: str = "cuda"  # "auto" | "cuda" | "cpu"

    #  Training — loss & optimisation 
    class_weight_background: float = 1.0
    class_weight_anthill: float = 6.0

    # Focal Loss: gamma=0 disables it.
    focal_loss_gamma: float = 2.0

    # Tversky Loss: TL = 1 - TP / (TP + α·FP + β·FN).
    # α=0.3, β=0.7 penalises FN more → pushes Recall.
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_loss_weight: float = 0.5

    # Lovász Hinge: direct IoU surrogate.
    # Combined: w_T·Tversky + w_L·Lovász + (1-w_T-w_L)·Focal.
    lovasz_loss_weight: float = 0.3

    grad_clip_max_norm: float = 1.0

    # ReduceLROnPlateau (ignored when use_cosine_scheduler=True).
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5

    # CosineAnnealingLR (immune to noisy val_loss).
    use_cosine_scheduler: bool = True
    cosine_eta_min: float = 1e-6

    #  Training — augmentations 
    # Spatial (applied jointly to image + mask).
    aug_horizontal_flip: bool = True
    aug_vertical_flip: bool = True
    aug_rotation_degrees: int = 15  # 0 = off
    aug_random_rotate_90: bool = False

    # Elastic deformation.
    aug_elastic_transform: bool = True
    aug_elastic_alpha: float = 25.0
    aug_elastic_sigma: float = 4.0

    # Colour (image only, does not affect mask).
    aug_color_jitter: bool = True
    aug_color_jitter_brightness: float = 0.2
    aug_color_jitter_contrast: float = 0.2
    aug_color_jitter_saturation: float = 0.1

    # Copy-paste: paste anthill regions from positive tiles onto negatives.
    aug_copy_paste: bool = False
    aug_copy_paste_prob: float = 0.4

    # Anthill duplication: rotated copies within the same tile.
    aug_anthill_duplicate: bool = True
    aug_anthill_duplicate_prob: float = 0.7
    aug_anthill_duplicate_max_copies: int = 2

    #  Data 
    data_mode: str = "local"  # "local" | "online"
    local_data_dir: str = "data"
    train_rgb_subdir: str = "training/rgb/rgb"
    train_labels_subdir: str = "training/labels/labels"
    val_rgb_subdir: str = "validation/rgb/rgb"
    val_labels_subdir: str = "validation/labels/labels"
    preload_dataset: bool = False

    # Drop tiles with >70% ignore pixels (border padding with no labels).
    max_ignore_pixel_pct: float = 0.7

    #  Pipeline 
    pipeline_mode: str = "train"  # "train" | "validate"
    validation_output_dir: str = "output/validation_results"
    anthill_save_threshold: float = 0.0
    anthill_confidence_threshold: float = 0.40

    # Connected-component region filter (post confidence threshold).
    min_anthill_region_px: int = 5
    max_anthill_region_px: int = 5000
    use_region_filter: bool = True

    model_config = {"env_file": ".env", "env_prefix": "UNET_"}


settings = Settings()
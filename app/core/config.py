"""Centralized application configuration via Pydantic BaseSettings.

All sensitive data and tuneable parameters are read from environment
variables (optionally backed by a ``.env`` file).  The ``settings``
singleton must be treated as **read-only** throughout the application
lifecycle.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings # pyright: ignore[reportMissingImports]


class Settings(BaseSettings):
    """Immutable application settings loaded from environment / .env."""

    #  General 
    app_name: str = "UNet Segmentation Pipeline"
    debug: bool = False

    #  Google Drive 
    google_credentials_path: str = "credentials.json"
    base_folder_id: str = "1slS6V7OWBaBny7v94K3Vx9eGHp7lph91"

    #  Model 
    model_save_path: str = "u_net.pth"
    best_model_params_path: str = "best_model_params.pth"
    n_channels: int = 3
    n_classes: int = 2

    #  Training hyper-parameters 
    batch_size: int = 4
    learning_rate: float = 0.001
    num_epochs: int = 200
    num_workers: int = 2

    #  Data 
    data_mode: str = "online"  # "online" | "local"
    local_data_dir: str = "data"

    #  Pipeline mode 
    # "train"    → stream training data, train model, save weights
    # "validate" → load saved weights, stream validation data, report metrics
    pipeline_mode: str = "train"
    validation_output_dir: str = "validation_results"
    # Minimum % of pixels predicted as anthill to save the image to disk.
    # E.g. 20.0 means at least 20% of the image must be anthill.
    anthill_save_threshold: float = 20.0

    model_config = {"env_file": ".env", "env_prefix": "UNET_"}


settings = Settings()

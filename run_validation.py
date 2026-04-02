"""Entry point for model validation.

Usage (inside the container)::

    python run_validation.py
    python run_validation.py --device cpu --output-dir results_v2
    python run_validation.py --threshold 5.0 --data-mode local

All flags are optional and override the corresponding UNET_* env-var / .env
value for this run only — no container restart required.
"""

import argparse
import asyncio
import os


def _apply_cli_overrides() -> None:
    """Parse CLI flags and push any provided values into os.environ.

    Must run *before* importing app.main so that Pydantic BaseSettings
    picks them up at construction time.
    """
    parser = argparse.ArgumentParser(
        description="Run validation on the trained UNet model."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        help="Compute device (UNET_DEVICE)")
    parser.add_argument("--data-mode", choices=["local", "online"],
                        help="Dataset source (UNET_DATA_MODE)")
    parser.add_argument("--output-dir", metavar="DIR",
                        help="Directory for saved results (UNET_VALIDATION_OUTPUT_DIR)")
    parser.add_argument("--threshold", type=float, metavar="PCT",
                        help="Min anthill %% to save image (UNET_ANTHILL_SAVE_THRESHOLD)")
    parser.add_argument("--weights", metavar="PATH",
                        help="Model weights file to load (UNET_MODEL_SAVE_PATH)")
    args = parser.parse_args()

    mapping = {
        "device": ("UNET_DEVICE", str),
        "data_mode": ("UNET_DATA_MODE", str),
        "output_dir": ("UNET_VALIDATION_OUTPUT_DIR", str),
        "threshold": ("UNET_ANTHILL_SAVE_THRESHOLD", str),
        "weights": ("UNET_MODEL_SAVE_PATH", str),
    }
    for attr, (env_key, cast) in mapping.items():
        value = getattr(args, attr)
        if value is not None:
            os.environ[env_key] = cast(value)


_apply_cli_overrides()
os.environ.setdefault("UNET_PIPELINE_MODE", "validate")

from app.main import main  # noqa: E402

asyncio.run(main())

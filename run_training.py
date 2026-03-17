"""Entry point for model training.

Usage (inside the container):
    python run_training.py
"""

import asyncio
import os

os.environ.setdefault("UNET_PIPELINE_MODE", "train")

from app.main import main  # noqa: E402

asyncio.run(main())

"""Entry point for model validation.

Usage (inside the container):
    python run_validation.py
"""

import asyncio
import os

os.environ.setdefault("UNET_PIPELINE_MODE", "validate")

from app.main import main  # noqa: E402

asyncio.run(main())

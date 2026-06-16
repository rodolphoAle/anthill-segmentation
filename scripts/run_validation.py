"""Validation CLI — parse args, apply env overrides, run validation pipeline.

Can be invoked directly::

    python scripts/run_validation.py --device cpu --output-dir results_v2

Or via the unified CLI::

    python -m app.main validate --device cpu --output-dir results_v2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_parser(subparsers: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    """Build the validation argument parser.

    If *subparsers* is provided, the parser is registered as a sub-command
    named ``validate``.  Otherwise a standalone parser is returned.
    """
    kwargs: dict = dict(
        description="Run validation on the trained UNet model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    if subparsers is not None:
        parser = subparsers.add_parser("validate", **kwargs)
    else:
        parser = argparse.ArgumentParser(**kwargs)

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        help="Compute device (UNET_DEVICE)")
    parser.add_argument("--data-mode", choices=["local", "online"],
                        help="Dataset source (UNET_DATA_MODE)")
    parser.add_argument("--output-dir", metavar="DIR",
                        help="Directory for saved results (UNET_VALIDATION_OUTPUT_DIR)")
    parser.add_argument("--threshold", type=float, metavar="PCT",
                        help="Min anthill %% to save image (UNET_ANTHILL_SAVE_THRESHOLD)")
    parser.add_argument("--weights", metavar="PATH",
                        help="Model weights file to load (UNET_BEST_MODEL_PARAMS_PATH)")

    region_filter = parser.add_mutually_exclusive_group()
    region_filter.add_argument(
        "--region-filter",
        dest="region_filter",
        action="store_true",
        default=None,
        help="Enable connected-component size filter (UNET_USE_REGION_FILTER=true)",
    )
    region_filter.add_argument(
        "--no-region-filter",
        dest="region_filter",
        action="store_false",
        help="Disable connected-component size filter (UNET_USE_REGION_FILTER=false)",
    )

    return parser


def apply_overrides(args: argparse.Namespace) -> None:
    """Push parsed CLI values into ``os.environ`` for Pydantic to pick up."""
    mapping = {
        "device":     "UNET_DEVICE",
        "data_mode":  "UNET_DATA_MODE",
        "output_dir": "UNET_VALIDATION_OUTPUT_DIR",
        "threshold":  "UNET_ANTHILL_SAVE_THRESHOLD",
        "weights":    "UNET_BEST_MODEL_PARAMS_PATH",
    }
    for attr, env_key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env_key] = str(value)

    if getattr(args, "region_filter", None) is not None:
        os.environ["UNET_USE_REGION_FILTER"] = str(args.region_filter).lower()

    os.environ.setdefault("UNET_PIPELINE_MODE", "validate")


def run(args: argparse.Namespace) -> None:
    """Execute validation with the given arguments."""
    apply_overrides(args)

    from app.main import pipeline_main  # noqa: E402

    asyncio.run(pipeline_main())


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)

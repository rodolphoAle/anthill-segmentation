"""Structured logging configuration using *loguru*.

Call ``setup_logging`` once at application startup (e.g. inside the
FastAPI lifespan handler).
"""

from __future__ import annotations

import logging
import sys

from loguru import logger # type: ignore


def setup_logging(debug: bool = False) -> None:
    """Configure loguru sinks and intercept stdlib logging."""

    logger.remove()

    log_level = "DEBUG" if debug else "INFO"
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=log_format, level=log_level, colorize=True)
    logger.add(
        "logs/app.log",
        format=log_format,
        level=log_level,
        rotation="10 MB",
        retention="7 days",
    )

    # Intercept standard-library logging → loguru
    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = str(record.levelno)
            logger.opt(depth=6, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Silence noisy third-party loggers that spam DEBUG even in normal runs
    for _noisy in (
        "PIL",
        "PIL.PngImagePlugin",
        "googleapiclient",
        "googleapiclient.discovery",
        "googleapiclient.discovery_cache",
        "google_auth_httplib2",
        "urllib3",
        "urllib3.connectionpool",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

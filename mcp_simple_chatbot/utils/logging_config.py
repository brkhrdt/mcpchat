"""Logging configuration for the application."""

import logging


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.ERROR, format="%(asctime)s][CHAT ]%(levelname)s- %(message)s"
    )

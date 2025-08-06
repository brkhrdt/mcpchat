"""Logging configuration for the application."""

import logging


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.ERROR, format="%(asctime)s][CHAT ]%(levelname)s- %(message)s"
    )
    
    # Ensure all mcp_simple_chatbot loggers start at ERROR level
    for name in logging.root.manager.loggerDict:
        if name.startswith('mcp_simple_chatbot'):
            logging.getLogger(name).setLevel(logging.ERROR)

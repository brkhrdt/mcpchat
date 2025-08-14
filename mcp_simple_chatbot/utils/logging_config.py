"""Logging configuration for the application."""

import logging


def setup_logging() -> None:
    """Configure logging for the application."""
    # Set the root logger level to INFO or DEBUG to see more messages
    logging.basicConfig(
        level=logging.INFO, # Changed from logging.ERROR to logging.INFO
        format="%(asctime)s][CHAT ]%(levelname)s- %(message)s"
    )

    # Ensure all mcp_simple_chatbot loggers start at INFO or DEBUG level
    # You can set this to logging.DEBUG if you want to see all debug messages
    for name in logging.root.manager.loggerDict:
        if name.startswith("mcp_simple_chatbot"):
            logging.getLogger(name).setLevel(logging.INFO) # Changed from logging.ERROR to logging.INFO

"""Logging utility for the CustomGrid environment."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)

    # Always set level to DEBUG to ensure all messages can be captured by handlers
    logger.setLevel(logging.DEBUG)

    # Only add handler if it doesn't have one to avoid duplicate logs
    if not logger.handlers:
        print(f"Adding handlers to logger: {name}")
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (DEBUG and above)
        file_handler = logging.FileHandler("custom_grid_env.log", mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

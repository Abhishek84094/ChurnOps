"""
ChurnOps — Structured Logging
Consistent log format across all src modules.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger.

    Usage:
        from src.utils.logging import get_logger
        log = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Force UTF-8 on Windows to avoid cp1252 UnicodeEncodeError
        import io
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
            if hasattr(sys.stdout, "buffer") else sys.stdout
        handler = logging.StreamHandler(stream)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger

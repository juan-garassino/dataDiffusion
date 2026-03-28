"""Singleton logger for datadiffusion — one logger, one file per run."""

import logging
import os
from datetime import datetime

_configured = False


def setup_logging(level=logging.INFO, log_dir="logs", verbose=False):
    """Configure the 'datadiffusion' root logger once per process."""
    global _configured
    if _configured:
        return logging.getLogger("datadiffusion")

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"datadiffusion_{timestamp}.log")

    logger = logging.getLogger("datadiffusion")
    logger.setLevel(logging.DEBUG if verbose else level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    _configured = True
    logger.info("Logging initialised → %s", log_file)
    return logger


def get_logger(name: str = "datadiffusion"):
    """Return a child logger under the datadiffusion namespace."""
    return logging.getLogger(name)

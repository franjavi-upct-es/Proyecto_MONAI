"""Logging helpers. W&B is optional and only activated when WANDB_API_KEY is set."""

from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str = "lungseg") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def wandb_enabled() -> bool:
    return bool(os.environ.get("WANDB_API_KEY"))

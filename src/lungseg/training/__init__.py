"""Training stack: losses, schedulers, iteration-based trainer."""

from __future__ import annotations

from lungseg.training.losses import build_loss, deep_supervision_loss
from lungseg.training.schedulers import build_poly_scheduler, poly_lr
from lungseg.training.trainer import train_iters

__all__ = [
    "build_loss",
    "build_poly_scheduler",
    "deep_supervision_loss",
    "poly_lr",
    "train_iters",
]

"""Training stack: losses, schedulers, epoch-based SSL trainer."""

from __future__ import annotations

from lungseg.training.losses import build_loss
from lungseg.training.schedulers import build_cosine_warmup_scheduler, build_poly_scheduler, poly_lr
from lungseg.training.trainer import Trainer, train_iters

__all__ = [
    "Trainer",
    "build_cosine_warmup_scheduler",
    "build_loss",
    "build_poly_scheduler",
    "poly_lr",
    "train_iters",
]

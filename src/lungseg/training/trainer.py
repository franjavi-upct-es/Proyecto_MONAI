"""Iteration-based trainer (no epochs).

B4 will implement `train_iters(cfg, model, loaders)`:
  - AMP + GradScaler
  - gradient accumulation
  - polyLR step-based
  - sliding-window validation with DSC + HD95
  - early stopping by val_dice
  - best-model checkpointing into outputs/
  - optional W&B logging when WANDB_API_KEY is set
  - set_global_determinism(cfg.seed) at start, seed_worker on DataLoaders
"""

from __future__ import annotations

from omegaconf import DictConfig


def train_iters(cfg: DictConfig, model, loaders) -> None:
    raise NotImplementedError("B4 will implement train_iters.")

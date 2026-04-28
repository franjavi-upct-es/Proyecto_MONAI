"""Sliding-window inference wrapper. Filled in B4."""

from __future__ import annotations

import torch
from omegaconf import DictConfig


def predict_volume(model: torch.nn.Module, image: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    raise NotImplementedError("B4 will implement sliding-window inference.")

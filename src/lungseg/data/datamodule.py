"""DataLoader factory. Filled in B2."""

from __future__ import annotations

from omegaconf import DictConfig


def build_loaders(cfg: DictConfig, fold: int):
    raise NotImplementedError("B2 will implement build_loaders.")

"""LR schedulers. polyLR follows nnU-Net: lr * (1 - step/max_steps) ** exp."""

from __future__ import annotations


def poly_lr(step: int, max_steps: int, base_lr: float, exp: float = 0.9) -> float:
    raise NotImplementedError("B3 will implement poly_lr.")

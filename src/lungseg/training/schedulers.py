"""Planificadores de tasa de aprendizaje (LR). polyLR sigue a nnU-Net: lr * (1 - paso/pasos_máximos) ** exp."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def poly_lr(step: int, max_steps: int, base_lr: float, exp: float = 0.9) -> float:
    if max_steps <= 0:
        raise ValueError("pasos_máximos debe ser > 0")
    clamped = min(max(int(step), 0), int(max_steps))
    factor = (1.0 - clamped / float(max_steps)) ** float(exp)
    return float(base_lr) * factor


def build_poly_scheduler(optimizer: Optimizer, max_steps: int, exp: float = 0.9) -> LambdaLR:
    """Construye un planificador poly por pasos para los pasos del optimizador, no por épocas."""
    if max_steps <= 0:
        raise ValueError("pasos_máximos debe ser > 0")

    def lr_lambda(step: int) -> float:
        clamped = min(max(int(step), 0), int(max_steps))
        return (1.0 - clamped / float(max_steps)) ** float(exp)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

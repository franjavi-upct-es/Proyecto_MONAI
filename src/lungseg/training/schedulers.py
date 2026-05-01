"""Planificadores de tasa de aprendizaje (LR). polyLR sigue a nnU-Net: lr * (1 - paso/pasos_máximos) ** exp."""

from __future__ import annotations

import math

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


def build_cosine_warmup_scheduler(
    optimizer: Optimizer, max_steps: int, warmup_steps: int = 500
) -> LambdaLR:
    """Cosine Annealing con Warmup lineal inicial."""
    if max_steps <= 0:
        raise ValueError("max_steps debe ser > 0")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

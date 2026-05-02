# src/lungseg/training/losses.py
"""Focal Tversky loss para segmentación asimétrica de tumores pulmonares."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from omegaconf import DictConfig


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss penalizando más los falsos negativos.

    Tversky index: ``TI = TP / (TP + alpha*FP + beta*FN)``
    Loss:         ``(1 - TI) ** gamma`` promediado sobre clases foreground.

    Con ``alpha=0.3, beta=0.7`` los FN cuestan 7/3 veces más que los FP, lo
    que es apropiado para segmentación de tumores donde perder una lesión es
    peor que un falso positivo.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 4.0 / 3.0,
        to_onehot_y: bool = True,
        softmax: bool = True,
        include_background: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
    ) -> None:
        super().__init__()
        if alpha < 0.0 or beta < 0.0:
            raise ValueError("alpha y beta deben ser >= 0")
        if gamma <= 0.0:
            raise ValueError("gamma debe ser > 0")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.to_onehot_y = bool(to_onehot_y)
        self.softmax = bool(softmax)
        self.include_background = bool(include_background)
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def _prepare_target(self, target: torch.Tensor, n_classes: int) -> torch.Tensor:
        if not self.to_onehot_y:
            return target.to(torch.float32)
        if target.shape[1] == n_classes:
            return target.to(torch.float32)
        # `target` viene como (B, 1, ...) con índices enteros.
        long_target = target.long()
        if long_target.dim() < 2:
            raise ValueError("target debe tener al menos 2 dimensiones (B, 1, ...)")
        if long_target.shape[1] != 1:
            raise ValueError(
                f"to_onehot_y espera canal único en target, recibido {long_target.shape[1]}"
            )
        squeezed = long_target[:, 0]
        return functional.one_hot(squeezed, num_classes=n_classes).movedim(-1, 1).to(torch.float32)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() < 3:
            raise ValueError(f"pred debe tener forma (B, C, ...), recibido {tuple(pred.shape)}")
        n_classes = pred.shape[1]
        if self.softmax:
            pred = pred.float().softmax(dim=1)
        target = self._prepare_target(target, n_classes).to(pred.dtype)
        if pred.shape != target.shape:
            raise ValueError(
                f"shapes no coinciden: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
            )

        spatial_dims = tuple(range(2, pred.dim()))
        tp = (pred * target).sum(dim=spatial_dims)
        fp = (pred * (1.0 - target)).sum(dim=spatial_dims)
        fn = ((1.0 - pred) * target).sum(dim=spatial_dims)

        ti = (tp + self.smooth_nr) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth_dr
        )
        focal = (1.0 - ti).clamp(min=0.0).pow(self.gamma)  # (B, C)

        if not self.include_background and n_classes > 1:
            focal = focal[:, 1:]
        return focal.mean()


def build_loss(cfg: DictConfig | None = None) -> nn.Module:
    """Construye `FocalTverskyLoss` con hiperparámetros desde `cfg.model.loss`."""
    loss_cfg = {}
    if cfg is not None and "model" in cfg and "loss" in cfg.model:
        loss_cfg = cfg.model.loss

    return FocalTverskyLoss(
        alpha=float(loss_cfg.get("alpha", 0.3)),
        beta=float(loss_cfg.get("beta", 0.7)),
        gamma=float(loss_cfg.get("gamma", 4.0 / 3.0)),
        to_onehot_y=bool(loss_cfg.get("to_onehot_y", True)),
        softmax=bool(loss_cfg.get("softmax", True)),
        include_background=bool(loss_cfg.get("include_background", False)),
    )

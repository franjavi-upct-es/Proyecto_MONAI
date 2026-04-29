"""Funciones de pérdida y el envoltorio de supervisión profunda utilizado con DynUNet."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from monai.losses import DiceCELoss
from omegaconf import DictConfig


def build_loss(cfg: DictConfig) -> torch.nn.Module:
    """Construye la pérdida de segmentación a partir de ``cfg.model.loss``."""
    loss_cfg = cfg.model.loss if "model" in cfg and "loss" in cfg.model else cfg.loss
    name = str(loss_cfg.get("name", "dice_ce")).lower()
    if name != "dice_ce":
        raise ValueError(f"unknown loss.name={name!r}; solo se soporta 'dice_ce'")
    return DiceCELoss(
        include_background=bool(loss_cfg.get("include_background", False)),
        to_onehot_y=bool(loss_cfg.get("to_onehot_y", True)),
        sigmoid=bool(loss_cfg.get("sigmoid", False)),
        softmax=bool(loss_cfg.get("softmax", True)),
        lambda_dice=float(loss_cfg.get("lambda_dice", 1.0)),
        lambda_ce=float(loss_cfg.get("lambda_ce", 1.0)),
    )


def _normalized_weights(n_outputs: int, weights: list[float] | None) -> torch.Tensor:
    if n_outputs <= 0:
        raise ValueError("la pila de salida de supervisión profunda debe contener al menos una salida")
    raw = weights if weights is not None else [0.5 ** (i + 1) for i in range(n_outputs)]
    if len(raw) != n_outputs:
        raise ValueError(f"se esperaban {n_outputs} pesos de supervisión profunda, se obtuvieron {len(raw)}")
    tensor = torch.as_tensor(raw, dtype=torch.float32)
    if torch.any(tensor < 0) or float(tensor.sum()) <= 0.0:
        raise ValueError("los pesos de supervisión profunda deben ser no negativos y sumar > 0")
    return tensor / tensor.sum()


def deep_supervision_loss(
    out_stacked: torch.Tensor,
    target: torch.Tensor,
    base_loss,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Aplica ``base_loss`` sobre una pila de supervisión profunda de DynUNet.

    DynUNet devuelve ``(B, n_ds, C, X, Y, Z)``. Las salidas de menor resolución se
    comparan con etiquetas redimensionadas mediante el vecino más cercano, y luego se combinan con
    pesos geométricos normalizados.
    """
    if out_stacked.dim() == target.dim() + 1:
        n_outputs = int(out_stacked.shape[1])
        norm_weights = _normalized_weights(n_outputs, weights).to(
            device=out_stacked.device, dtype=out_stacked.dtype
        )
        total = out_stacked.new_tensor(0.0)
        for idx in range(n_outputs):
            pred = out_stacked[:, idx]
            label = target
            if tuple(label.shape[2:]) != tuple(pred.shape[2:]):
                label = functional.interpolate(label.float(), size=pred.shape[2:], mode="nearest")
                if target.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                    label = label.to(dtype=target.dtype)
            total = total + norm_weights[idx] * base_loss(pred, label)
        return total
    return base_loss(out_stacked, target)

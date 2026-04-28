"""Loss functions and the deep-supervision wrapper used with DynUNet."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from monai.losses import DiceCELoss
from omegaconf import DictConfig


def build_loss(cfg: DictConfig) -> torch.nn.Module:
    """Build the segmentation loss from ``cfg.model.loss``."""
    loss_cfg = cfg.model.loss if "model" in cfg and "loss" in cfg.model else cfg.loss
    name = str(loss_cfg.get("name", "dice_ce")).lower()
    if name != "dice_ce":
        raise ValueError(f"unknown loss.name={name!r}; only 'dice_ce' is supported")
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
        raise ValueError("deep supervision output stack must contain at least one output")
    raw = weights if weights is not None else [0.5 ** (i + 1) for i in range(n_outputs)]
    if len(raw) != n_outputs:
        raise ValueError(f"expected {n_outputs} deep-supervision weights, got {len(raw)}")
    tensor = torch.as_tensor(raw, dtype=torch.float32)
    if torch.any(tensor < 0) or float(tensor.sum()) <= 0.0:
        raise ValueError("deep-supervision weights must be non-negative and sum to > 0")
    return tensor / tensor.sum()


def deep_supervision_loss(
    out_stacked: torch.Tensor,
    target: torch.Tensor,
    base_loss,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Apply ``base_loss`` over a DynUNet deep-supervision stack.

    DynUNet returns ``(B, n_ds, C, X, Y, Z)``. Lower-resolution outputs are
    compared against nearest-neighbor resized labels, then combined with
    normalized geometric weights.
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

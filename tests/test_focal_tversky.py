"""Tests para `FocalTverskyLoss`."""

from __future__ import annotations

import torch

from lungseg.training.losses import FocalTverskyLoss


def _logits_from_target(target: torch.Tensor) -> torch.Tensor:
    """Convierte target (B, 1, ...) en logits saturados que predicen perfecto."""
    n_classes = 2
    logits = torch.full((target.shape[0], n_classes, *target.shape[2:]), -8.0)
    long_target = target[:, 0]
    logits[:, 0][long_target == 0] = 8.0
    logits[:, 1][long_target == 1] = 8.0
    return logits


def test_perfect_prediction_loss_near_zero() -> None:
    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4.0 / 3.0)
    target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    target[:, :, 2:6, 2:6, 2:6] = 1
    logits = _logits_from_target(target)
    assert float(loss_fn(logits, target)) < 0.05


def test_completely_wrong_loss_close_to_one() -> None:
    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4.0 / 3.0)
    target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    target[:, :, 2:6, 2:6, 2:6] = 1
    inverse_target = 1 - target
    logits = _logits_from_target(inverse_target)
    assert float(loss_fn(logits, target)) > 0.9


def test_fn_penalized_more_than_fp() -> None:
    """Con alpha=0.3 (FP) y beta=0.7 (FN), perder un voxel cuesta más que añadir uno espurio."""
    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0)
    # FP-only: predigo foreground en un voxel donde no lo hay.
    target_fp = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
    target_fp[:, :, 1:3, 1:3, 1:3] = 1
    pred_fp = target_fp.clone()
    pred_fp[:, :, 0, 0, 0] = 1  # un FP extra
    logits_fp = _logits_from_target(pred_fp)

    # FN-only: predigo background donde había foreground.
    pred_fn = target_fp.clone()
    pred_fn[:, :, 1, 1, 1] = 0  # un FN
    logits_fn = _logits_from_target(pred_fn)

    loss_fp = float(loss_fn(logits_fp, target_fp))
    loss_fn_only = float(loss_fn(logits_fn, target_fp))
    assert loss_fp < loss_fn_only, (
        f"Esperado loss(FP)={loss_fp:.4f} < loss(FN)={loss_fn_only:.4f} "
        "para alpha=0.3, beta=0.7"
    )

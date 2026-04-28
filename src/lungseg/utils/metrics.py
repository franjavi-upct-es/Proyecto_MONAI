"""Metric wrappers used by trainer and ablation analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from scipy import ndimage


def _to_binary_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim >= 5 and arr.shape[1] > 1:
        arr = np.argmax(arr, axis=1, keepdims=True)
    elif arr.ndim >= 5 and arr.shape[1] == 1:
        arr = arr > 0.5
    elif arr.ndim == 4:
        arr = arr[:, None] > 0.5
    return np.asarray(arr).astype(bool)


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    eroded = ndimage.binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def _hd95_one(pred: np.ndarray, label: np.ndarray, spacing: Sequence[float] | None) -> float:
    if not pred.any() and not label.any():
        return 0.0
    if not pred.any() or not label.any():
        return float("nan")
    pred_surface = _surface(pred)
    label_surface = _surface(label)
    sampling = None if spacing is None else tuple(float(s) for s in spacing)
    pred_to_label = ndimage.distance_transform_edt(~label_surface, sampling=sampling)[pred_surface]
    label_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=sampling)[label_surface]
    distances = np.concatenate([pred_to_label, label_to_pred])
    return float(np.percentile(distances, 95))


def compute_segmentation_metrics(
    prediction: torch.Tensor | np.ndarray,
    label: torch.Tensor | np.ndarray,
    spacing: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute foreground Dice and HD95 for binary lung-tumor masks.

    ``prediction`` may be logits/probabilities with two channels, a one-channel
    probability map, or an already-binary mask. ``label`` is expected to be a
    one-channel binary mask.
    """
    pred_bin = _to_binary_array(prediction)
    label_bin = _to_binary_array(label)
    if pred_bin.shape != label_bin.shape:
        raise ValueError(f"prediction/label shape mismatch: {pred_bin.shape} != {label_bin.shape}")

    dice_scores: list[float] = []
    hd95_scores: list[float] = []
    for pred_one, label_one in zip(pred_bin[:, 0], label_bin[:, 0], strict=True):
        intersection = float(np.logical_and(pred_one, label_one).sum())
        denom = float(pred_one.sum() + label_one.sum())
        dice_scores.append(1.0 if denom == 0.0 else 2.0 * intersection / denom)
        hd95_scores.append(_hd95_one(pred_one, label_one, spacing=spacing))

    hd95_arr = np.asarray(hd95_scores, dtype=np.float64)
    hd95 = float(np.nanmean(hd95_arr)) if not np.isnan(hd95_arr).all() else float("nan")
    return {"dice": float(np.mean(dice_scores)), "hd95": hd95}

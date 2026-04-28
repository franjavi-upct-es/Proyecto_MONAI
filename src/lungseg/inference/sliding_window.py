"""Sliding-window inference wrapper."""

from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference
from omegaconf import DictConfig


def predict_volume(model: torch.nn.Module, image: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Run MONAI sliding-window inference and normalize model output shape.

    DynUNet with deep supervision returns ``(B, n_ds, C, ...)`` for each
    window. Validation and inference always consume the final-resolution
    prediction, ``out[:, 0]``.
    """

    def _predictor(window: torch.Tensor) -> torch.Tensor:
        out = model(window)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, torch.Tensor) and out.dim() == window.dim() + 1:
            out = out[:, 0]
        return out

    inference_cfg = cfg.training.get("inference", {})
    return sliding_window_inference(
        inputs=image,
        roi_size=tuple(int(v) for v in cfg.training.patch_size),
        sw_batch_size=int(inference_cfg.get("sw_batch_size", 1)),
        predictor=_predictor,
        overlap=float(inference_cfg.get("overlap", 0.25)),
        mode=str(inference_cfg.get("mode", "gaussian")),
        padding_mode=str(inference_cfg.get("padding_mode", "constant")),
    )

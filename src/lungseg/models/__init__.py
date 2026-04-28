"""Model factories used by the segmentation and classification stacks."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from lungseg.models.dynunet import build_dynunet
from lungseg.models.segresnet import build_segresnet
from lungseg.models.unet_baseline import build_unet

_BUILDERS = {
    "segresnet": build_segresnet,
    "dynunet": build_dynunet,
    "unet": build_unet,
    "unet_baseline": build_unet,
}


def build_model(cfg: DictConfig) -> torch.nn.Module:
    """Build the configured segmentation model."""
    model_cfg = cfg.model if "model" in cfg else cfg
    name = str(model_cfg.get("name", "segresnet")).lower()
    try:
        return _BUILDERS[name](cfg)
    except KeyError as exc:
        raise ValueError(f"unknown model.name={name!r}; expected {sorted(_BUILDERS)}") from exc


__all__ = [
    "build_dynunet",
    "build_model",
    "build_segresnet",
    "build_unet",
]

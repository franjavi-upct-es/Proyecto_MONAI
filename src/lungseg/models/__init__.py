"""Fábricas de modelos de producción: ViT 2.5D."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from lungseg.models.vit_25d import ViT25D


def build_vit_25d(cfg: DictConfig) -> ViT25D:
    """Construye ViT25D desde la configuración."""
    model_cfg = cfg.model if "model" in cfg else cfg
    return ViT25D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 2)),
        encoder_name=str(model_cfg.get("encoder_name", "vit_base_patch16_224")),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )


_BUILDERS = {
    "vit_25d": build_vit_25d,
}


def build_model(cfg: DictConfig) -> torch.nn.Module:
    """Construye el modelo de segmentación configurado."""
    model_cfg = cfg.model if "model" in cfg else cfg
    name = str(model_cfg.get("name", "vit_25d")).lower()
    try:
        return _BUILDERS[name](cfg)
    except KeyError as exc:
        raise ValueError(f"unknown model.name={name!r}; expected {list(_BUILDERS.keys())}") from exc


__all__ = [
    "build_model",
    "build_vit_25d",
]

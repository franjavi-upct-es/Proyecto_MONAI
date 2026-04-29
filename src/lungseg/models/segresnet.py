"""Fábrica de SegResNet 3D (modelo por defecto en este proyecto)."""

from __future__ import annotations

from monai.networks.nets import SegResNet
from omegaconf import DictConfig


def build_segresnet(cfg: DictConfig):
    """Construye la SegResNet de MONAI descrita por ``cfg.model``.

    La SegResNet de MONAI 1.5.x no admite supervisión profunda. Mantener la
    guardia aquí hace que las derivaciones accidentales de la configuración fallen pronto en lugar de cambiar
    silenciosamente el contrato de entrenamiento.
    """
    model_cfg = cfg.model if "model" in cfg else cfg
    if bool(model_cfg.get("deep_supervision", False)):
        raise ValueError("SegResNet does not support deep_supervision in MONAI 1.5.x")

    return SegResNet(
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 2)),
        init_filters=int(model_cfg.get("init_filters", 16)),
        blocks_down=tuple(model_cfg.get("blocks_down", [1, 2, 2, 4])),
        blocks_up=tuple(model_cfg.get("blocks_up", [1, 1, 1])),
        dropout_prob=float(model_cfg.get("dropout_prob", 0.0)),
        norm=str(model_cfg.get("norm", "instance")).upper(),
        act=str(model_cfg.get("act", "relu")).upper(),
    )

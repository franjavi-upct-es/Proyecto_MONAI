"""Línea base de UNet 3D estándar (Vanilla)."""

from __future__ import annotations

from monai.networks.nets import UNet
from omegaconf import DictConfig


def build_unet(cfg: DictConfig):
    model_cfg = cfg.model if "model" in cfg else cfg
    channels = tuple(model_cfg.get("channels", [16, 32, 64, 128, 256]))
    strides = tuple(model_cfg.get("strides", [2, 2, 2, 2]))
    return UNet(
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 2)),
        channels=channels,
        strides=strides,
        num_res_units=int(model_cfg.get("num_res_units", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        norm=str(model_cfg.get("norm", "instance")).upper(),
        act=str(model_cfg.get("act", "prelu")).upper(),
    )

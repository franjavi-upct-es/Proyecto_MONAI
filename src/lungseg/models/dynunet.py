"""DynUNet 3D factory (optional model with deep-supervision support)."""

from __future__ import annotations

from monai.networks.nets import DynUNet
from omegaconf import DictConfig


def build_dynunet(cfg: DictConfig):
    """Build the MONAI DynUNet described by ``cfg.model``.

    With ``deep_supervision=True`` the forward pass returns
    ``(B, n_ds, C, ...)`` and the final-resolution prediction is ``out[:, 0]``.
    """
    model_cfg = cfg.model if "model" in cfg else cfg
    filters = model_cfg.get("filters", None)
    dropout = model_cfg.get("dropout", None)
    return DynUNet(
        spatial_dims=int(model_cfg.get("spatial_dims", 3)),
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 2)),
        kernel_size=[tuple(k) for k in model_cfg.kernel_size],
        strides=[tuple(s) for s in model_cfg.strides],
        upsample_kernel_size=[tuple(k) for k in model_cfg.upsample_kernel_size],
        filters=None if filters is None else list(filters),
        dropout=dropout,
        norm_name=str(model_cfg.get("norm_name", "instance")).upper(),
        deep_supervision=bool(model_cfg.get("deep_supervision", False)),
        deep_supr_num=int(model_cfg.get("deep_supr_num", 1)),
        res_block=bool(model_cfg.get("res_block", True)),
    )

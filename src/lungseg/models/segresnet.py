"""SegResNet 3D factory (default model in this project).

Filled in B3. Note: MONAI's SegResNet does NOT support deep_supervision;
only DynUNet does. Don't try to enable it here.
"""

from __future__ import annotations

from omegaconf import DictConfig


def build_segresnet(cfg: DictConfig):
    raise NotImplementedError("B3 will implement build_segresnet.")

"""DynUNet 3D factory (optional model with deep_supervision support).

Filled in B3. With deep_supervision=True the network returns a stacked tensor
of shape (B, n_ds, C, ...); index [:, 0] is the final-resolution prediction.
"""

from __future__ import annotations

from omegaconf import DictConfig


def build_dynunet(cfg: DictConfig):
    raise NotImplementedError("B3 will implement build_dynunet.")

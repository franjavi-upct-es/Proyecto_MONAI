"""Loss functions and the deep-supervision wrapper used with DynUNet.

B3 will implement `deep_supervision_loss(out_stacked, target, base_loss, weights)`
where `weights` are normalized as [0.5, 0.25, 0.125, ...] over the depth
axis. The base loss defaults to
`DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)`.
"""

from __future__ import annotations

import torch


def deep_supervision_loss(
    out_stacked: torch.Tensor,
    target: torch.Tensor,
    base_loss,
    weights: list[float] | None = None,
) -> torch.Tensor:
    raise NotImplementedError("B3 will implement deep_supervision_loss.")

"""Bottleneck hook that turns a trained DynUNet/SegResNet into a 320-dim
deep-feature extractor used by Phase 5 (B5).
"""

from __future__ import annotations

import torch


def extract_bottleneck_features(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("B5 will implement the bottleneck hook.")

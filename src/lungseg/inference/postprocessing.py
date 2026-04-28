"""Postprocessing: small-component removal and optional lung-mask gating."""

from __future__ import annotations

import numpy as np


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    raise NotImplementedError("B4 will implement remove_small_components.")

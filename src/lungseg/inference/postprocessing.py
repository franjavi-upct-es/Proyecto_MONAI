"""Postprocessing: small-component removal and optional lung-mask gating."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than ``min_size`` voxels."""
    if min_size <= 0:
        return mask
    binary = np.asarray(mask).astype(bool)
    labeled, n_components = ndimage.label(binary)
    if n_components == 0:
        return np.zeros_like(mask)
    counts = np.bincount(labeled.ravel())
    keep = counts >= int(min_size)
    keep[0] = False
    return keep[labeled].astype(mask.dtype)

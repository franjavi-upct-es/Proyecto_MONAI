"""Single-feature sanity baseline: classify on tumor VoxelVolume only.

Phase 5 acceptance criterion: report this baseline alongside the full radiomic
model so the gain attributable to texture/shape features is measurable.
"""

from __future__ import annotations

import numpy as np


def evaluate_size_only(volumes: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    raise NotImplementedError("B5 will implement evaluate_size_only.")

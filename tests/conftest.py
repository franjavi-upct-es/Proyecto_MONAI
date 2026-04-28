"""Shared pytest fixtures.

The synthetic-blob fixture is the canonical 3D test volume used across
B2 (transforms), B3 (loss), and B4 (inference shape). It encodes a
sphere of HU=50 inside a cube of HU=-700 so that CT-aware transforms can
exercise foreground sampling and intensity windowing on a known input.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_blob() -> dict[str, np.ndarray]:
    shape = (200, 200, 200)
    image = np.full(shape, fill_value=-700.0, dtype=np.float32)
    label = np.zeros(shape, dtype=np.uint8)

    cz, cy, cx = (s // 2 for s in shape)
    radius = 20
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    image[mask] = 50.0
    label[mask] = 1
    return {"image": image, "label": label}

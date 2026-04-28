"""PyRadiomics wrapper. Resamples to 1mm³, binWidth=25, ImageTypes
[Original, LoG sigma=1,2,3, Wavelet]. Filled in B5.
"""

from __future__ import annotations

from pathlib import Path


def extract(image_path: Path, mask_path: Path) -> dict[str, float]:
    raise NotImplementedError("B5 will implement extract.")

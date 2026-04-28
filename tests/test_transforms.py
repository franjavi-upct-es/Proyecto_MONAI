"""Transforms tests. Real assertions land in B2 (HU clip, CropForegroundd,
RandCropByPosNegLabeld with pos=2 producing >0 voxels >60% of the time).
"""

from __future__ import annotations


def test_placeholder() -> None:
    assert True

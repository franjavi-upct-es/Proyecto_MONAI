"""Tests para `MaskNonLungVoxelsd`."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.data import MetaTensor

from lungseg.data.transforms import LUNG_MASK_PATH_KEY, MaskNonLungVoxelsd


def _make_volume(tmp_path: Path) -> tuple[MetaTensor, np.ndarray, Path]:
    shape = (16, 16, 16)
    image = np.full(shape, fill_value=50.0, dtype=np.float32)
    affine = np.eye(4, dtype=np.float64)

    mask = np.zeros(shape, dtype=np.uint8)
    # Lung: cube centered.
    mask[4:12, 4:12, 4:12] = 1

    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), str(mask_path))

    image_meta = MetaTensor(
        torch.from_numpy(image)[None],  # (1, H, W, D)
        affine=torch.as_tensor(affine),
    )
    return image_meta, mask, mask_path


def test_mask_replaces_non_lung_voxels(tmp_path: Path) -> None:
    image_meta, mask, mask_path = _make_volume(tmp_path)
    transform = MaskNonLungVoxelsd(keys=("image",), fill_value=-1024.0)
    out = transform({"image": image_meta, LUNG_MASK_PATH_KEY: str(mask_path)})

    arr = out["image"].as_tensor().numpy()[0]
    # Voxels dentro de la máscara conservan su valor original.
    assert np.allclose(arr[4:12, 4:12, 4:12], 50.0)
    # Voxels fuera quedan a fill_value.
    assert np.allclose(arr[mask == 0], -1024.0)


def test_mask_inside_voxels_unchanged(tmp_path: Path) -> None:
    image_meta, mask, mask_path = _make_volume(tmp_path)
    original_inside = image_meta.as_tensor().numpy()[0][mask == 1].copy()
    transform = MaskNonLungVoxelsd(keys=("image",), fill_value=-1024.0)
    out = transform({"image": image_meta, LUNG_MASK_PATH_KEY: str(mask_path)})
    arr = out["image"].as_tensor().numpy()[0]
    np.testing.assert_array_equal(arr[mask == 1], original_inside)


def test_mask_missing_file_raises(tmp_path: Path) -> None:
    image_meta, _, _ = _make_volume(tmp_path)
    bad_path = tmp_path / "does_not_exist.nii.gz"
    transform = MaskNonLungVoxelsd(keys=("image",), fill_value=-1024.0)
    with pytest.raises(FileNotFoundError, match="precompute-lung-masks"):
        transform({"image": image_meta, LUNG_MASK_PATH_KEY: str(bad_path)})

"""Shared pytest fixtures.

The synthetic-blob fixture is the canonical 3D test volume used across
B2 (transforms), B3 (loss), and B4 (inference shape). It encodes a
sphere of HU=50 inside a cube of HU=-700 so that CT-aware transforms can
exercise foreground sampling and intensity windowing on a known input.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
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


@pytest.fixture
def synthetic_blob_paths(tmp_path: Path, synthetic_blob: dict[str, np.ndarray]) -> dict[str, str]:
    """Materialize the synthetic blob to NIfTI at the configured target spacing.

    Spacing matches `cfg.data.target_spacing = (0.79, 0.79, 1.24)` so MONAI's
    `Spacingd` is a near-no-op and downstream tests stay deterministic. Also
    materializa una máscara pulmonar sintética que cubre todo el volumen,
    necesaria para `MaskNonLungVoxelsd`.
    """
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = 0.79
    affine[1, 1] = 0.79
    affine[2, 2] = 1.24

    img_path = tmp_path / "image.nii.gz"
    lbl_path = tmp_path / "label.nii.gz"
    mask_path = tmp_path / "lung_mask.nii.gz"
    nib.save(nib.Nifti1Image(synthetic_blob["image"], affine), str(img_path))
    nib.save(nib.Nifti1Image(synthetic_blob["label"].astype(np.uint8), affine), str(lbl_path))
    lung = np.ones_like(synthetic_blob["label"], dtype=np.uint8)
    nib.save(nib.Nifti1Image(lung, affine), str(mask_path))
    return {
        "image": str(img_path),
        "label": str(lbl_path),
        "patient_id": "synthetic",
        "lung_mask_path": str(mask_path),
    }

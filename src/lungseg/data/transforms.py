"""CT-aware MONAI transforms for chest CT.

B2 will implement `build_train_transforms(cfg)` and `build_val_transforms(cfg)`
following the recipe in CLAUDE.md (HU clip [-1024, 400], CropForegroundd,
RandCropByPosNegLabeld with pos=2/neg=1/num_samples=4, RandFlipd ONLY on
spatial_axis=2).
"""

from __future__ import annotations

from omegaconf import DictConfig


def build_train_transforms(cfg: DictConfig):
    raise NotImplementedError("B2 will implement train transforms.")


def build_val_transforms(cfg: DictConfig):
    raise NotImplementedError("B2 will implement val transforms.")

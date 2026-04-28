"""CT-aware MONAI transforms for chest CT.

Recipe matches the refactor template:
- HU clip [a_min, a_max] from cfg.data.hu_clip (a_max defaults to 400 here).
- CropForegroundd via lung-air threshold.
- Train sampler: RandCropByPosNegLabeld(pos=2, neg=1, num_samples=4).
- Augmentations gated by cfg.training.augment_regime in {none, standard, aggressive}.
- RandFlipd is only allowed on spatial_axis=2 (Z); flipping LR (axis=0) is
  forbidden by CLAUDE.md and a defensive ValueError will be raised if a
  config sneaks it in.
"""

from __future__ import annotations

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)
from omegaconf import DictConfig

KEYS = ["image", "label"]
_AUG_PROB = {"none": 0.0, "standard": 0.15, "aggressive": 0.30}


def _check_no_lr_flip(transforms: list) -> None:
    """Defensive guard against re-introducing the (c) bug from REPORT_DIAGNOSIS."""
    for tr in transforms:
        if isinstance(tr, RandFlipd):
            axis = tr.flipper.spatial_axis
            if axis is None:
                raise ValueError(
                    "RandFlipd without spatial_axis flips all axes, including LR. "
                    "Forbidden on chest CT (CLAUDE.md hard rule)."
                )
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            if 0 in axes:
                raise ValueError(
                    "RandFlipd spatial_axis=0 (LR) is forbidden on chest CT "
                    "(CLAUDE.md hard rule). Use spatial_axis=2 only."
                )


def _pre_transforms(cfg: DictConfig, with_label: bool = True) -> list:
    keys = list(KEYS) if with_label else ["image"]
    spacing_modes = ("bilinear", "nearest") if with_label else ("bilinear",)
    return [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=tuple(cfg.data.target_spacing), mode=spacing_modes),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=float(cfg.data.hu_clip.a_min),
            a_max=float(cfg.data.hu_clip.a_max),
            b_min=float(cfg.data.hu_clip.b_min),
            b_max=float(cfg.data.hu_clip.b_max),
            clip=bool(cfg.data.hu_clip.clip),
        ),
        CropForegroundd(
            keys=keys,
            source_key="image",
            select_fn=lambda x, t=float(cfg.data.crop_foreground.threshold): x > t,
            allow_smaller=True,
        ),
    ]


def _augmentations(prob: float) -> list:
    if prob <= 0.0:
        return []
    return [
        RandFlipd(keys=KEYS, prob=prob, spatial_axis=2),
        RandRotate90d(keys=KEYS, prob=prob, max_k=3, spatial_axes=(0, 1)),
        RandGaussianNoised(keys=["image"], prob=prob, mean=0.0, std=0.02),
        RandGaussianSmoothd(
            keys=["image"],
            prob=prob,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),
        RandScaleIntensityd(keys=["image"], factors=0.10, prob=prob),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=prob),
    ]


def build_train_transforms(cfg: DictConfig) -> Compose:
    regime = str(cfg.training.augment_regime)
    if regime not in _AUG_PROB:
        raise ValueError(f"unknown augment_regime: {regime!r} (expected one of {list(_AUG_PROB)})")
    prob = _AUG_PROB[regime]
    pre = _pre_transforms(cfg)
    crop = RandCropByPosNegLabeld(
        keys=KEYS,
        label_key="label",
        spatial_size=tuple(cfg.training.patch_size),
        pos=float(cfg.data.sampler.pos),
        neg=float(cfg.data.sampler.neg),
        num_samples=int(cfg.data.sampler.num_samples),
        image_key="image",
        image_threshold=0.0,
        allow_smaller=True,
    )
    aug = _augmentations(prob)
    transforms = [*pre, crop, *aug, EnsureTyped(keys=KEYS)]
    _check_no_lr_flip(transforms)
    return Compose(transforms)


def build_val_transforms(cfg: DictConfig, with_label: bool = True) -> Compose:
    keys = list(KEYS) if with_label else ["image"]
    return Compose([*_pre_transforms(cfg, with_label=with_label), EnsureTyped(keys=keys)])

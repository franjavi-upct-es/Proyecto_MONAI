"""Transforms tests: shape, sampler balance, defensive flip-axis check."""

from __future__ import annotations

import numpy as np
import pytest
from monai.transforms import RandFlipd
from omegaconf import OmegaConf

from lungseg.data.transforms import build_train_transforms, build_val_transforms


def _cfg(augment_regime: str = "standard"):
    return OmegaConf.create(
        {
            "data": {
                "target_spacing": [0.79, 0.79, 1.24],
                "hu_clip": {"a_min": -1024, "a_max": 400, "b_min": 0.0, "b_max": 1.0, "clip": True},
                "crop_foreground": {"threshold": 0.1},
                "sampler": {"pos": 2, "neg": 1, "num_samples": 4},
            },
            "training": {
                "patch_size": [96, 96, 96],
                "augment_regime": augment_regime,
            },
        }
    )


def test_no_lr_flip_in_train_transforms() -> None:
    train_tf = build_train_transforms(_cfg(augment_regime="aggressive"))
    flips = [t for t in train_tf.transforms if isinstance(t, RandFlipd)]
    assert flips, "expected at least one RandFlipd in aggressive regime"
    for tr in flips:
        axis = tr.flipper.spatial_axis
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        assert 0 not in axes, f"RandFlipd has forbidden spatial_axis=0 (LR): {axes}"


def test_train_patch_shape(synthetic_blob_paths: dict[str, str]) -> None:
    train_tf = build_train_transforms(_cfg(augment_regime="none"))
    samples = train_tf(synthetic_blob_paths)
    assert isinstance(samples, list)
    assert len(samples) == 4  # cfg.data.sampler.num_samples
    for s in samples:
        assert s["image"].shape == (1, 96, 96, 96), s["image"].shape
        assert s["label"].shape == (1, 96, 96, 96), s["label"].shape


def test_pos_neg_sampling(synthetic_blob_paths: dict[str, str]) -> None:
    """With pos=2, neg=1 the expected positive fraction is ~2/3.
    20 iterations x 4 samples = 80 patches; assert >60% have any tumor voxel.
    """
    train_tf = build_train_transforms(_cfg(augment_regime="none"))
    has_fg = []
    for _ in range(20):
        for s in train_tf(synthetic_blob_paths):
            has_fg.append(int(s["label"].sum()) > 0)
    rate = float(np.mean(has_fg))
    assert rate > 0.60, f"foreground rate {rate:.2f} below 60%"


def test_defensive_check_rejects_lr_flip() -> None:
    """The defensive check must raise if someone constructs a Compose that
    sneaks RandFlipd(spatial_axis=0). Use the private helper directly.
    """
    from lungseg.data.transforms import _check_no_lr_flip

    bad = [RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0)]
    with pytest.raises(ValueError, match="LR"):
        _check_no_lr_flip(bad)


def test_val_transforms_no_random(synthetic_blob_paths: dict[str, str]) -> None:
    val_tf = build_val_transforms(_cfg(augment_regime="aggressive"))
    out_a = val_tf(synthetic_blob_paths)
    out_b = val_tf(synthetic_blob_paths)
    np.testing.assert_array_equal(np.asarray(out_a["image"]), np.asarray(out_b["image"]))
    np.testing.assert_array_equal(np.asarray(out_a["label"]), np.asarray(out_b["label"]))

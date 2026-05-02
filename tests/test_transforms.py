"""Transforms tests: shape, sampler balance, defensive flip-axis check."""

from __future__ import annotations

import numpy as np
import pytest
from monai.transforms import CropForegroundd, RandFlipd
from omegaconf import OmegaConf

from lungseg.data.transforms import MultiWindowHUd, build_train_transforms, build_val_transforms


def _cfg(augment_regime: str = "standard"):
    return OmegaConf.create(
        {
            "data": {
                "target_spacing": [0.79, 0.79, 1.24],
                "hu_windows": [
                    [-1000, 0],
                    [-150, 250],
                    [-1024, 400],
                ],
                "lung_mask": {
                    "cache_dir": "data/cache/lung_masks",
                    "fill_value": -1024.0,
                },
                "crop_foreground": {"threshold": -1024.0},
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


def test_crop_foreground_before_multi_window() -> None:
    train_tf = build_train_transforms(_cfg(augment_regime="none"))
    crop_idx = next(
        i for i, tr in enumerate(train_tf.transforms) if isinstance(tr, CropForegroundd)
    )
    multi_window_idx = next(
        i for i, tr in enumerate(train_tf.transforms) if isinstance(tr, MultiWindowHUd)
    )
    assert crop_idx < multi_window_idx


def test_multi_window_does_not_mutate_input_metatensor() -> None:
    """`MultiWindowHUd` debe construir un MetaTensor nuevo, no mutar el de entrada.

    Si mutara, vivir dentro de `CacheDataset(copy_cache=False)` corrompería el
    cache (1→3 canales tras la iteración 1) e inflaría la RAM por COW desde los
    workers — el OOM clásico de mid-training.
    """
    import torch
    from monai.data import MetaTensor

    image = MetaTensor(torch.full((1, 8, 8, 8), -200.0))
    transform = MultiWindowHUd(keys=("image",))
    out = transform({"image": image})

    assert tuple(image.shape) == (1, 8, 8, 8), "MultiWindowHUd mutó la entrada"
    assert tuple(out["image"].shape) == (3, 8, 8, 8)
    assert out["image"] is not image


def test_multi_window_is_non_cacheable() -> None:
    """`MultiWindowHUd` debe ser tratado como per-sample por CacheDataset.

    Si se ejecutara dentro del cache, el CacheDataset guardaría volúmenes con
    3 canales float32 en RAM, triplicando la huella y causando OOM en el
    perfil local_5060. La invariante: la transform hereda de
    `RandomizableTrait`, que MONAI usa como marcador "no cacheable".
    """
    from monai.transforms import RandomizableTrait

    train_tf = build_train_transforms(_cfg(augment_regime="none"))
    multi_window = next(t for t in train_tf.transforms if isinstance(t, MultiWindowHUd))
    assert isinstance(multi_window, RandomizableTrait)


def test_train_patch_shape(synthetic_blob_paths: dict[str, str]) -> None:
    train_tf = build_train_transforms(_cfg(augment_regime="none"))
    samples = train_tf(synthetic_blob_paths)
    assert isinstance(samples, list)
    assert len(samples) == 4  # cfg.data.sampler.num_samples
    for s in samples:
        # 3 ventanas HU -> 3 canales en 'image'.
        assert s["image"].shape == (3, 96, 96, 96), s["image"].shape
        assert s["label"].shape == (1, 96, 96, 96), s["label"].shape
        assert hasattr(s["foreground_start_coord"], "numel")
        assert hasattr(s["foreground_end_coord"], "numel")


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
    assert hasattr(out_a["foreground_start_coord"], "numel")
    assert hasattr(out_a["foreground_end_coord"], "numel")
    np.testing.assert_array_equal(np.asarray(out_a["image"]), np.asarray(out_b["image"]))
    np.testing.assert_array_equal(np.asarray(out_a["label"]), np.asarray(out_b["label"]))

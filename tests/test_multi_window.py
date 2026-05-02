"""Tests para `MultiWindowHUd`."""

from __future__ import annotations

import torch

from lungseg.data.transforms import MultiWindowHUd


def _input(value: float) -> dict:
    image = torch.full((1, 4, 4, 4), value, dtype=torch.float32)
    return {"image": image}


def test_three_channels_in_zero_one_range() -> None:
    transform = MultiWindowHUd(keys=("image",))
    sample = {"image": torch.full((1, 4, 4, 4), 0.0, dtype=torch.float32)}
    out = transform(sample)["image"]
    assert tuple(out.shape) == (3, 4, 4, 4)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_low_hu_saturates_to_zero() -> None:
    transform = MultiWindowHUd(keys=("image",))
    out = transform(_input(-1500.0))["image"]
    # -1500 está por debajo de a_min de las tres ventanas (-1000, -150, -1024).
    assert torch.allclose(out, torch.zeros_like(out))


def test_high_hu_saturates_to_one() -> None:
    transform = MultiWindowHUd(keys=("image",))
    out = transform(_input(2000.0))["image"]
    # 2000 está por encima de a_max de las tres ventanas (0, 250, 400).
    assert torch.allclose(out, torch.ones_like(out))


def test_custom_windows_respected() -> None:
    transform = MultiWindowHUd(keys=("image",), windows=[[-100, 100]])
    out = transform(_input(0.0))["image"]
    assert tuple(out.shape) == (1, 4, 4, 4)
    # 0 HU está al 50% de la ventana [-100, 100].
    assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-6)

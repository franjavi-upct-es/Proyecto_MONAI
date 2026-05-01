"""Pruebas de pérdida, planificador y fábrica de modelos."""

from __future__ import annotations

from itertools import pairwise

import torch
from omegaconf import OmegaConf

from lungseg.models import build_model
from lungseg.training.losses import build_loss
from lungseg.training.schedulers import poly_lr


def _cfg(model_name: str = "vit_25d"):
    model = {
        "name": model_name,
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
        "encoder_name": "vit_base_patch16_224",
        "pretrained": False,
        "loss": {
            "name": "dice_focal",
            "to_onehot_y": True,
            "softmax": True,
            "include_background": False,
        },
    }
    return OmegaConf.create({"model": model})


def test_model_factories_forward_shape() -> None:
    # 2.5D ViT procesará cortes de profundidad como batch.
    # 32x32x32 evita problemas de resolución de parches
    x = torch.randn(1, 1, 32, 32, 32)
    model = build_model(_cfg("vit_25d"))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 32, 32, 32)


def test_dice_focal_loss_perfect_prediction_is_small() -> None:
    cfg = _cfg()
    loss_fn = build_loss(cfg)
    target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    target[:, :, 2:6, 2:6, 2:6] = 1
    logits = torch.full((1, 2, 8, 8, 8), -8.0)
    logits[:, 0][target[:, 0] == 0] = 8.0
    logits[:, 1][target[:, 0] == 1] = 8.0
    random_logits = torch.zeros_like(logits)
    assert float(loss_fn(logits, target)) < 0.1
    assert float(loss_fn(random_logits, target)) > 0.5


def test_poly_lr_is_monotonic_and_clamped() -> None:
    values = [poly_lr(step, max_steps=10, base_lr=1.0) for step in range(12)]
    assert values[0] == 1.0
    assert values[-1] == 0.0
    assert all(a >= b for a, b in pairwise(values))

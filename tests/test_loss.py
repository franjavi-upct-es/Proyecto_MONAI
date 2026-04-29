"""Pruebas de pérdida, planificador y fábrica de modelos."""

from __future__ import annotations

from itertools import pairwise

import torch
from omegaconf import OmegaConf

from lungseg.models import build_model
from lungseg.training.losses import build_loss, deep_supervision_loss
from lungseg.training.schedulers import poly_lr


def _cfg(model_name: str = "segresnet"):
    model = {
        "name": model_name,
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 2,
        "loss": {
            "name": "dice_ce",
            "to_onehot_y": True,
            "softmax": True,
            "include_background": False,
            "lambda_dice": 1.0,
            "lambda_ce": 1.0,
        },
    }
    if model_name == "segresnet":
        model.update(
            {
                "init_filters": 4,
                "blocks_down": [1, 1],
                "blocks_up": [1],
                "dropout_prob": 0.0,
                "norm": "instance",
                "act": "relu",
            }
        )
    elif model_name == "dynunet":
        model.update(
            {
                "kernel_size": [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                "upsample_kernel_size": [[2, 2, 2], [2, 2, 2]],
                "norm_name": "instance",
                "deep_supervision": True,
                "deep_supr_num": 1,
                "res_block": True,
            }
        )
    else:
        model.update({"channels": [4, 8], "strides": [2], "num_res_units": 1})
    return OmegaConf.create({"model": model})


def test_model_factories_forward_shape() -> None:
    x = torch.randn(1, 1, 16, 16, 16)
    for name in ["segresnet", "dynunet", "unet"]:
        model = build_model(_cfg(name))
        model.train(name == "dynunet")
        if name != "dynunet":
            model.eval()
        with torch.no_grad():
            out = model(x)
        if name == "dynunet":
            assert out.shape[:3] == (1, 2, 2)
            assert out[:, 0].shape == (1, 2, 16, 16, 16)
        else:
            assert out.shape == (1, 2, 16, 16, 16)


def test_dice_ce_loss_perfect_prediction_is_small() -> None:
    cfg = _cfg()
    loss_fn = build_loss(cfg)
    target = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    target[:, :, 2:6, 2:6, 2:6] = 1
    logits = torch.full((1, 2, 8, 8, 8), -8.0)
    logits[:, 0][target[:, 0] == 0] = 8.0
    logits[:, 1][target[:, 0] == 1] = 8.0
    random_logits = torch.zeros_like(logits)
    assert float(loss_fn(logits, target)) < 0.05
    assert float(loss_fn(random_logits, target)) > 0.5


def test_deep_supervision_loss_uses_normalized_geometric_weights() -> None:
    out = torch.stack(
        [
            torch.ones(1, 2, 4, 4, 4),
            torch.full((1, 2, 4, 4, 4), 2.0),
            torch.full((1, 2, 4, 4, 4), 3.0),
        ],
        dim=1,
    )
    target = torch.zeros(1, 1, 4, 4, 4)
    value = deep_supervision_loss(out, target, lambda pred, _target: pred.mean())
    expected = (1.0 * 0.5 + 2.0 * 0.25 + 3.0 * 0.125) / (0.5 + 0.25 + 0.125)
    assert torch.isclose(value, torch.tensor(expected))


def test_poly_lr_is_monotonic_and_clamped() -> None:
    values = [poly_lr(step, max_steps=10, base_lr=1.0) for step in range(12)]
    assert values[0] == 1.0
    assert values[-1] == 0.0
    assert all(a >= b for a, b in pairwise(values))

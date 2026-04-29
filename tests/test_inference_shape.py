"""Pruebas de humo de inferencia y entrenador."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from lungseg.inference import predict_volume
from lungseg.training import train_iters
from lungseg.utils.metrics import compute_segmentation_metrics


class DeepSupervisionToy(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bg = -x
        fg = x
        final = torch.cat([bg, fg], dim=1)
        aux = final * 0.5
        return torch.stack([final, aux], dim=1)


def _trainer_cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "seed": 0,
            "fold": 0,
            "paths": {"outputs": str(tmp_path / "outputs")},
            "data": {"target_spacing": [1.0, 1.0, 1.0]},
            "model": {
                "loss": {
                    "name": "dice_ce",
                    "to_onehot_y": True,
                    "softmax": True,
                    "include_background": False,
                }
            },
            "training": {
                "patch_size": [8, 8, 8],
                "batch_size": 1,
                "amp": False,
                "optimizer": {"name": "adamw", "lr": 1.0e-3, "weight_decay": 0.0},
                "scheduler": {"name": "poly", "exp": 0.9},
                "inference": {"sw_batch_size": 1, "overlap": 0.0, "mode": "constant"},
            },
            "experiment": {
                "max_iterations": 2,
                "val_every": 1,
                "patience": 10,
                "grad_accum_steps": 1,
                "log_every": 1,
                "fixed_iterations": True,
            },
        }
    )


def test_predict_volume_selects_final_deep_supervision_output() -> None:
    cfg = _trainer_cfg(Path("/tmp"))
    image = torch.ones(1, 1, 16, 16, 16)
    out = predict_volume(DeepSupervisionToy(), image, cfg)
    assert out.shape == (1, 2, 16, 16, 16)
    assert torch.all(out[:, 1] == 1.0)


def test_compute_segmentation_metrics_perfect_mask() -> None:
    label = torch.zeros(1, 1, 8, 8, 8)
    label[:, :, 2:6, 2:6, 2:6] = 1
    logits = torch.cat([1 - label, label], dim=1)
    metrics = compute_segmentation_metrics(logits, label)
    assert metrics["dice"] == 1.0
    assert metrics["hd95"] == 0.0


def test_train_iters_smoke_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    cfg = _trainer_cfg(tmp_path)
    image = torch.zeros(1, 8, 8, 8)
    label = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    image[:, 2:6, 2:6, 2:6] = 1
    label[:, 2:6, 2:6, 2:6] = 1
    loader = DataLoader([{"image": image, "label": label}], batch_size=1)
    model = torch.nn.Conv3d(1, 2, kernel_size=1)
    summary = train_iters(cfg, model, (loader, loader))
    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["metrics_path"]).exists()
    payload = json.loads((tmp_path / "outputs" / "summary.json").read_text())
    assert payload["last_step"] == 2

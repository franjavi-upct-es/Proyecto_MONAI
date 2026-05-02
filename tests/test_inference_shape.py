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


def _trainer_cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "seed": 0,
            "fold": 0,
            "paths": {"outputs": str(tmp_path / "outputs")},
            "data": {"target_spacing": [1.0, 1.0, 1.0]},
            "model": {
                "loss": {
                    "name": "focal_tversky",
                    "alpha": 0.3,
                    "beta": 0.7,
                    "gamma": 1.333,
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


def test_predict_volume_shape() -> None:
    cfg = _trainer_cfg(Path("/tmp"))
    image = torch.ones(1, 1, 16, 16, 16)
    model = torch.nn.Conv3d(1, 2, kernel_size=1)
    out = predict_volume(model, image, cfg)
    assert out.shape == (1, 2, 16, 16, 16)


def test_predict_volume_auto_bypasses_3d_sliding_window_for_vit25d() -> None:
    cfg = _trainer_cfg(Path("/tmp"))
    OmegaConf.update(cfg, "model.name", "vit_25d", force_add=True)
    image = torch.ones(1, 1, 8, 8, 4)

    class FakeSliceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            assert x.shape == image.shape
            return torch.zeros(x.shape[0], 2, *x.shape[2:])

    model = FakeSliceModel()
    out = predict_volume(model, image, cfg)
    assert model.calls == 1
    assert out.shape == (1, 2, 8, 8, 4)


def test_predict_volume_auto_slides_vit25d_in_xy_only() -> None:
    cfg = _trainer_cfg(Path("/tmp"))
    OmegaConf.update(cfg, "model.name", "vit_25d", force_add=True)
    image = torch.ones(1, 1, 16, 16, 5)

    class FakeSliceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            assert x.shape[-3:] == (8, 8, 5)
            return torch.zeros(x.shape[0], 2, *x.shape[2:])

    model = FakeSliceModel()
    out = predict_volume(model, image, cfg)
    assert model.calls == 4
    assert out.shape == (1, 2, 16, 16, 5)


def test_compute_segmentation_metrics_perfect_mask() -> None:
    label = torch.zeros(1, 1, 8, 8, 8)
    label[:, :, 2:6, 2:6, 2:6] = 1
    logits = torch.cat([1 - label, label], dim=1)
    metrics = compute_segmentation_metrics(logits, label)
    assert metrics["dice"] == 1.0
    assert metrics["hd95"] == 0.0


def test_compute_segmentation_metrics_accepts_bfloat16_logits() -> None:
    label = torch.zeros(1, 1, 8, 8, 8)
    label[:, :, 2:6, 2:6, 2:6] = 1
    logits = torch.zeros(1, 2, 8, 8, 8, dtype=torch.bfloat16)
    logits[:, 0] = 1
    logits[:, 0, 2:6, 2:6, 2:6] = 0
    logits[:, 1, 2:6, 2:6, 2:6] = 1
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
    checkpoint = torch.load(summary["last_checkpoint_path"], map_location="cpu")
    # En el nuevo Trainer, step se incrementa cada vez que se llama a _optimizer_step
    # Con 1 batch y grad_accum=1, step == epoch_idx + 1 al final de la época
    # Si max_iterations=2 (épocas en el nuevo Trainer), step será 2.
    assert checkpoint["step"] == 2
    assert "scheduler_state_dict" in checkpoint
    payload = json.loads((tmp_path / "outputs" / "summary.json").read_text())
    assert payload["last_step"] == 2

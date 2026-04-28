"""Ablation runner driven by Hydra/CLI overrides."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from lungseg.data import build_loaders
from lungseg.models import build_model
from lungseg.training import train_iters


def _select(cfg: DictConfig, key: str, default=None):
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def _outputs_dir(cfg: DictConfig) -> Path:
    value = _select(cfg, "paths.outputs", "outputs/manual")
    if "${hydra:" in str(value):
        value = "outputs/manual"
    path = Path(str(value))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cell_value(cfg: DictConfig, name: str, default):
    return _select(cfg, name, _select(cfg, f"experiment.{name}", default))


def _sample_train_records(records: list[dict], fraction: float, seed: int) -> list[dict]:
    if fraction >= 1.0:
        return list(records)
    if fraction <= 0.0:
        raise ValueError("data_fraction must be > 0")
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    strata = sorted({int(r.get("stratum", 0)) for r in records})
    for stratum in strata:
        bucket = [r for r in records if int(r.get("stratum", 0)) == stratum]
        n_keep = max(1, round(len(bucket) * fraction))
        indices = np.sort(rng.choice(len(bucket), size=min(n_keep, len(bucket)), replace=False))
        selected.extend(bucket[int(i)] for i in indices)
    return sorted(selected, key=lambda r: r["patient_id"])


def _write_fractional_split(cfg: DictConfig, fraction: float, seed: int) -> Path:
    fold = int(_select(cfg, "fold", 0))
    source_dir = Path(str(_select(cfg, "paths.splits", "data/splits")))
    source_path = source_dir / f"fold_{fold}.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["train"] = _sample_train_records(payload["train"], fraction=fraction, seed=seed)
    payload["n_train"] = len(payload["train"])
    payload["ablation"] = {"data_fraction": fraction, "seed": seed}

    out_dir = _outputs_dir(cfg) / "ablation_splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold_{fold}_frac_{fraction:g}_seed_{seed}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    active_dir = out_dir / f"active_frac_{fraction:g}_seed_{seed}"
    active_dir.mkdir(parents=True, exist_ok=True)
    (active_dir / f"fold_{fold}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return active_dir


def run_cell(cfg: DictConfig) -> dict:
    """Run one ablation cell and write a JSON result."""
    seed = int(_cell_value(cfg, "seed", 42))
    fraction = float(_cell_value(cfg, "data_fraction", 1.0))
    augment_regime = str(_cell_value(cfg, "aug_regime", _select(cfg, "training.augment_regime", "standard")))

    cell_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(cell_cfg, "seed", seed, merge=False)
    OmegaConf.update(cell_cfg, "training.augment_regime", augment_regime, merge=False)
    OmegaConf.update(cell_cfg, "experiment.fixed_iterations", True, force_add=True)
    split_dir = _write_fractional_split(cell_cfg, fraction=fraction, seed=seed)
    OmegaConf.update(cell_cfg, "paths.splits", str(split_dir), merge=False)

    model = build_model(cell_cfg)
    loaders = build_loaders(cell_cfg, fold=int(_select(cell_cfg, "fold", 0)))
    summary = train_iters(cell_cfg, model, loaders)
    result = {
        "seed": seed,
        "data_fraction": fraction,
        "augment_regime": augment_regime,
        **summary,
    }
    out_dir = _outputs_dir(cell_cfg) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"frac_{fraction:g}_aug_{augment_regime}_seed_{seed}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = str(out_path)
    return result

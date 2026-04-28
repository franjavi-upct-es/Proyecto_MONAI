"""Phase 5/6 and Kaggle notebook contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from lungseg.ablation.analysis import analyze
from lungseg.ablation.runner import _sample_train_records
from lungseg.classification import evaluate_pipeline, evaluate_size_only
from lungseg.radiomics import build_radiomic_dataset


def test_task06_classification_fails_clearly() -> None:
    cfg = OmegaConf.create({"data": {"name": "task06"}})
    with pytest.raises(ValueError, match="no benign/malignant labels"):
        build_radiomic_dataset(cfg)


def test_classification_pipeline_grouped_cv_outputs_metrics() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 8)
    groups = np.array([f"p{i}" for i in range(len(y))])
    X = rng.normal(size=(len(y), 6)).astype(np.float32)
    X[:, 0] += y * 2.0
    cfg = OmegaConf.create({"seed": 0, "data": {"n_folds": 4}})
    full = evaluate_pipeline(X, y, groups, cfg)
    size = evaluate_size_only(X[:, 0], y, groups)
    assert {"rf", "lasso", "mlp"}.issubset(full)
    assert 0.0 <= full["rf"]["balanced_acc"] <= 1.0
    assert 0.0 <= size["brier"] <= 1.0


def test_ablation_fractional_sampling_is_deterministic_and_stratified() -> None:
    records = [
        {"patient_id": f"p{i}", "stratum": i % 3}
        for i in range(18)
    ]
    a = _sample_train_records(records, fraction=0.5, seed=7)
    b = _sample_train_records(records, fraction=0.5, seed=7)
    assert a == b
    assert {r["stratum"] for r in a} == {0, 1, 2}
    assert len(a) < len(records)


def test_ablation_analysis_from_fake_cells(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    ablation = out / "ablation"
    ablation.mkdir(parents=True)
    for seed in [0, 1, 2]:
        for aug, offset in [("none", 0.0), ("standard", 0.1)]:
            payload = {
                "seed": seed,
                "data_fraction": 0.5,
                "augment_regime": aug,
                "best_val_dice": 0.2 + offset + seed * 0.01,
                "best_val_hd95": 12.0 - offset,
            }
            (ablation / f"{aug}_{seed}.json").write_text(json.dumps(payload), encoding="utf-8")
    report = analyze(out)
    assert report.exists()
    assert (out / "ablation_summary.csv").exists()


def test_kaggle_notebook_is_valid_json_and_references_configs() -> None:
    path = Path("notebooks/kaggle_phase4_phase6.ipynb")
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert payload["nbformat"] == 4
    assert "training=kaggle_p100" in source
    assert "experiment=phase4_full" in source

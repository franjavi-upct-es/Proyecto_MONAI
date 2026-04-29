"""Línea base de cordura de una sola característica: clasificar solo según el VoxelVolume del tumor."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from lungseg.classification.pipeline import _ece, _n_splits


def evaluate_size_only(volumes: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    volumes = np.asarray(volumes, dtype=np.float32).reshape(-1, 1)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    n_splits = _n_splits(y, groups, requested=5)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float64)
    folds = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(volumes, y, groups=groups)):
        pipe = Pipeline(
            steps=[
                ("scale", RobustScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe.fit(volumes[train_idx], y[train_idx])
        proba = pipe.predict_proba(volumes[val_idx])[:, 1]
        oof[val_idx] = proba
        pred = (proba >= 0.5).astype(int)
        folds.append(
            {
                "fold": fold,
                "auc": float(roc_auc_score(y[val_idx], proba))
                if len(np.unique(y[val_idx])) == 2
                else float("nan"),
                "balanced_acc": float(balanced_accuracy_score(y[val_idx], pred)),
                "brier": float(brier_score_loss(y[val_idx], proba)),
                "ece": _ece(y[val_idx], proba),
            }
        )
    pred = (oof >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, oof)) if len(np.unique(y)) == 2 else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, oof)),
        "ece": _ece(y, oof),
        "n_splits": n_splits,
        "folds": folds,
    }

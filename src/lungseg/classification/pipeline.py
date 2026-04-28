"""Radiomic classification pipelines for Phase 5."""

from __future__ import annotations

from functools import partial
from itertools import pairwise

import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    if total == 0:
        return float("nan")
    ece = 0.0
    for lo, hi in pairwise(bins):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_prob[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += (float(mask.sum()) / total) * abs(confidence - accuracy)
    return float(ece)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": _safe_auc(y_true, y_prob),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": _ece(y_true, y_prob),
    }


def _n_splits(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("classification CV requires at least two patient groups")
    min_class = int(np.min(np.bincount(y.astype(int))))
    n_splits = min(int(requested), len(unique_groups), min_class)
    if n_splits < 2:
        raise ValueError("classification CV requires at least two samples in each class")
    return n_splits


def _models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "rf": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "lasso": LogisticRegression(
            l1_ratio=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32,),
            alpha=1.0e-3,
            max_iter=500,
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgb"] = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        )
    except ImportError:
        pass
    return models


def _make_pipeline(estimator: object, k: int, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", RobustScaler()),
            ("select", SelectKBest(partial(mutual_info_classif, random_state=seed), k=k)),
            ("clf", estimator),
        ]
    )


def _positive_proba(model: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    decision = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-decision))


def evaluate_pipeline(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: DictConfig) -> dict:
    """Evaluate full radiomic models with patient-level grouped CV."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y and groups must have the same length")

    seed = int(OmegaConf.select(cfg, "seed", default=42))
    requested_splits = int(OmegaConf.select(cfg, "data.n_folds", default=5))
    n_splits = _n_splits(y, groups, requested_splits)
    k = min(20, X.shape[1])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results: dict[str, dict] = {}
    for name, estimator in _models(seed).items():
        oof = np.zeros(len(y), dtype=np.float64)
        fold_rows = []
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=groups)):
            pipe = _make_pipeline(clone(estimator), k=k, seed=seed)
            pipe.fit(X[train_idx], y[train_idx])
            proba = _positive_proba(pipe, X[val_idx])
            oof[val_idx] = proba
            fold_metric = _metrics(y[val_idx], proba)
            fold_rows.append({"fold": fold, **fold_metric})
        results[name] = {
            **_metrics(y, oof),
            "n_splits": n_splits,
            "k_features": k,
            "folds": fold_rows,
        }
    return results

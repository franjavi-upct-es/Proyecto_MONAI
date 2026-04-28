"""sklearn pipeline:
  RobustScaler -> SelectKBest(MI, k=20) -> {RF, XGB, LASSO, MLP}
GroupKFold by patient_id; AUC + balanced_acc + Brier + ECE.
Filled in B5.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


def evaluate_pipeline(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: DictConfig) -> dict:
    raise NotImplementedError("B5 will implement evaluate_pipeline.")

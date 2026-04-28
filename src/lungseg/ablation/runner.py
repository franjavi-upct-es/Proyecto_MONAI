"""Ablation runner driven by Hydra multirun.

B6 wiring (planned):
  data_fraction in {0.25, 0.5, 1.0}
  aug_regime in {none, standard}
  seed in {0, 1, 2}
  max_iterations: fixed (10_000 by default; see configs/experiment/phase6_ablation.yaml)
  val_every: 200
"""

from __future__ import annotations

from omegaconf import DictConfig


def run_cell(cfg: DictConfig) -> dict:
    raise NotImplementedError("B6 will implement run_cell.")

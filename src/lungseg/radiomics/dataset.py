"""Dataset builder that extracts radiomic features for every case using either
GT masks (default) or predicted masks (optional --e2e mode). Filled in B5.
"""

from __future__ import annotations

from omegaconf import DictConfig


def build_radiomic_dataset(cfg: DictConfig, e2e: bool = False):
    raise NotImplementedError("B5 will implement build_radiomic_dataset.")

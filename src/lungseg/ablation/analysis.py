"""Read outputs/ablation/*.json, build pivot tables (median +/- IQR of DSC and
HD95), draw violin plots per configuration, run pairwise Wilcoxon between
configs (matched by seed) and emit REPORT_ABLATION.md. Filled in B6.
"""

from __future__ import annotations

from pathlib import Path


def analyze(outputs_dir: Path) -> Path:
    raise NotImplementedError("B6 will implement analyze.")

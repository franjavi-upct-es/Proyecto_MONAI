"""Patient-level GroupKFold splits with rough stratification by tumor volume.

B2 will implement `make_splits(seed=42, k=5)`. The stub raises so anyone who
imports it during B1 verification is told why.
"""

from __future__ import annotations

from pathlib import Path


def make_splits(seed: int = 42, k: int = 5, out_dir: Path | None = None) -> list[Path]:
    raise NotImplementedError("B2 will implement patient-level GroupKFold splits.")

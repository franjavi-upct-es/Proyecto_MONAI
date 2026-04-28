"""Classification pipelines (Phase 5).

NOTE: MSD Task06_Lung does NOT contain benign/malignant labels. The
B/M-classification path requires `--dataset=lidc-idri` (B5) or a clinically
meaningful proxy. The pipeline raises a clear error if invoked with Task06.
"""

from __future__ import annotations

from lungseg.classification.baseline import evaluate_size_only
from lungseg.classification.pipeline import evaluate_pipeline

__all__ = ["evaluate_pipeline", "evaluate_size_only"]

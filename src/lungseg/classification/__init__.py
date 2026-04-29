"""Pipelines de clasificación (Fase 5).

NOTA: MSD Task06_Lung NO contiene etiquetas de benigno/maligno. El
camino de clasificación B/M requiere `--dataset=lidc-idri` (B5) o un
proxy clínicamente significativo. El pipeline lanza un error claro si se invoca con Task06.
"""

from __future__ import annotations

from lungseg.classification.baseline import evaluate_size_only
from lungseg.classification.pipeline import evaluate_pipeline

__all__ = ["evaluate_pipeline", "evaluate_size_only"]

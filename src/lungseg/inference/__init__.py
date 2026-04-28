"""Sliding-window inference + postprocessing."""

from __future__ import annotations

from lungseg.inference.postprocessing import remove_small_components
from lungseg.inference.sliding_window import predict_volume

__all__ = ["predict_volume", "remove_small_components"]

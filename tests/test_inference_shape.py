"""Inference-shape tests. Real assertions in B4: DynUNet with deep_supervision
returns (B, n_ds, C, ...) and select(1, 0) has the expected spatial shape;
sliding_window_inference output matches input volume shape.
"""

from __future__ import annotations


def test_placeholder() -> None:
    assert True

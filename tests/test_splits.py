"""Splits tests. Real assertion in B2: no patient appears in train and val
of the same fold simultaneously.
"""

from __future__ import annotations


def test_placeholder() -> None:
    assert True

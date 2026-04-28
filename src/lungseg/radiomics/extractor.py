"""PyRadiomics wrapper used by Phase 5."""

from __future__ import annotations

from pathlib import Path


def extract(image_path: Path, mask_path: Path) -> dict[str, float]:
    """Extract numeric radiomic features for one image/mask pair.

    PyRadiomics is an optional dependency (`pip install -e .[radiomics]`).
    Diagnostics and other non-numeric values are intentionally dropped so the
    returned dict can be fed directly into pandas/sklearn.
    """
    try:
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError(
            "PyRadiomics is required for Phase 5. Install with `pip install -e .[radiomics]`."
        ) from exc

    image_path = Path(image_path)
    mask_path = Path(mask_path)
    if not image_path.exists():
        raise FileNotFoundError(f"radiomics image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"radiomics mask not found: {mask_path}")

    extractor = featureextractor.RadiomicsFeatureExtractor(
        resampledPixelSpacing=[1.0, 1.0, 1.0],
        interpolator="sitkBSpline",
        binWidth=25,
        label=1,
    )
    extractor.enableImageTypes(Original={}, LoG={"sigma": [1.0, 2.0, 3.0]}, Wavelet={})
    result = extractor.execute(str(image_path), str(mask_path))
    features: dict[str, float] = {}
    for key, value in result.items():
        if key.startswith("diagnostics_"):
            continue
        try:
            features[key] = float(value)
        except (TypeError, ValueError):
            continue
    return features

"""Cómputo y cacheo de máscaras pulmonares para preprocesamiento.

Genera una máscara binaria del parénquima pulmonar (left + right merged)
usando `lungmask` (modelo R231) y la cachea en NIfTI bajo
`data/cache/lung_masks/<patient_id>.nii.gz`. Cualquier consumidor del
pipeline (ver `MaskNonLungVoxelsd`) carga la máscara cacheada y la usa para
ponerlos voxels no-pulmón a un HU constante antes de ventanear.

El cómputo es idempotente: si la máscara existe se devuelve sin recomputar.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk  # noqa: N813 — convención estándar del paquete

from lungseg.utils.logging import get_logger

LOGGER = get_logger(__name__)

_LUNGMASK_MODEL = "R231"


def _patient_id_from_path(image_path: Path) -> str:
    name = image_path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return image_path.stem


def _load_inferer():
    try:
        from lungmask import LMInferer
    except ImportError as exc:  # pragma: no cover - dependencia opcional
        raise ImportError(
            "lungmask no está instalado. Añade `lungmask` a tu entorno "
            "(p.ej. `uv pip install lungmask`) antes de pre-computar máscaras."
        ) from exc
    return LMInferer(modelname=_LUNGMASK_MODEL)


def compute_lung_mask(image_path: Path, cache_dir: Path) -> Path:
    """Calcula la máscara pulmonar binaria y la cachea como NIfTI.

    Usa SimpleITK para preservar la geometría física que `lungmask` necesita
    (`GetDirection`, `GetSpacing`...). El resultado es una máscara en formato
    sitk con la misma geometría que la imagen de entrada; volcamos a NIfTI
    fijando el affine de nibabel desde la imagen original para que MONAI
    encuentre las máscaras alineadas espacialmente. La salida vale {0, 1}
    con 1 = parénquima pulmonar (left + right fundidos).
    """
    image_path = Path(image_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    patient_id = _patient_id_from_path(image_path)
    cache_path = cache_dir / f"{patient_id}.nii.gz"
    if cache_path.exists():
        return cache_path

    image_sitk = sitk.ReadImage(str(image_path))
    inferer = _load_inferer()
    raw_mask = inferer.apply(image_sitk)  # numpy array en orden (D, H, W)

    # `raw_mask` es ZYX (sitk convention). NIfTI con nibabel espera XYZ, así
    # que reusamos el affine/header del NIfTI original y transponemos a XYZ.
    mask_zyx = (np.asarray(raw_mask) > 0).astype(np.uint8)
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0))

    image_nii = nib.load(str(image_path))
    if mask_xyz.shape != tuple(image_nii.shape):
        raise RuntimeError(
            f"lungmask produjo shape {mask_xyz.shape}, esperado {tuple(image_nii.shape)}"
        )

    nib.save(nib.Nifti1Image(mask_xyz, image_nii.affine, image_nii.header), str(cache_path))
    LOGGER.info("lung mask cached at %s", cache_path)
    return cache_path


def lung_mask_path(image_path: Path, cache_dir: Path) -> Path:
    """Devuelve la ruta donde estaría cacheada la máscara para `image_path`."""
    return Path(cache_dir) / f"{_patient_id_from_path(Path(image_path))}.nii.gz"

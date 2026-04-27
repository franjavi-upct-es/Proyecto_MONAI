"""
Nucleo compartido del pipeline 2D con preprocesamiento 3D fisicamente
consistente via transformaciones MONAI.

Flujo:
  1. Compose 3D (MONAI) por volumen NIfTI: orientacion RAS, remuestreo a
     TARGET_SPACING_3D, ventaneo CT, crop de foreground.
  2. Estimacion de priors pulmonares sobre la reconstruccion pseudo-HU.
  3. Cacheo en disco de volumenes (Z, Y, X) con tamano espacial variable.
  4. Extraccion axial 2.5D a demanda; crop/pad 2D al vuelo a PATCH_SIZE_2D
     durante entrenamiento/validacion.
  5. Validacion cruzada K-Fold (sklearn) sustituyendo el split estatico.

Mejoras v4 respecto a v3:
  - Funcion de perdida: DiceFocalLoss (gamma=2) sustituye a Dice+BCE.
    Penaliza activamente los pixeles que el modelo clasifica mal de forma
    sistematica, lo que es critico con el fuerte desbalance de este dataset.
  - Arquitectura: UNet con channels (32,64,128,256,512) en lugar de
    (16,32,64,128,256). Mayor capacidad sin cambiar el flujo de datos.
  - Augmentacion: se anaden gamma augmentation y deformacion elastica 2D
    ligera al regimen Fuerte (nivel 2).
  - Scheduler: 3 epocas de warmup lineal seguidas de CosineAnnealingLR.
    Evita divergencias al inicio con LR alto.
  - Manifest: re-muestreo de negativos cada epoca para exponer al modelo
    a mayor variedad sin repetir los mismos negativos faciles.
  - Muestreo de parches: el crop 2D de entrenamiento deja de ser ciego y
    pasa a guiarse por tumor/pulmon para no vaciar ejemplos positivos.
  - Inferencia: overlap 0.25 -> 0.5 en sliding_window_inference para
    reducir artefactos en bordes de parche.
  - TTA: nueva funcion inferir_volumen_tta que promedia predicciones con
    flip horizontal y vertical sin necesidad de reentrenar.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.inferers.utils import sliding_window_inference
from monai.losses.dice import DiceFocalLoss
from monai.networks.nets.unet import UNet
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, Spacingd
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from scipy import ndimage
from sklearn.model_selection import KFold
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset

from monai_pipeline.core.config import (
    ALARM_COMPONENT_MIN_SIZE,
    ALARM_NEGATIVE_EXCLUSION_PIXELS,
    ALARM_SLICE_MIN_PIXELS,
    AMP_ENABLED,
    CACHE_VERSION_2D,
    CONTEXT_SLICES,
    CT_CLIP_RANGE,
    DEVICE,
    DIR_CACHE_2D,
    EARLY_STOPPING_PATIENCE,
    HARD_NEGATIVE_MARGIN,
    HU_HEALTHY_LUNG_RANGE,
    HU_LUNG_CANDIDATE_MAX,
    HU_SOFT_TISSUE_RANGE,
    INFER_BATCH_SIZE,
    INPUT_CHANNELS_2D,
    KFOLD_DEFAULT_INDEX,
    LOSS_FOCAL_GAMMA,
    LOSS_LAMBDA_DICE,
    LOSS_LAMBDA_FOCAL,
    LR,
    LR_WARMUP_EPOCAS,
    MIN_COMPONENT_SIZE,
    MODELO_2D_NOMBRE,
    N_FOLDS_KFOLD,
    N_WORKERS_DATALOADER,
    NEGATIVE_TO_POSITIVE_RATIO,
    PATCH_SIZE_2D,
    POSITIVE_SLICE_MIN_PIXELS,
    PRIOR_CHANNELS_2D,
    RUTA_DATASET_JSON,
    SEMILLA,
    TARGET_SPACING_3D,
    TASK_DIR,
    TRAIN_BATCH_SIZE,
    VAL_INTERVAL,
    WEIGHT_DECAY,
)
from monai_pipeline.core.utils import (
    dice_score,
    estimar_priores_pulmonares,
    evaluar_segmentacion,
    limpiar_componentes_pequenas,
)


@dataclass(slots=True)
class Case2D:
    case_id: str
    image: np.ndarray  # (Z, Y, X), float16 en [0, 1]
    lung_mask: np.ndarray
    healthy_lung_mask: np.ndarray
    soft_tissue_alarm_mask: np.ndarray
    alarm_pixels_per_slice: np.ndarray
    spacing: tuple[float, float, float]  # (Z, Y, X) en mm tras Spacingd
    original_shape: tuple[int, int, int]  # shape NIfTI original (Z, Y, X)
    original_spacing: tuple[float, float, float]  # spacing original (Z, Y, X)
    crop_start: tuple[int, int, int]  # offset de CropForegroundd en (Z, Y, X) post-Spacing
    shape_post_spacing: tuple[int, int, int]  # shape pre-crop post-Spacing en (Z, Y, X)
    label: np.ndarray | None = None
    positive_slices: np.ndarray | None = None
    alarm_slices: np.ndarray | None = None
    source_volume_hu: np.ndarray | None = None


def _ruta_cache(case_id: str) -> Path:
    return DIR_CACHE_2D / f"{case_id}_{CACHE_VERSION_2D}.npz"


def leer_registros_training() -> list[dict[str, Path | str]]:
    if not RUTA_DATASET_JSON.exists():
        raise FileNotFoundError(
            f"No se ha encontrado {RUTA_DATASET_JSON}. "
            "Comprueba que el dataset Task06_Lung este extraido."
        )

    contenido = json.loads(RUTA_DATASET_JSON.read_text(encoding="utf-8"))
    registros = []
    for fila in contenido["training"]:
        ruta_img = TASK_DIR / fila["image"].replace("./", "")
        ruta_lbl = TASK_DIR / fila["label"].replace("./", "")
        case_id = Path(fila["image"]).name.replace(".nii.gz", "")
        registros.append(
            {
                "case_id": case_id,
                "image": ruta_img,
                "label": ruta_lbl,
            }
        )
    return registros


def _crear_compose_pre_crop(con_label: bool) -> Compose:
    """Carga, orientacion RAS, remuestreo isotropico XY y ventaneo CT."""
    keys = ["image", "label"] if con_label else ["image"]
    modos_spacing = ("bilinear", "nearest") if con_label else ("bilinear",)
    return Compose(
        [
            LoadImaged(keys=keys, image_only=True),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=TARGET_SPACING_3D, mode=modos_spacing),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=CT_CLIP_RANGE[0],
                a_max=CT_CLIP_RANGE[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )


def _crear_compose_crop(con_label: bool) -> Compose:
    """
    CropForegroundd aislado para poder capturar la shape post-Spacingd
    (necesaria para re-proyectar predicciones al espacio original).

    Nota: tras ScaleIntensityRanged el aire CT (HU ~ -1000) queda en
    ~0.017 (solo HU <= -1024 mapean exactamente a 0). El select_fn por
    defecto (x > 0) apenas recortaria; usamos un umbral equivalente a
    HU > -882 para que el crop realmente excluya el aire circundante.
    """
    keys = ["image", "label"] if con_label else ["image"]
    return Compose(
        [
            CropForegroundd(
                keys=keys,
                source_key="image",
                select_fn=lambda x: x > 0.1,
                allow_smaller=True,
            ),
        ]
    )


def _extraer_crop_start(meta_tensor, n_dims_spatial: int) -> tuple[int, ...]:
    """Lee el offset de CropForegroundd desde el meta dict (en orden espacial XYZ)."""
    meta = getattr(meta_tensor, "meta", None) or {}
    for k, v in meta.items():
        if k.endswith("foreground_start_coord"):
            coords = np.asarray(v).astype(int).tolist()
            return tuple(int(c) for c in coords[:n_dims_spatial])
    return tuple(0 for _ in range(n_dims_spatial))


def preprocess_case_paths(
    image_path: Path,
    label_path: Path | None = None,
    include_source_volume: bool = False,
) -> Case2D:
    con_label = label_path is not None
    compose_pre = _crear_compose_pre_crop(con_label=con_label)
    compose_crop = _crear_compose_crop(con_label=con_label)

    datos_in: dict[str, object] = {"image": str(image_path)}
    if con_label:
        datos_in["label"] = str(label_path)

    # Etapa 1: carga + orientacion + spacing + ventaneo.
    datos_pre = compose_pre(datos_in)

    # Shape y spacing originales (leidos del meta de LoadImaged).
    imagen_pre = datos_pre["image"]  # type: ignore
    meta = getattr(imagen_pre, "meta", {}) or {}
    spatial_shape_orig = meta.get("spatial_shape")
    if spatial_shape_orig is not None:
        arr_shape = np.asarray(spatial_shape_orig).tolist()
        original_shape_xyz = (int(arr_shape[0]), int(arr_shape[1]), int(arr_shape[2]))
    else:
        original_shape_xyz = (
            int(imagen_pre.shape[1]),
            int(imagen_pre.shape[2]),
            int(imagen_pre.shape[3]),
        )
    original_shape_zyx = (
        original_shape_xyz[2],
        original_shape_xyz[1],
        original_shape_xyz[0],
    )

    pixdim_orig = meta.get("pixdim")
    if pixdim_orig is not None:
        arr = np.asarray(pixdim_orig).astype(float).tolist()
        # pixdim NIfTI: ejes espaciales en 1:4 (X, Y, Z).
        if len(arr) >= 4:
            original_spacing_zyx = (float(arr[3]), float(arr[2]), float(arr[1]))
        else:
            original_spacing_zyx = (float(arr[2]), float(arr[1]), float(arr[0]))
    else:
        original_spacing_zyx = (
            float(TARGET_SPACING_3D[2]),
            float(TARGET_SPACING_3D[1]),
            float(TARGET_SPACING_3D[0]),
        )

    # Shape post-Spacingd pero pre-crop (tensor (C, X, Y, Z)).
    shape_post_spacing_xyz = (
        int(imagen_pre.shape[1]),
        int(imagen_pre.shape[2]),
        int(imagen_pre.shape[3]),
    )
    shape_post_spacing_zyx = (
        shape_post_spacing_xyz[2],
        shape_post_spacing_xyz[1],
        shape_post_spacing_xyz[0],
    )

    # Etapa 2: crop de foreground.
    datos_cropped = compose_crop(datos_pre)
    imagen_tensor = datos_cropped["image"]  # type: ignore
    imagen_xyz = imagen_tensor[0].detach().cpu().numpy().astype(np.float32)
    imagen_zyx = np.transpose(imagen_xyz, (2, 1, 0))

    label_zyx: np.ndarray | None = None
    if con_label:
        label_tensor = datos_cropped["label"]  # type: ignore
        label_xyz = label_tensor[0].detach().cpu().numpy()
        label_zyx = (np.transpose(label_xyz, (2, 1, 0)) > 0.5).astype(np.uint8)

    crop_start_xyz = _extraer_crop_start(imagen_tensor, n_dims_spatial=3)
    crop_start_zyx = (
        int(crop_start_xyz[2]),
        int(crop_start_xyz[1]),
        int(crop_start_xyz[0]),
    )

    spacing_zyx = (
        float(TARGET_SPACING_3D[2]),
        float(TARGET_SPACING_3D[1]),
        float(TARGET_SPACING_3D[0]),
    )

    # Reconstruir pseudo-HU para priors (HU_* thresholds viven en [-1024, 400]).
    pseudo_hu = (
        imagen_zyx.astype(np.float32) * (CT_CLIP_RANGE[1] - CT_CLIP_RANGE[0]) + CT_CLIP_RANGE[0]
    )

    priors = estimar_priores_pulmonares(
        volumen_hu=pseudo_hu,
        hu_lung_candidate_max=HU_LUNG_CANDIDATE_MAX,
        hu_healthy_range=HU_HEALTHY_LUNG_RANGE,
        hu_soft_tissue_range=HU_SOFT_TISSUE_RANGE,
        alarm_component_min_size=ALARM_COMPONENT_MIN_SIZE,
        alarm_slice_min_pixels=ALARM_SLICE_MIN_PIXELS,
    )

    positivos: np.ndarray | None = None
    if label_zyx is not None:
        positivos = np.flatnonzero(np.sum(label_zyx > 0, axis=(1, 2)) >= POSITIVE_SLICE_MIN_PIXELS)

    case_id = image_path.name.replace(".nii.gz", "")
    return Case2D(
        case_id=case_id,
        image=imagen_zyx.astype(np.float16),
        lung_mask=priors["lung_mask"].astype(np.uint8),
        healthy_lung_mask=priors["healthy_lung_mask"].astype(np.uint8),
        soft_tissue_alarm_mask=priors["soft_tissue_alarm_mask"].astype(np.uint8),
        alarm_pixels_per_slice=priors["alarm_pixels_per_slice"].astype(np.int32),
        label=label_zyx,
        positive_slices=positivos,
        alarm_slices=priors["alarm_slices"].astype(np.int32),
        spacing=spacing_zyx,
        original_shape=original_shape_zyx,
        original_spacing=original_spacing_zyx,
        crop_start=crop_start_zyx,
        shape_post_spacing=shape_post_spacing_zyx,
        source_volume_hu=pseudo_hu if include_source_volume else None,
    )


def _guardar_case_cache(case: Case2D) -> None:
    np.savez_compressed(
        _ruta_cache(case.case_id),
        image=case.image.astype(np.float16),
        lung_mask=case.lung_mask.astype(np.uint8),
        healthy_lung_mask=case.healthy_lung_mask.astype(np.uint8),
        soft_tissue_alarm_mask=case.soft_tissue_alarm_mask.astype(np.uint8),
        alarm_pixels_per_slice=case.alarm_pixels_per_slice.astype(np.int32),
        label=np.zeros((0,), dtype=np.uint8) if case.label is None else case.label.astype(np.uint8),
        spacing=np.asarray(case.spacing, dtype=np.float32),
        original_shape=np.asarray(case.original_shape, dtype=np.int32),
        original_spacing=np.asarray(case.original_spacing, dtype=np.float32),
        crop_start=np.asarray(case.crop_start, dtype=np.int32),
        shape_post_spacing=np.asarray(case.shape_post_spacing, dtype=np.int32),
        positive_slices=np.asarray([] if case.positive_slices is None else case.positive_slices),
        alarm_slices=np.asarray([] if case.alarm_slices is None else case.alarm_slices),
    )


def _cargar_case_cache(case_id: str) -> Case2D:
    datos = np.load(_ruta_cache(case_id))
    label = datos["label"]
    if label.size == 0:
        label = None
        positivos = None
    else:
        label = label.astype(np.uint8)
        positivos = datos["positive_slices"].astype(int)
    alarm_slices = datos["alarm_slices"].astype(int)
    if alarm_slices.size == 0:
        alarm_slices = None
    return Case2D(
        case_id=case_id,
        image=datos["image"].astype(np.float16),
        lung_mask=datos["lung_mask"].astype(np.uint8),
        healthy_lung_mask=datos["healthy_lung_mask"].astype(np.uint8),
        soft_tissue_alarm_mask=datos["soft_tissue_alarm_mask"].astype(np.uint8),
        alarm_pixels_per_slice=datos["alarm_pixels_per_slice"].astype(np.int32),
        label=label,
        positive_slices=positivos,
        alarm_slices=alarm_slices,
        spacing=(
            float(datos["spacing"][0]),
            float(datos["spacing"][1]),
            float(datos["spacing"][2]),
        ),
        original_shape=(
            int(datos["original_shape"][0]),
            int(datos["original_shape"][1]),
            int(datos["original_shape"][2]),
        ),
        original_spacing=(
            float(datos["original_spacing"][0]),
            float(datos["original_spacing"][1]),
            float(datos["original_spacing"][2]),
        ),
        crop_start=(
            int(datos["crop_start"][0]),
            int(datos["crop_start"][1]),
            int(datos["crop_start"][2]),
        ),
        shape_post_spacing=(
            int(datos["shape_post_spacing"][0]),
            int(datos["shape_post_spacing"][1]),
            int(datos["shape_post_spacing"][2]),
        ),
    )


def cargar_casos_preprocesados(
    force_rebuild: bool = False,
    limit: int | None = None,
) -> list[Case2D]:
    casos = []
    registros = leer_registros_training()
    if limit is not None:
        registros = registros[:limit]

    for registro in registros:
        case_id = str(registro["case_id"])
        cache_path = _ruta_cache(case_id)
        if cache_path.exists() and not force_rebuild:
            casos.append(_cargar_case_cache(case_id))
            continue

        case = preprocess_case_paths(
            image_path=Path(registro["image"]),
            label_path=Path(registro["label"]),
        )
        _guardar_case_cache(case)
        casos.append(case)
    return casos


# =============================================================================
# Validacion cruzada K-Fold
# =============================================================================
def obtener_splits_kfold(
    casos: list[Case2D],
    n_splits: int = N_FOLDS_KFOLD,
    seed: int = SEMILLA,
) -> list[tuple[list[Case2D], list[Case2D]]]:
    """Devuelve la lista de folds (train, val) para K-Fold determinista."""
    if len(casos) < n_splits:
        raise ValueError(f"No hay suficientes casos ({len(casos)}) para {n_splits} folds.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(casos))
    folds: list[tuple[list[Case2D], list[Case2D]]] = []
    for train_idx, val_idx in kf.split(indices):
        train = [casos[i] for i in train_idx]
        val = [casos[i] for i in val_idx]
        folds.append((train, val))
    return folds


def split_por_fold(
    casos: list[Case2D],
    fold_idx: int = KFOLD_DEFAULT_INDEX,
    n_splits: int = N_FOLDS_KFOLD,
    seed: int = SEMILLA,
) -> tuple[list[Case2D], list[Case2D]]:
    """Devuelve el split (train, val) correspondiente a un fold concreto."""
    folds = obtener_splits_kfold(casos, n_splits=n_splits, seed=seed)
    if not 0 <= fold_idx < len(folds):
        raise IndexError(f"fold_idx={fold_idx} fuera de rango (0..{len(folds) - 1}).")
    return folds[fold_idx]


def seleccionar_subconjunto_casos(
    casos: list[Case2D],
    fraccion: float,
    seed: int = SEMILLA,
) -> list[Case2D]:
    if fraccion >= 1.0:
        return list(casos)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(casos))
    rng.shuffle(indices)
    n = max(1, round(len(casos) * fraccion))
    idx_keep = set(indices[:n].tolist())
    return [caso for i, caso in enumerate(casos) if i in idx_keep]


def resumir_casos(casos: list[Case2D]) -> pd.DataFrame:
    filas = []
    for caso in casos:
        if caso.label is None:
            continue
        positivos = (
            caso.positive_slices.astype(int).tolist() if caso.positive_slices is not None else []
        )
        alarmas = caso.alarm_slices.astype(int).tolist() if caso.alarm_slices is not None else []
        tumor_voxels = int(np.sum(caso.label > 0))
        filas.append(
            {
                "case_id": caso.case_id,
                "slices_totales": int(caso.image.shape[0]),
                "slices_positivos": len(positivos),
                "slices_alarma": len(alarmas),
                "ratio_slices_positivos": round(
                    float(len(positivos) / max(caso.image.shape[0], 1)),
                    4,
                ),
                "ratio_slices_alarma": round(
                    float(len(alarmas) / max(caso.image.shape[0], 1)),
                    4,
                ),
                "voxeles_tumor": tumor_voxels,
                "volumen_tumor_cm3": round(tumor_voxels * np.prod(caso.spacing) / 1000.0, 3),
                "voxeles_pulmon": int(np.sum(caso.lung_mask > 0)),
                "voxeles_pulmon_sano": int(np.sum(caso.healthy_lung_mask > 0)),
                "voxeles_alarma_tejido_blando": int(np.sum(caso.soft_tissue_alarm_mask > 0)),
            }
        )
    return pd.DataFrame(filas)


def extraer_contexto(volumen: np.ndarray, z: int) -> np.ndarray:
    indices = [
        int(np.clip(z + offset, 0, volumen.shape[0] - 1))
        for offset in range(-CONTEXT_SLICES, CONTEXT_SLICES + 1)
    ]
    contexto = volumen[indices].astype(np.float32)
    return contexto


def extraer_priores(caso: Case2D, z: int) -> np.ndarray:
    priors = np.stack(
        [
            caso.lung_mask[z],
            caso.healthy_lung_mask[z],
            caso.soft_tissue_alarm_mask[z],
        ],
        axis=0,
    ).astype(np.float32)
    return priors


def construir_entrada_modelo(caso: Case2D, z: int) -> np.ndarray:
    entrada = np.concatenate(
        [extraer_contexto(caso.image, z), extraer_priores(caso, z)],
        axis=0,
    ).astype(np.float32)
    if entrada.shape[0] != INPUT_CHANNELS_2D:
        raise ValueError(
            f"Se esperaban {INPUT_CHANNELS_2D} canales y se obtuvieron {entrada.shape[0]}."
        )
    return entrada


def crear_manifest_entrenamiento(
    casos: list[Case2D],
    negative_ratio: float = NEGATIVE_TO_POSITIVE_RATIO,
    seed: int = SEMILLA,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    manifest: list[tuple[int, int]] = []

    for case_idx, caso in enumerate(casos):
        if caso.label is None or caso.positive_slices is None:
            continue

        positivos = caso.positive_slices.astype(int).tolist()
        if not positivos:
            continue

        pos_set = set(positivos)
        alarm_set = (
            set(caso.alarm_slices.astype(int).tolist()) if caso.alarm_slices is not None else set()
        )
        z_min = max(0, min(positivos) - HARD_NEGATIVE_MARGIN)
        z_max = min(caso.image.shape[0], max(positivos) + HARD_NEGATIVE_MARGIN + 1)
        hard_neg = [z for z in range(z_min, z_max) if z not in pos_set and z not in alarm_set]
        pool_neg = [
            z
            for z in range(caso.image.shape[0])
            if z not in pos_set
            and (caso.alarm_pixels_per_slice[z] < ALARM_NEGATIVE_EXCLUSION_PIXELS)
            and float(np.mean(caso.lung_mask[z])) > 0.01
        ]

        rng.shuffle(hard_neg)
        rng.shuffle(pool_neg)

        n_neg = math.ceil(len(positivos) * negative_ratio)
        n_hard = min(len(hard_neg), max(1, n_neg // 2))
        elegidos = hard_neg[:n_hard]
        usados = set(elegidos)
        restantes = [z for z in pool_neg if z not in usados]
        elegidos.extend(restantes[: max(0, n_neg - len(elegidos))])

        manifest.extend((case_idx, z) for z in positivos)
        manifest.extend((case_idx, z) for z in elegidos)

    rng.shuffle(manifest)
    return manifest


# =============================================================================
# Extraccion 2D de parches (pad + crop guiado)
# =============================================================================
def _pad_2d_to_patch(
    image: np.ndarray,
    label: np.ndarray,
    reference_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Rellena al minimo necesario para poder extraer PATCH_SIZE_2D sin error."""
    patch_h, patch_w = PATCH_SIZE_2D
    _, height, width = image.shape

    pad_h = max(0, patch_h - height)
    pad_w = max(0, patch_w - width)
    if pad_h == 0 and pad_w == 0:
        return image, label, reference_mask

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_spec_img = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    pad_spec_mask = ((pad_top, pad_bottom), (pad_left, pad_right))

    image = np.pad(image, pad_spec_img, mode="constant")
    label = np.pad(label, pad_spec_img, mode="constant")
    if reference_mask is not None:
        reference_mask = np.pad(reference_mask, pad_spec_mask, mode="constant")

    return image, label, reference_mask


def _center_patch_origin(height: int, width: int) -> tuple[int, int]:
    patch_h, patch_w = PATCH_SIZE_2D
    max_top = max(height - patch_h, 0)
    max_left = max(width - patch_w, 0)
    return max_top // 2, max_left // 2


def _sample_patch_origin_from_mask(mask: np.ndarray) -> tuple[int, int]:
    """
    Devuelve el origen (top, left) de un parche PATCH_SIZE_2D.

    Si la ROI cabe completa en el parche, se muestrea un origen que la cubra
    entera. Si no cabe, se centra el parche en un pixel activo aleatorio para
    seguir exponiendo tejido relevante en vez de aire o borde corporal.
    """
    patch_h, patch_w = PATCH_SIZE_2D
    height, width = mask.shape
    max_top = max(height - patch_h, 0)
    max_left = max(width - patch_w, 0)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return _center_patch_origin(height, width)

    top_min = max(0, int(ys.max()) - patch_h + 1)
    top_max = min(int(ys.min()), max_top)
    left_min = max(0, int(xs.max()) - patch_w + 1)
    left_max = min(int(xs.min()), max_left)

    if top_min <= top_max and left_min <= left_max:
        top = int(np.random.randint(top_min, top_max + 1))
        left = int(np.random.randint(left_min, left_max + 1))
        return top, left

    pick = int(np.random.randint(len(ys)))
    center_y = int(ys[pick])
    center_x = int(xs[pick])
    top = int(np.clip(center_y - patch_h // 2, 0, max_top))
    left = int(np.clip(center_x - patch_w // 2, 0, max_left))
    return top, left


def _extraer_patch_2d(
    image: np.ndarray,
    label: np.ndarray,
    reference_mask: np.ndarray | None,
    train: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrae un parche fijo. En train usa una ROI guiada; en eval, centro."""
    image, label, reference_mask = _pad_2d_to_patch(image, label, reference_mask)
    height, width = image.shape[1:]

    if train and reference_mask is not None and np.any(reference_mask > 0):
        top, left = _sample_patch_origin_from_mask(reference_mask)
    else:
        top, left = _center_patch_origin(height, width)

    patch_h, patch_w = PATCH_SIZE_2D
    image_patch = image[:, top : top + patch_h, left : left + patch_w]
    label_patch = label[:, top : top + patch_h, left : left + patch_w]
    return image_patch, label_patch


class SliceDataset(Dataset):
    def __init__(
        self,
        casos: list[Case2D],
        manifest: list[tuple[int, int]],
        augment_level: int = 0,
        train: bool = True,
    ) -> None:
        self.casos = casos
        self.manifest = manifest
        self.augment_level = augment_level
        self.train = train

    def __len__(self) -> int:
        return len(self.manifest)

    def _augment(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Augmentaciones geometricas e intensimetricas.

        Nivel 1 (Base): flips y rotaciones.
        Nivel 2 (Fuerte): nivel 1 + ruido gaussiano + shift de brillo +
            gamma augmentation + deformacion elastica 2D.
        """
        if self.augment_level <= 0 or not self.train:
            return image, label

        # --- Nivel 1: geometricas ---
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=2)
            label = np.flip(label, axis=2)
        if np.random.rand() < 0.5:
            k = int(np.random.randint(0, 4))
            image = np.rot90(image, k=k, axes=(1, 2))
            label = np.rot90(label, k=k, axes=(1, 2))

        if self.augment_level >= 2:
            # --- Intensimetricos basicos ---
            if np.random.rand() < 0.3:
                image = np.clip(
                    image + np.random.normal(0.0, 0.02, size=image.shape),
                    0.0,
                    1.0,
                )
            if np.random.rand() < 0.3:
                shift = float(np.random.uniform(-0.08, 0.08))
                image = np.clip(image + shift, 0.0, 1.0)
            if np.random.rand() < 0.25:
                gain = float(np.random.uniform(0.9, 1.15))
                image = np.clip((image - 0.5) * gain + 0.5, 0.0, 1.0)

            # --- Gamma augmentation ---
            # Simula variabilidad de escaner entre centros. Solo se aplica
            # a los canales de intensidad CT, no a los priors anatomicos.
            if np.random.rand() < 0.3:
                gamma = float(np.random.uniform(0.7, 1.5))
                n_img_channels = 2 * CONTEXT_SLICES + 1
                image[:n_img_channels] = np.clip(image[:n_img_channels] ** gamma, 0.0, 1.0)

            # --- Deformacion elastica 2D ---
            # Simula deformaciones por respiracion y posicion del paciente.
            # sigma controla la suavidad del campo, alpha la magnitud.
            if np.random.rand() < 0.25:
                from scipy.ndimage import gaussian_filter, map_coordinates

                H, W = image.shape[1], image.shape[2]
                sigma = 12.0
                alpha = 120.0
                dx = gaussian_filter(np.random.randn(H, W), sigma) * alpha
                dy = gaussian_filter(np.random.randn(H, W), sigma) * alpha
                x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
                coords_x = np.clip(x_grid + dx, 0, W - 1).ravel()
                coords_y = np.clip(y_grid + dy, 0, H - 1).ravel()
                coords = [coords_y, coords_x]
                # Interpolar imagen con orden 1 (bilineal) y label con orden 0.
                for c in range(image.shape[0]):
                    image[c] = map_coordinates(image[c], coords, order=1, mode="reflect").reshape(
                        H, W
                    )
                label[0] = (
                    map_coordinates(
                        label[0].astype(np.float32), coords, order=0, mode="reflect"
                    ).reshape(H, W)
                    > 0.5
                ).astype(np.float32)

        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        case_idx, z = self.manifest[idx]
        caso = self.casos[case_idx]
        if caso.label is None:
            raise ValueError(f"El caso {caso.case_id} no tiene mascara cargada.")

        image = construir_entrada_modelo(caso, z)  # (C, H, W)
        label = caso.label[z][None].astype(np.float32)  # (1, H, W)

        if self.train and np.any(label > 0):
            reference_mask = label[0] > 0
        elif self.train and np.any(caso.soft_tissue_alarm_mask[z] > 0):
            reference_mask = caso.soft_tissue_alarm_mask[z] > 0
        elif self.train and np.any(caso.lung_mask[z] > 0):
            reference_mask = caso.lung_mask[z] > 0
        else:
            reference_mask = None

        image_t, label_t = _extraer_patch_2d(
            image=image,
            label=label,
            reference_mask=reference_mask,
            train=self.train,
        )

        image_t, label_t = self._augment(image_t, label_t)
        return {
            "image": torch.from_numpy(image_t.astype(np.float32)),
            "label": torch.from_numpy(label_t.astype(np.float32)),
            "case_id": caso.case_id,
            "slice_idx": int(z),
        }


def crear_modelo_2d() -> UNet:
    """
    UNet 2D con capacidad doblada respecto a v3.

    channels: (32, 64, 128, 256, 512) frente a los (16, 32, 64, 128, 256)
    anteriores. El mayor numero de filtros mejora la representacion de
    caracteristicas texturales finas del tumor sin cambiar el flujo de datos.
    INPUT_CHANNELS_2D = 8 (5 de contexto + 3 de priors).
    """
    return UNet(
        spatial_dims=2,
        in_channels=INPUT_CHANNELS_2D,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    )


def contar_parametros(modelo: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in modelo.parameters()))


def estimar_pos_weight(
    casos: list[Case2D],
    manifest: list[tuple[int, int]],
) -> float:
    """
    Calcula el ratio negativos/positivos a nivel de pixel en el manifest.
    Se mantiene como referencia para logging aunque ya no se usa en la loss.
    """
    positivos = 0
    total = 0
    for case_idx, z in manifest:
        mascara = casos[case_idx].label
        if mascara is None:
            continue
        slice_mask = mascara[z]
        positivos += int(np.sum(slice_mask > 0))
        total += int(slice_mask.size)
    negativos = max(total - positivos, 1)
    if positivos == 0:
        return 1.0
    return float(np.clip(negativos / positivos, 1.0, 25.0))


def postprocesar_mascara_predicha(
    pred: np.ndarray,
    caso: Case2D,
    cleanup: bool = True,
) -> np.ndarray:
    """Aplica el postproceso minimo defendible: prior pulmonar y componentes."""
    pred_proc = ((pred > 0) & (caso.lung_mask > 0)).astype(np.uint8)
    if cleanup:
        pred_proc = limpiar_componentes_pequenas(pred_proc, min_size=MIN_COMPONENT_SIZE)
    return pred_proc.astype(np.uint8)


def inferir_probabilidades_volumen(
    modelo: torch.nn.Module,
    caso: Case2D,
    batch_size: int = INFER_BATCH_SIZE,
) -> np.ndarray:
    """
    Inferencia probabilistica slice a slice con sliding_window_inference.

    Tras CropForegroundd los cortes 2.5D tienen tamano variable. El modelo
    se entrena con parches PATCH_SIZE_2D, asi que usamos sliding window
    2D por corte para cubrir el FoV completo sin distorsion.

    Devolver probabilidades permite buscar el threshold sin re-ejecutar la red.
    """
    modelo.eval()
    probs: list[np.ndarray] = []
    n_z = caso.image.shape[0]

    with torch.no_grad():
        for inicio in range(0, n_z, batch_size):
            fin = min(n_z, inicio + batch_size)
            batch_np = np.stack(
                [construir_entrada_modelo(caso, z) for z in range(inicio, fin)],
                axis=0,
            ).astype(np.float32)
            batch = torch.from_numpy(batch_np).to(DEVICE)
            logits = sliding_window_inference(
                inputs=batch,
                roi_size=PATCH_SIZE_2D,
                sw_batch_size=max(1, batch.shape[0]),
                predictor=modelo,
                overlap=0.5,
                mode="gaussian",
                padding_mode="constant",
            )
            prob = torch.sigmoid(logits).cpu().numpy()[:, 0]  # type: ignore
            probs.append(prob.astype(np.float32))

    return np.concatenate(probs, axis=0)


def inferir_volumen(
    modelo: torch.nn.Module,
    caso: Case2D,
    threshold: float = 0.5,
    cleanup: bool = True,
    batch_size: int = INFER_BATCH_SIZE,
) -> np.ndarray:
    prob = inferir_probabilidades_volumen(modelo, caso, batch_size=batch_size)
    pred = (prob >= threshold).astype(np.uint8)
    return postprocesar_mascara_predicha(pred, caso, cleanup=cleanup)


def inferir_probabilidades_volumen_tta(
    modelo: torch.nn.Module,
    caso: Case2D,
    batch_size: int = INFER_BATCH_SIZE,
) -> np.ndarray:
    """
    Probabilidades con Test-Time Augmentation (TTA).

    Promedia tres pasadas: original, flip horizontal y flip vertical. El flip se
    deshace antes del promedio, por lo que no introduce sesgo geometrico.
    """

    def _inferir_probs(flip_h: bool = False, flip_v: bool = False) -> np.ndarray:
        modelo.eval()
        probs_list: list[np.ndarray] = []
        n_z = caso.image.shape[0]
        with torch.no_grad():
            for inicio in range(0, n_z, batch_size):
                fin = min(n_z, inicio + batch_size)
                batch_np = np.stack(
                    [construir_entrada_modelo(caso, z) for z in range(inicio, fin)],
                    axis=0,
                ).astype(np.float32)
                if flip_h:
                    batch_np = np.flip(batch_np, axis=-1).copy()
                if flip_v:
                    batch_np = np.flip(batch_np, axis=-2).copy()
                batch = torch.from_numpy(batch_np).to(DEVICE)
                logits = sliding_window_inference(
                    inputs=batch,
                    roi_size=PATCH_SIZE_2D,
                    sw_batch_size=max(1, batch.shape[0]),
                    predictor=modelo,
                    overlap=0.5,
                    mode="gaussian",
                    padding_mode="constant",
                )
                prob = torch.sigmoid(logits).cpu().numpy()[:, 0]  # type: ignore
                if flip_h:
                    prob = np.flip(prob, axis=-1).copy()
                if flip_v:
                    prob = np.flip(prob, axis=-2).copy()
                probs_list.append(prob.astype(np.float32))
        return np.concatenate(probs_list, axis=0)

    prob_orig = _inferir_probs(flip_h=False, flip_v=False)
    prob_fh = _inferir_probs(flip_h=True, flip_v=False)
    prob_fv = _inferir_probs(flip_h=False, flip_v=True)
    return ((prob_orig + prob_fh + prob_fv) / 3.0).astype(np.float32)


def inferir_volumen_tta(
    modelo: torch.nn.Module,
    caso: Case2D,
    threshold: float = 0.5,
    cleanup: bool = True,
    batch_size: int = INFER_BATCH_SIZE,
) -> np.ndarray:
    prob = inferir_probabilidades_volumen_tta(modelo, caso, batch_size=batch_size)
    pred = (prob >= threshold).astype(np.uint8)
    return postprocesar_mascara_predicha(pred, caso, cleanup=cleanup)


def evaluar_casos(
    modelo: torch.nn.Module,
    casos: list[Case2D],
    full_metrics: bool = True,
    cleanup: bool = True,
    use_tta: bool = False,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Evalua el modelo sobre una lista de casos.

    Args:
        use_tta: Si True usa inferir_volumen_tta en lugar de inferir_volumen.
        threshold: Umbral de binarizacion. Permite pasar el threshold optimo
            encontrado en fase4 sin modificar la funcion de inferencia.
    """
    filas = []
    _infer_fn = inferir_volumen_tta if use_tta else inferir_volumen
    for caso in casos:
        if caso.label is None:
            continue
        pred = _infer_fn(modelo, caso, threshold=threshold, cleanup=cleanup)
        if full_metrics:
            met = evaluar_segmentacion(pred, caso.label, caso.spacing)
        else:
            met = {"dice": round(dice_score(pred, caso.label), 4)}
        met["case_id"] = caso.case_id
        filas.append(met)
    return pd.DataFrame(filas)


def _mean_val_dice(modelo: torch.nn.Module, casos_val: list[Case2D]) -> float:
    dices = []
    for caso in casos_val:
        if caso.label is None:
            continue
        pred = inferir_volumen(modelo, caso, cleanup=False)
        dices.append(dice_score(pred, caso.label))
    if not dices:
        return 0.0
    return float(np.mean(dices))


def entrenar_modelo(
    casos_train: list[Case2D],
    casos_val: list[Case2D],
    augment_level: int,
    num_epocas: int,
    checkpoint_path: Path | None,
    descripcion: str | None = None,
    fold_idx: int | None = None,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """
    Entrena el UNet 2D con las siguientes mejoras respecto a v3:

    - DiceFocalLoss (gamma=2) en lugar de Dice+BCE. La focal loss penaliza
      los pixeles que el modelo clasifica mal de forma consistente, lo que
      es especialmente util con desbalance extremo de clases.
    - LR warmup: 3 epocas de subida lineal antes del cosine annealing.
      Evita divergencias tempranas cuando el LR inicial es relativamente alto.
    - Re-muestreo del manifest por epoca: los negativos se vuelven a muestrear
      con una semilla diferente en cada epoca, aumentando la variedad de
      ejemplos negativos que ve el modelo sin incrementar el coste por epoch.
    - Gradient clipping (max_norm=1.0): previene la explosion de gradiente,
      especialmente relevante en las primeras epocas con el warmup activo.
    """
    modelo = crear_modelo_2d().to(DEVICE)
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Scheduler: warmup lineal LR_WARMUP_EPOCAS epocas + cosine annealing.
    warmup_epocas = min(LR_WARMUP_EPOCAS, num_epocas)
    cosine_epocas = max(num_epocas - warmup_epocas, 1)
    sched_warmup = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epocas,
    )
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cosine_epocas,
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[sched_warmup, sched_cosine],
        milestones=[warmup_epocas],
    )

    # DiceFocalLoss: combina Dice y Focal en una sola funcion.
    # Pesos expuestos via env (MONAI_LOSS_LAMBDA_DICE/FOCAL/GAMMA) para
    # poder barrer sin tocar codigo. Defaults: lambda 0.5/0.5, gamma=2.0.
    criterio = DiceFocalLoss(
        sigmoid=True,
        gamma=LOSS_FOCAL_GAMMA,
        lambda_dice=LOSS_LAMBDA_DICE,
        lambda_focal=LOSS_LAMBDA_FOCAL,
    )
    scaler = GradScaler(DEVICE.type, enabled=AMP_ENABLED)

    # Calcular pos_weight del primer manifest para logging.
    manifest_inicial = crear_manifest_entrenamiento(casos_train, seed=SEMILLA)
    pos_weight_log = estimar_pos_weight(casos_train, manifest_inicial)

    historia: dict[str, object] = {
        "descripcion": descripcion or MODELO_2D_NOMBRE,
        "train_loss": [],
        "val_dice": [],
        "pos_weight_log": round(pos_weight_log, 4),
        "train_slices": len(manifest_inicial),
        "n_params": contar_parametros(modelo),
        "prior_channels": PRIOR_CHANNELS_2D,
        "context_slices": CONTEXT_SLICES,
        "negative_alarm_exclusion_pixels": ALARM_NEGATIVE_EXCLUSION_PIXELS,
        "fold_idx": fold_idx,
        "n_folds": N_FOLDS_KFOLD,
        "warmup_epocas": warmup_epocas,
        "loss_fn": (
            f"DiceFocalLoss(gamma={LOSS_FOCAL_GAMMA},"
            f" lambda_dice={LOSS_LAMBDA_DICE},"
            f" lambda_focal={LOSS_LAMBDA_FOCAL})"
        ),
        "model_channels": "(32,64,128,256,512)",
    }

    mejor_dice = -1.0
    mejor_epoca = 0
    paciencia = 0
    mejor_state = None
    t0 = time.time()

    for epoca in range(1, num_epocas + 1):
        modelo.train()
        loss_epoca = 0.0
        n_batches = 0

        # Re-muestreo de negativos por epoca: semilla diferente en cada vuelta.
        # Los positivos siempre se incluyen todos; solo cambian los negativos.
        manifest_ep = crear_manifest_entrenamiento(casos_train, seed=SEMILLA + epoca)
        dataset_ep = SliceDataset(casos_train, manifest_ep, augment_level=augment_level, train=True)
        loader = DataLoader(
            dataset_ep,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=N_WORKERS_DATALOADER,
            pin_memory=AMP_ENABLED,
        )

        for batch in loader:
            imagenes = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                logits = modelo(imagenes)
                loss = criterio(logits, labels)
            scaler.scale(loss).backward()
            # Gradient clipping: previene explosion de gradiente.
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            loss_epoca += float(loss.item())
            n_batches += 1

        sched.step()
        avg_loss = loss_epoca / max(n_batches, 1)
        cast_list = historia["train_loss"]
        if isinstance(cast_list, list):
            cast_list.append(round(avg_loss, 4))

        if epoca % VAL_INTERVAL == 0 or epoca == num_epocas:
            val_dice = _mean_val_dice(modelo, casos_val)
            cast_val = historia["val_dice"]
            if isinstance(cast_val, list):
                cast_val.append(round(val_dice, 4))

            if val_dice > mejor_dice:
                mejor_dice = val_dice
                mejor_epoca = epoca
                paciencia = 0
                mejor_state = {
                    clave: valor.detach().cpu().clone()
                    for clave, valor in modelo.state_dict().items()
                }
                if checkpoint_path is not None:
                    torch.save(mejor_state, checkpoint_path)
            else:
                paciencia += 1

            lr_actual = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else LR
            print(
                f"  ep {epoca:>2d}/{num_epocas} | "
                f"loss {avg_loss:.4f} | val_dice {val_dice:.4f} | "
                f"mejor {mejor_dice:.4f} | lr {lr_actual:.2e}"
            )

            if paciencia >= EARLY_STOPPING_PATIENCE:
                print(f"  early stopping en epoca {epoca}")
                break

    if mejor_state is not None:
        modelo.load_state_dict(mejor_state)

    historia["mejor_dice"] = round(max(mejor_dice, 0.0), 4)  # type: ignore
    historia["mejor_epoca"] = mejor_epoca  # type: ignore
    historia["tiempo_total_s"] = round(time.time() - t0, 1)  # type: ignore
    return modelo, historia


def entrenar_fold(
    casos: list[Case2D],
    fold_idx: int,
    augment_level: int,
    num_epocas: int,
    checkpoint_path: Path | None,
    descripcion: str | None = None,
    n_splits: int = N_FOLDS_KFOLD,
    seed: int = SEMILLA,
) -> tuple[torch.nn.Module, dict[str, object], list[Case2D], list[Case2D]]:
    """Selecciona el fold K-Fold indicado y entrena sobre el."""
    train_cases, val_cases = split_por_fold(casos, fold_idx=fold_idx, n_splits=n_splits, seed=seed)
    desc = descripcion if descripcion is not None else f"{MODELO_2D_NOMBRE} (fold {fold_idx})"
    modelo, historia = entrenar_modelo(
        casos_train=train_cases,
        casos_val=val_cases,
        augment_level=augment_level,
        num_epocas=num_epocas,
        checkpoint_path=checkpoint_path,
        descripcion=desc,
        fold_idx=fold_idx,
    )
    return modelo, historia, train_cases, val_cases


def cargar_modelo_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
    modelo = crear_modelo_2d().to(DEVICE)
    modelo.load_state_dict(torch.load(str(checkpoint_path), map_location=DEVICE))
    modelo.eval()
    return modelo


def restaurar_mascara_resolucion_original(
    mascara_proc: np.ndarray,
    original_shape: tuple[int, int, int],
    crop_start: tuple[int, int, int] | None = None,
    shape_post_spacing: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """
    Deshace el crop + resample para proyectar una mascara procesada al
    espacio del NIfTI original.

    Si no se aporta metadata de crop se asume que la mascara cubre el
    volumen completo tras Spacingd (comportamiento legado).
    """
    if crop_start is None or shape_post_spacing is None:
        full = mascara_proc
    else:
        full = np.zeros(shape_post_spacing, dtype=np.uint8)
        z0, y0, x0 = crop_start
        z1 = z0 + mascara_proc.shape[0]
        y1 = y0 + mascara_proc.shape[1]
        x1 = x0 + mascara_proc.shape[2]
        full[z0:z1, y0:y1, x0:x1] = mascara_proc.astype(np.uint8)

    zoom = (
        original_shape[0] / full.shape[0],
        original_shape[1] / full.shape[1],
        original_shape[2] / full.shape[2],
    )
    restaurada = ndimage.zoom(full.astype(np.float32), zoom=zoom, order=0)
    return np.asarray(restaurada > 0.5, dtype=np.uint8)

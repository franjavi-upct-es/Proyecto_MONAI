"""
Utilidades comunes del pipeline 2D.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from monai.metrics import compute_hausdorff_distance
from scipy import ndimage

ESTILO_OSCURO = {
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "grid.color": "#334155",
    "grid.alpha": 0.25,
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.titlesize": 15,
    "legend.facecolor": "#111827",
    "legend.edgecolor": "#334155",
    "text.usetex": False,
}


def estilo_oscuro() -> None:
    plt.rcParams.update(ESTILO_OSCURO)


def ventanear(imagen: np.ndarray, width: float, level: float) -> np.ndarray:
    hu_min = level - width / 2.0
    hu_max = level + width / 2.0
    return np.clip((imagen - hu_min) / max(hu_max - hu_min, 1e-8), 0.0, 1.0)


def guardar_json(datos, ruta: Path) -> None:
    with open(ruta, "w", encoding="utf-8") as handle:
        json.dump(datos, handle, indent=2, ensure_ascii=False, default=str)
    print(f"  -> {ruta.name}")


def guardar_csv(df: pd.DataFrame, ruta: Path) -> None:
    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"  -> {ruta.name}")


def guardar_figura(fig: Figure, ruta: Path, dpi: int = 150) -> None:
    fig.savefig(
        ruta,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    print(f"  -> {ruta.name}")


def mantener_componentes_mas_grandes(
    mascara: np.ndarray,
    max_componentes: int = 2,
    min_size: int = 0,
) -> np.ndarray:
    labels, n = ndimage.label(mascara > 0)
    if n == 0:
        return np.zeros_like(mascara, dtype=np.uint8)

    tamanos = ndimage.sum(mascara > 0, labels, range(1, n + 1))
    orden = np.argsort(tamanos)[::-1]
    salida = np.zeros_like(mascara, dtype=np.uint8)

    kept = 0
    for idx in orden:
        etiqueta = idx + 1
        tam = int(tamanos[idx])
        if tam < min_size:
            continue
        salida[labels == etiqueta] = 1
        kept += 1
        if kept >= max_componentes:
            break
    return salida


def construir_mascara_cuerpo(volumen_hu: np.ndarray) -> np.ndarray:
    """
    Estima la silueta corporal slice a slice para separar el aire externo del
    interior toracico.
    """
    cuerpo = volumen_hu > -600
    salida = np.zeros_like(cuerpo, dtype=np.uint8)
    estructura = np.ones((5, 5), dtype=bool)

    for z in range(cuerpo.shape[0]):
        sl = cuerpo[z]
        labels, n = ndimage.label(sl)
        if n > 0:
            tamanos = ndimage.sum(sl, labels, range(1, n + 1))
            sl = labels == (int(np.argmax(tamanos)) + 1)
        sl = ndimage.binary_closing(sl, structure=estructura)
        sl = ndimage.binary_fill_holes(sl)
        salida[z] = sl.astype(np.uint8)
    return salida


def estimar_priores_pulmonares(
    volumen_hu: np.ndarray,
    hu_lung_candidate_max: float,
    hu_healthy_range: tuple[float, float],
    hu_soft_tissue_range: tuple[float, float],
    alarm_component_min_size: int,
    alarm_slice_min_pixels: int,
) -> dict[str, np.ndarray]:
    """
    Genera priors anatomicos y heuristicas de alarma:
      - lung_mask: region intrapulmonar esperada
      - healthy_lung_mask: parenquima bien aireado
      - soft_tissue_alarm_mask: tejido mas denso dentro del pulmon

    La mascara de alarma es deliberadamente sensible: no pretende sustituir la
    etiqueta, sino evitar que el modelo trate tejido sospechoso intrapulmonar
    como fondo indiscutible.
    """
    cuerpo = construir_mascara_cuerpo(volumen_hu) > 0
    lung = (volumen_hu < hu_lung_candidate_max) & cuerpo
    lung = ndimage.binary_opening(lung, structure=np.ones((1, 3, 3), dtype=bool))
    lung = ndimage.binary_closing(lung, structure=np.ones((1, 5, 5), dtype=bool))
    lung = mantener_componentes_mas_grandes(lung.astype(np.uint8), max_componentes=2) > 0

    lung_filled = np.zeros_like(lung, dtype=bool)
    for z in range(lung.shape[0]):
        lung_filled[z] = ndimage.binary_fill_holes(lung[z])
    lung = lung_filled

    hu_healthy_min, hu_healthy_max = hu_healthy_range
    healthy = lung & (volumen_hu >= hu_healthy_min) & (volumen_hu <= hu_healthy_max)

    inner_lung = ndimage.binary_erosion(lung, structure=np.ones((1, 3, 3), dtype=bool))
    hu_soft_min, hu_soft_max = hu_soft_tissue_range
    alarma = inner_lung & (volumen_hu >= hu_soft_min) & (volumen_hu <= hu_soft_max)
    alarma &= ~healthy
    alarma = ndimage.binary_opening(alarma, structure=np.ones((1, 3, 3), dtype=bool))
    alarma = ndimage.binary_closing(alarma, structure=np.ones((1, 3, 3), dtype=bool))
    alarma = (
        mantener_componentes_mas_grandes(
            limpiar_componentes_pequenas(
                alarma.astype(np.uint8),
                min_size=alarm_component_min_size,
                max_componentes=128,
            ),
            max_componentes=128,
            min_size=alarm_component_min_size,
        )
        > 0
    )

    alarm_pixels = np.sum(alarma, axis=(1, 2))
    alarm_slices = np.flatnonzero(alarm_pixels >= alarm_slice_min_pixels)

    return {
        "lung_mask": lung.astype(np.uint8),
        "healthy_lung_mask": healthy.astype(np.uint8),
        "soft_tissue_alarm_mask": alarma.astype(np.uint8),
        "alarm_slices": alarm_slices.astype(np.int32),
        "alarm_pixels_per_slice": alarm_pixels.astype(np.int32),
    }


def seleccionar_indices_tumor(mascara: np.ndarray, n: int = 6) -> list[int]:
    positivos = np.flatnonzero(np.sum(mascara > 0, axis=(1, 2)) > 0)
    if len(positivos) == 0:
        centro = mascara.shape[0] // 2
        return [centro]
    if len(positivos) <= n:
        return positivos.tolist()
    idx = np.linspace(0, len(positivos) - 1, n, dtype=int)
    return positivos[idx].tolist()


def crear_montaje_slices(
    volumen: np.ndarray,
    mascara: np.ndarray | None = None,
    indices: list[int] | None = None,
    titulo: str = "",
    ventana: tuple[float, float] = (600, 40),
) -> Figure:
    estilo_oscuro()
    if indices is None:
        indices = [volumen.shape[0] // 2]

    n = len(indices)
    cols = min(3, n)
    filas = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        filas,
        cols,
        figsize=(5 * cols, 5 * filas),
        squeeze=False,
        facecolor="#0f172a",
    )
    if titulo:
        fig.suptitle(titulo, y=1.01)

    for ax, idx in zip(axes.flat, indices):
        img = ventanear(volumen[idx], ventana[0], ventana[1])
        ax.imshow(img, cmap="gray", origin="lower")
        if mascara is not None and np.sum(mascara[idx] > 0) > 0:
            ax.contour(
                mascara[idx] > 0,
                levels=[0.5],
                colors=["#ef4444"],
                linewidths=1.5,
                origin="lower",
            )
        ax.set_title(f"z={idx}")
        ax.axis("off")

    for ax in axes.flat[n:]:
        ax.axis("off")

    plt.tight_layout()
    return fig


def estadisticas_tumor(
    volumen: np.ndarray,
    mascara: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    tumor = mascara > 0
    n_tumor = int(np.sum(tumor))
    hu_tumor = volumen[tumor]
    voxel_mm3 = float(np.prod(spacing))
    coords = np.argwhere(tumor)

    if len(coords) > 0:
        rangos_mm = (coords.max(axis=0) - coords.min(axis=0)) * np.asarray(spacing)
        diametro = float(np.linalg.norm(rangos_mm))
    else:
        diametro = 0.0

    return {
        "num_voxeles": n_tumor,
        "volumen_tumor_cm3": round((n_tumor * voxel_mm3) / 1000.0, 3),
        "diametro_max_mm": round(diametro, 2),
        "hu_media": round(float(np.mean(hu_tumor)), 2) if len(hu_tumor) else 0.0,
        "hu_std": round(float(np.std(hu_tumor)), 2) if len(hu_tumor) else 0.0,
        "hu_min": round(float(np.min(hu_tumor)), 2) if len(hu_tumor) else 0.0,
        "hu_max": round(float(np.max(hu_tumor)), 2) if len(hu_tumor) else 0.0,
    }


def limpiar_componentes_pequenas(
    mascara: np.ndarray,
    min_size: int,
    max_componentes: int = 2,
) -> np.ndarray:
    etiquetada, n = ndimage.label(mascara > 0)
    if n == 0:
        return np.zeros_like(mascara, dtype=np.uint8)

    tamanos = ndimage.sum(mascara > 0, etiquetada, range(1, n + 1))
    orden = np.argsort(tamanos)[::-1]
    salida = np.zeros_like(mascara, dtype=np.uint8)

    kept = 0
    for idx in orden:
        etiqueta = idx + 1
        tam = int(tamanos[idx])
        if tam < min_size:
            continue
        salida[etiquetada == etiqueta] = 1
        kept += 1
        if kept >= max_componentes:
            break
    return salida


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0
    gt_b = gt > 0
    inter = np.sum(pred_b & gt_b)
    total = np.sum(pred_b) + np.sum(gt_b)
    if total == 0:
        return 1.0
    return float(2.0 * inter / total)


def sensibilidad(pred: np.ndarray, gt: np.ndarray) -> float:
    gt_b = gt > 0
    tp = np.sum((pred > 0) & gt_b)
    fn = np.sum((pred == 0) & gt_b)
    return float(tp / max(tp + fn, 1))


def especificidad(pred: np.ndarray, gt: np.ndarray) -> float:
    gt_b = gt > 0
    tn = np.sum((pred == 0) & (~gt_b))
    fp = np.sum((pred > 0) & (~gt_b))
    return float(tn / max(tn + fp, 1))


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = np.sum((pred > 0) & (gt > 0))
    fp = np.sum((pred > 0) & (gt == 0))
    return float(tp / max(tp + fp, 1))


def hausdorff_95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, ...],
) -> float:
    pred_b = pred > 0
    gt_b = gt > 0
    if not np.any(pred_b) or not np.any(gt_b):
        return float("inf")

    try:
        struct = ndimage.generate_binary_structure(3, 1)
        pred_surf = pred_b ^ ndimage.binary_erosion(pred_b, struct)
        gt_surf = gt_b ^ ndimage.binary_erosion(gt_b, struct)

        dt_gt = ndimage.distance_transform_edt(~gt_b, sampling=spacing)
        dt_pred = ndimage.distance_transform_edt(~pred_b, sampling=spacing)

        d1 = np.percentile(dt_gt[pred_surf], 95) if np.any(pred_surf) else float("inf")
        d2 = np.percentile(dt_pred[gt_surf], 95) if np.any(gt_surf) else float("inf")

        hd95 = max(d1, d2)
        return round(float(hd95), 2)
    except Exception:
        # Fallback to monai if scipy approach fails for some reason
        pred_t = torch.from_numpy(pred_b).unsqueeze(0).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt_b).unsqueeze(0).unsqueeze(0).float()

        try:
            hd95_monai = compute_hausdorff_distance(pred_t, gt_t, percentile=95, spacing=spacing)
            return round(float(hd95_monai.item()), 2)
        except RuntimeError:
            return float("inf")


def evaluar_segmentacion(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, ...],
) -> dict[str, float]:
    return {
        "dice": round(dice_score(pred, gt), 4),
        "hd95_mm": hausdorff_95(pred, gt, spacing),
        "sensibilidad": round(sensibilidad(pred, gt), 4),
        "especificidad": round(especificidad(pred, gt), 4),
        "precision": round(precision_score(pred, gt), 4),
        "voxeles_pred": int(np.sum(pred > 0)),
        "voxeles_gt": int(np.sum(gt > 0)),
    }

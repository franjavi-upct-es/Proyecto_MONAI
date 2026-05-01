# src/lungseg/utils/visualization.py
"""Motor de visualización para reportes médicos y figuras de la memoria."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_triplanar_prediction(
    image: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    title: str = "Segmentación: ViT SSL vs Ground Truth",
) -> None:
    """Genera una vista triplanar (Axial, Coronal, Sagital) con overlays."""
    # Encontrar el centroide del tumor para centrar la vista
    coords = np.argwhere(label > 0)
    center = coords.mean(axis=0).astype(int) if len(coords) > 0 else np.array(image.shape) // 2

    _fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [
        (image[center[0], :, :], label[center[0], :, :], prediction[center[0], :, :], "Axial"),
        (image[:, center[1], :], label[:, center[1], :], prediction[:, center[1], :], "Coronal"),
        (image[:, :, center[2]], label[:, :, center[2]], prediction[:, :, center[2]], "Sagital"),
    ]

    for i, (ax, (img_slice, lab_slice, pred_slice, name)) in enumerate(zip(axes, views)):
        ax.imshow(img_slice.T, cmap="gray", origin="lower")
        # Overlay GT en Verde, Predicción en Rojo (transparencia)
        ax.imshow(
            np.ma.masked_where(lab_slice == 0, lab_slice).T,
            cmap="winter",
            alpha=0.5,
            origin="lower",
        )
        ax.imshow(
            np.ma.masked_where(pred_slice == 0, pred_slice).T,
            cmap="autumn",
            alpha=0.4,
            origin="lower",
        )
        ax.set_title(f"{name} (Slice {center[i]})")
        ax.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def save_segmentation_mosaic(
    image: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    num_slices: int = 12,
) -> None:
    """Genera un mosaico de cortes axiales con la predicción superpuesta."""
    indices = np.linspace(0, image.shape[0] - 1, num_slices).astype(int)
    cols = 4
    rows = (num_slices + cols - 1) // cols

    _fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    i = 0
    for i, idx in enumerate(indices):
        axes[i].imshow(image[idx].T, cmap="gray", origin="lower")
        axes[i].imshow(
            np.ma.masked_where(prediction[idx] == 0, prediction[idx]).T,
            cmap="hot",
            alpha=0.4,
            origin="lower",
        )
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close()

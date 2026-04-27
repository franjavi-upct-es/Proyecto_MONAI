"""
Fase 7 - Demo interactiva del pipeline SegResNet 3D.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from monai_pipeline.core.config import (
    CT_CLIP_RANGE,
    MODELO_3D_NOMBRE,
    RUTA_MODELO_3D,
    VENTANAS_CT,
)
from monai_pipeline.core.utils import estadisticas_tumor, ventanear
from monai_pipeline.pipelines.patch3d import (
    aplicar_threshold,
    cargar_modelo_checkpoint,
    inferir_caso,
)

warnings.filterwarnings("ignore")

try:
    import gradio as gr
except ImportError:
    gr = None
    print("Gradio no esta instalado. La demo queda desactivada.")


modelo = None
if RUTA_MODELO_3D.exists():
    modelo = cargar_modelo_checkpoint(RUTA_MODELO_3D)
    print(f"Modelo cargado: {MODELO_3D_NOMBRE}")
else:
    print(f"No existe {RUTA_MODELO_3D.name}. Ejecuta antes la fase 4.")


def _normalizar_file_input(archivo) -> Path | None:
    if archivo is None:
        return None
    if isinstance(archivo, str):
        return Path(archivo)
    if hasattr(archivo, "name"):
        return Path(archivo.name)
    return Path(str(archivo))


def pipeline_demo(archivo, ventana_nombre: str, z_idx: int, threshold: float):
    ruta = _normalizar_file_input(archivo)
    if ruta is None:
        return None, "Sube un volumen NIfTI."
    if modelo is None:
        return None, "No hay checkpoint 3D disponible. Ejecuta scripts/fase4_segmentacion.py."

    inferencia = inferir_caso(modelo, {"image": str(ruta), "case_id": ruta.stem})
    pred = aplicar_threshold(inferencia, threshold=float(threshold), cleanup=True)

    imagen_zyx = inferencia["image"]
    volumen_hu = imagen_zyx * (CT_CLIP_RANGE[1] - CT_CLIP_RANGE[0]) + CT_CLIP_RANGE[0]
    spacing = inferencia["spacing"]

    z = int(np.clip(z_idx, 0, volumen_hu.shape[0] - 1))
    y = volumen_hu.shape[1] // 2
    x = volumen_hu.shape[2] // 2
    width, level = VENTANAS_CT.get(ventana_nombre, VENTANAS_CT["tumor"])

    paneles = [
        ("Axial", volumen_hu[z], pred[z]),
        ("Coronal", volumen_hu[:, y, :], pred[:, y, :]),
        ("Sagital", volumen_hu[:, :, x], pred[:, :, x]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0f172a")
    for ax, (titulo, corte, mascara) in zip(axes, paneles):
        ax.set_facecolor("#111827")
        ax.imshow(ventanear(corte, width, level), cmap="gray", origin="lower")
        if np.sum(mascara > 0) > 0:
            ax.contour(
                mascara > 0,
                levels=[0.5],
                colors=["#ef4444"],
                linewidths=1.5,
                origin="lower",
            )
        ax.set_title(titulo, color="#e2e8f0")
        ax.axis("off")
    plt.tight_layout()

    stats = estadisticas_tumor(volumen_hu, pred, spacing)
    texto = (
        f"**Modelo:** {MODELO_3D_NOMBRE}\n\n"
        f"**Threshold:** {float(threshold):.2f}\n\n"
        f"**Voxeles tumorales:** {stats['num_voxeles']:,}\n\n"
        f"**Volumen estimado:** {stats['volumen_tumor_cm3']:.2f} cm3\n\n"
        f"**Diametro maximo:** {stats['diametro_max_mm']:.1f} mm\n\n"
        f"**HU media:** {stats['hu_media']:.1f} +/- {stats['hu_std']:.1f}"
    )
    return fig, texto


if gr is not None:
    with gr.Blocks(title="Segmentacion 3D de tumores pulmonares", theme=gr.themes.Base()) as demo:
        gr.Markdown(
            "# Segmentacion tumoral pulmonar en CT\n"
            "Pipeline activo: SegResNet 3D patch-based (MONAI)"
        )
        with gr.Row():
            with gr.Column(scale=1):
                archivo = gr.File(label="Volumen NIfTI", file_types=[".nii", ".nii.gz", ".gz"])
                ventana = gr.Radio(
                    ["pulmon", "mediastinica", "tumor"],
                    value="tumor",
                    label="Ventana CT",
                )
                z_idx = gr.Slider(0, 400, value=120, step=1, label="Corte axial")
                threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Threshold")
                boton = gr.Button("Analizar", variant="primary")
            with gr.Column(scale=2):
                salida_fig = gr.Plot(label="Prediccion")
                salida_txt = gr.Markdown()
        boton.click(pipeline_demo, [archivo, ventana, z_idx, threshold], [salida_fig, salida_txt])

    print("Demo lista. Lanzando...")

if __name__ == "__main__":
    if gr is not None:
        demo.launch(share=False)

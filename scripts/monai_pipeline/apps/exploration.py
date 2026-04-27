"""
Fase 1 - Analisis del dataset tras la migracion a 2D.

Objetivo:
  - Cuantificar por que el 3D estaba sobredimensionado para este problema.
  - Medir el desbalance real a nivel de corte axial.
  - Dejar una referencia visual y numerica para el pipeline 2D.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from monai_pipeline.core.config import (
    CT_CLIP_RANGE,
    DIR_FIGURAS,
    DIR_METRICAS,
    LEGACY_3D_ABLATION,
    LEGACY_3D_METRICS,
    VENTANAS_CT,
)
from monai_pipeline.core.utils import (
    crear_montaje_slices,
    estilo_oscuro,
    guardar_csv,
    guardar_figura,
    guardar_json,
    seleccionar_indices_tumor,
)
from monai_pipeline.pipelines.baseline2d import (
    cargar_casos_preprocesados,
    resumir_casos,
    split_por_fold,
)

warnings.filterwarnings("ignore")


casos = cargar_casos_preprocesados()
train_cases, val_cases = split_por_fold(casos, fold_idx=0)
df_resumen = resumir_casos(casos).sort_values("slices_positivos", ascending=False)
guardar_csv(df_resumen, DIR_METRICAS / "f1_resumen_casos.csv")

total_slices = int(df_resumen["slices_totales"].sum())
total_positive = int(df_resumen["slices_positivos"].sum())
total_alarm = int(df_resumen["slices_alarma"].sum())
legacy_best = max(fila["dice_medio"] for fila in LEGACY_3D_METRICS)
legacy_ablation_full = [
    fila["mejor_dice"] for fila in LEGACY_3D_ABLATION if fila["fraccion"] == 1.0
]

analisis = {
    "num_volumenes": len(casos),
    "num_train": len(train_cases),
    "num_val": len(val_cases),
    "num_slices_totales": total_slices,
    "num_slices_positivos": total_positive,
    "num_slices_alarma": total_alarm,
    "ratio_slices_positivos": round(total_positive / max(total_slices, 1), 4),
    "ratio_slices_alarma": round(total_alarm / max(total_slices, 1), 4),
    "mediana_slices_positivos_por_caso": float(df_resumen["slices_positivos"].median()),
    "mediana_slices_alarma_por_caso": float(df_resumen["slices_alarma"].median()),
    "min_slices_positivos_por_caso": int(df_resumen["slices_positivos"].min()),
    "max_slices_positivos_por_caso": int(df_resumen["slices_positivos"].max()),
    "mejor_dice_3d_legado": round(float(legacy_best), 4),
    "mejor_dice_fase6_3d_con_todos_los_datos": round(max(legacy_ablation_full), 4),
    "decision": (
        "Migrar el flujo activo a 2D axial con contexto entre cortes. "
        "Solo el 9-10% de los slices contienen tumor, asi que el 3D estaba "
        "gastando demasiada capacidad y computo en contexto vacio."
    ),
    "mascaras_prior": (
        "Se anaden lung_mask, healthy_lung_mask y soft_tissue_alarm_mask para "
        "dar contexto anatomico al modelo y evitar usar como fondo tajante "
        "tejido sospechoso no etiquetado."
    ),
}
guardar_json(analisis, DIR_METRICAS / "f1_analisis_estrategia.json")


# =============================================================================
# Figura 1 - Distribucion de slices positivos
# =============================================================================
estilo_oscuro()
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0f172a")

axes[0].hist(
    df_resumen["slices_positivos"],
    bins=12,
    color="#1e88e5",
    alpha=0.85,
    edgecolor="#0f172a",
)
axes[0].set_title("Distribucion de cortes con tumor por caso")
axes[0].set_xlabel("Slices positivos")
axes[0].set_ylabel("Numero de casos")
axes[0].grid(alpha=0.25)

axes[1].scatter(
    df_resumen["slices_positivos"],
    df_resumen["volumen_tumor_cm3"],
    s=80,
    color="#fb8c00",
    edgecolors="#0f172a",
)
axes[1].set_title("Relacion entre slices positivos y volumen tumoral")
axes[1].set_xlabel("Slices positivos")
axes[1].set_ylabel("Volumen tumoral (cm3)")
axes[1].grid(alpha=0.25)

fig.suptitle(
    "Diagnostico del dataset tras la migracion a 2D",
    y=1.02,
)
guardar_figura(fig, DIR_FIGURAS / "f1_distribucion_slices.png")


# =============================================================================
# Figura 2 - Montaje representativo
# =============================================================================
objetivo = float(df_resumen["slices_positivos"].median())
df_resumen["distancia_mediana"] = (df_resumen["slices_positivos"] - objetivo).abs()
case_id_rep = df_resumen.sort_values("distancia_mediana").iloc[0]["case_id"]
caso_rep = next(caso for caso in casos if caso.case_id == case_id_rep)

volumen_hu = (
    caso_rep.image.astype(np.float32) * (CT_CLIP_RANGE[1] - CT_CLIP_RANGE[0]) + CT_CLIP_RANGE[0]
)
indices = seleccionar_indices_tumor(caso_rep.label, n=6) if caso_rep.label is not None else [0]
fig = crear_montaje_slices(
    volumen=volumen_hu,
    mascara=caso_rep.label,
    indices=indices,
    titulo=f"Caso representativo para estrategia 2D: {caso_rep.case_id}",
)
guardar_figura(fig, DIR_FIGURAS / "f1_montaje_representativo.png")


# =============================================================================
# Figura 3 - Ventaneo por tejidos y nuevas mascaras
# =============================================================================
estilo_oscuro()
indices = seleccionar_indices_tumor(caso_rep.label, n=1) if caso_rep.label is not None else [0]
z = indices[0]
w_pulm, l_pulm = VENTANAS_CT["pulmon"]
w_tum, l_tum = VENTANAS_CT["tumor"]

fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0f172a")

axes[0].imshow(
    np.clip((volumen_hu[z] - (l_pulm - w_pulm / 2)) / w_pulm, 0, 1),
    cmap="gray",
    origin="lower",
)
axes[0].set_title("Ventana pulmon")
axes[0].axis("off")

axes[1].imshow(
    np.clip((volumen_hu[z] - (l_tum - w_tum / 2)) / w_tum, 0, 1),
    cmap="gray",
    origin="lower",
)
axes[1].contour(caso_rep.lung_mask[z] > 0, levels=[0.5], colors=["#38bdf8"], linewidths=1.5)
axes[1].set_title("Lung mask")
axes[1].axis("off")

axes[2].imshow(
    np.clip((volumen_hu[z] - (l_pulm - w_pulm / 2)) / w_pulm, 0, 1),
    cmap="gray",
    origin="lower",
)
axes[2].imshow(
    caso_rep.healthy_lung_mask[z] > 0,
    cmap="Greens",
    alpha=0.35,
    origin="lower",
)
axes[2].set_title("Pulmon sano esperado")
axes[2].axis("off")

axes[3].imshow(
    np.clip((volumen_hu[z] - (l_tum - w_tum / 2)) / w_tum, 0, 1),
    cmap="gray",
    origin="lower",
)
axes[3].imshow(
    caso_rep.soft_tissue_alarm_mask[z] > 0,
    cmap="autumn",
    alpha=0.45,
    origin="lower",
)
if caso_rep.label is not None and np.sum(caso_rep.label[z] > 0) > 0:
    axes[3].contour(
        caso_rep.label[z] > 0,
        levels=[0.5],
        colors=["#22c55e"],
        linewidths=1.5,
        origin="lower",
    )
axes[3].set_title("Alarma tejido blando")
axes[3].axis("off")

fig.suptitle(
    f"Ventaneo por tejidos y priors anatomicos: {caso_rep.case_id}",
    y=1.02,
)
guardar_figura(fig, DIR_FIGURAS / "f1_mascaras_prior.png")


# =============================================================================
# Resumen de priors y HU de la etiqueta
# =============================================================================
hu_label_mediana = []
hu_label_p10 = []
hu_label_p90 = []
for caso in casos:
    if caso.label is None or np.sum(caso.label > 0) == 0:
        continue
    vol_case_hu = (
        caso.image.astype(np.float32) * (CT_CLIP_RANGE[1] - CT_CLIP_RANGE[0]) + CT_CLIP_RANGE[0]
    )
    vals = vol_case_hu[caso.label > 0]
    hu_label_mediana.append(float(np.median(vals)))
    hu_label_p10.append(float(np.percentile(vals, 10)))
    hu_label_p90.append(float(np.percentile(vals, 90)))

prior_resumen = {
    "media_slices_alarma_por_caso": round(float(df_resumen["slices_alarma"].mean()), 2),
    "media_voxeles_alarma_por_caso": round(
        float(df_resumen["voxeles_alarma_tejido_blando"].mean()),
        2,
    ),
    "media_voxeles_pulmon_sano_por_caso": round(
        float(df_resumen["voxeles_pulmon_sano"].mean()),
        2,
    ),
    "hu_etiqueta_p10_medio": round(float(np.mean(hu_label_p10)), 2),
    "hu_etiqueta_mediana_media": round(float(np.mean(hu_label_mediana)), 2),
    "hu_etiqueta_p90_medio": round(float(np.mean(hu_label_p90)), 2),
    "lectura": (
        "La etiqueta de cancer existe, pero su distribucion HU es muy amplia. "
        "Por eso la mascara de alarma se usa como prior sensible, no como "
        "reemplazo directo de la etiqueta."
    ),
}
guardar_json(prior_resumen, DIR_METRICAS / "f1_priores_resumen.json")

print("Fase 1 completada.")

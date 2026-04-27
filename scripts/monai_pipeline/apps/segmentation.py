"""
Fase 4 - Segmentacion tumoral con SegResNet 3D patch-based.

Refactor 2026-04-25: la fase activa abandona el pipeline 2D/2.5D con priors y
usa SegResNet 3D entrenado con `RandCropByPosNegLabeld` para balancear
positivos/negativos. La motivacion es que el modelo 2D no resolvia la
localizacion (caso lung_078 daba el tamano correcto en sitio incorrecto) y los
SegResNet/DynUNet 3D legados ya alcanzaban Dice=0.32-0.33 sin priors.

Artefactos generados (sufijo `_3d` para distinguirlos de los del pipeline 2D
historico):
  - resultados/metricas/f4_historiales.json
  - resultados/metricas/f4_metricas_val_3d_std.csv
  - resultados/metricas/f4_metricas_val_3d.csv
  - resultados/metricas/f4_threshold_search_3d.csv
  - resultados/metricas/f4_metricas_val_3d_tta.csv (si TTA activo)
  - resultados/metricas/f4_threshold_search_3d_tta.csv (si TTA activo)
  - resultados/metricas/f4_resumen_3d.json
  - resultados/metricas/f4_comparativa.csv
  - resultados/figuras/f4_curvas_3d.png
  - resultados/figuras/f4_threshold_search_3d.png
  - resultados/figuras/f4_comparativa_3d.png
  - resultados/figuras/f4_predicciones_3d.png
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monai_pipeline.core.config import (
    COLORES,
    DIR_FIGURAS,
    DIR_METRICAS,
    FAST_DEV_RUN,
    KFOLD_DEFAULT_INDEX,
    LEGACY_3D_HISTORIES,
    LEGACY_3D_METRICS,
    MODELO_3D_NOMBRE,
    MODELO_CASCADE_FINE_NOMBRE,
    NOMBRES_REGIMEN,
    NUM_EPOCAS_3D,
    RUTA_MODELO_3D,
    SEGMENTATION_AUGMENT_LEVEL,
    SEGMENTATION_ENABLE_TTA_FINAL,
    SEGMENTATION_THRESHOLD_MAX,
    SEGMENTATION_THRESHOLD_MIN,
    SEGMENTATION_THRESHOLD_STEP,
)
from monai_pipeline.core.utils import (
    estilo_oscuro,
    evaluar_segmentacion,
    guardar_csv,
    guardar_figura,
    guardar_json,
    seleccionar_indices_tumor,
)
from monai_pipeline.pipelines.patch3d import (
    aplicar_threshold,
    construir_data_dicts,
    contar_parametros,
    entrenar_modelo,
    inferir_todos,
    split_kfold,
)

warnings.filterwarnings("ignore")


# =============================================================================
# Helpers
# =============================================================================
def asegurar_metricas_legado() -> None:
    ruta_csv = DIR_METRICAS / "f4_comparativa_legacy_3d.csv"
    ruta_json = DIR_METRICAS / "f4_historiales_legacy_3d.json"
    if not ruta_csv.exists():
        guardar_csv(pd.DataFrame(LEGACY_3D_METRICS), ruta_csv)
    if not ruta_json.exists():
        guardar_json(LEGACY_3D_HISTORIES, ruta_json)


def construir_thresholds() -> list[float]:
    valores = np.arange(
        SEGMENTATION_THRESHOLD_MIN,
        SEGMENTATION_THRESHOLD_MAX + (SEGMENTATION_THRESHOLD_STEP * 0.5),
        SEGMENTATION_THRESHOLD_STEP,
    )
    return [round(float(x), 2) for x in valores]


def evaluar_inferencias(
    inferencias: list[dict],
    threshold: float,
    cleanup: bool = True,
    use_lung_mask: bool = True,
) -> pd.DataFrame:
    filas = []
    for inf in inferencias:
        if "label" not in inf:
            continue
        pred = aplicar_threshold(
            inf,
            threshold=threshold,
            cleanup=cleanup,
            use_lung_mask=use_lung_mask,
        )
        met = evaluar_segmentacion(pred, inf["label"], inf["spacing"])
        met["case_id"] = inf["case_id"]
        filas.append(met)
    return pd.DataFrame(filas)


def resumir_eval(df_eval: pd.DataFrame) -> dict[str, float | int]:
    if df_eval.empty:
        return {
            "n_cases": 0,
            "dice_mean": 0.0,
            "dice_median": 0.0,
            "hd95_mean_mm": float("nan"),
            "sensibilidad_mean": 0.0,
            "especificidad_mean": 0.0,
            "precision_mean": 0.0,
            "zero_dice_cases": 0,
        }
    hd95 = df_eval["hd95_mm"].replace(np.inf, np.nan)
    return {
        "n_cases": len(df_eval),
        "dice_mean": round(float(df_eval["dice"].mean()), 4),
        "dice_median": round(float(df_eval["dice"].median()), 4),
        "hd95_mean_mm": round(float(hd95.mean()), 2),
        "sensibilidad_mean": round(float(df_eval["sensibilidad"].mean()), 4),
        "especificidad_mean": round(float(df_eval["especificidad"].mean()), 4),
        "precision_mean": round(float(df_eval["precision"].mean()), 4),
        "zero_dice_cases": int((df_eval["dice"] == 0).sum()),
    }


def buscar_threshold(
    inferencias: list[dict],
    mode: str,
) -> tuple[pd.DataFrame, float, pd.DataFrame, dict[str, float | int]]:
    filas: list[dict[str, object]] = []
    evals: dict[float, pd.DataFrame] = {}
    for threshold in construir_thresholds():
        df_eval = evaluar_inferencias(inferencias, threshold=threshold)
        resumen = resumir_eval(df_eval)
        filas.append({"mode": mode, "threshold": threshold, **resumen})
        evals[threshold] = df_eval

    df_search = pd.DataFrame(filas)
    df_rank = df_search.sort_values(
        ["dice_mean", "zero_dice_cases", "dice_median", "precision_mean"],
        ascending=[False, True, False, False],
    )
    best_threshold = float(df_rank.iloc[0]["threshold"]) if not df_rank.empty else 0.5
    df_best = evals.get(best_threshold, pd.DataFrame())
    return df_search, best_threshold, df_best, resumir_eval(df_best)


def fila_comparativa(
    *,
    metodo: str,
    tipo: str,
    df_eval: pd.DataFrame,
    tiempo_total_s: float,
    parametros: int,
) -> dict[str, object]:
    resumen = resumir_eval(df_eval)
    return {
        "metodo": metodo,
        "tipo": tipo,
        "dice_medio": resumen["dice_mean"],
        "dice_std": round(float(df_eval["dice"].std(ddof=0)), 4) if not df_eval.empty else 0.0,
        "hd95_medio_mm": resumen["hd95_mean_mm"],
        "sensibilidad": resumen["sensibilidad_mean"],
        "especificidad": resumen["especificidad_mean"],
        "precision": resumen["precision_mean"],
        "n_eval": len(df_eval),
        "parametros": parametros,
        "tiempo_total_s": tiempo_total_s,
        "fast_dev_run": FAST_DEV_RUN,
    }


def cargar_fila_cascada_historica() -> list[dict[str, object]]:
    ruta_eval = DIR_METRICAS / "f4_cascade_metricas_val.csv"
    ruta_resumen = DIR_METRICAS / "f4_cascade_resumen.json"
    if not ruta_eval.exists():
        return []
    df_eval = pd.read_csv(ruta_eval)
    threshold = None
    if ruta_resumen.exists():
        try:
            import json

            resumen = json.loads(ruta_resumen.read_text(encoding="utf-8"))
            threshold = resumen.get("threshold_optimo")
        except Exception:
            threshold = None
    suffix = f", thresh={float(threshold):.2f}" if threshold is not None else ""
    fila = fila_comparativa(
        metodo=f"{MODELO_CASCADE_FINE_NOMBRE} (historico descartado{suffix})",
        tipo="Cascada historica",
        df_eval=df_eval,
        tiempo_total_s=float("nan"),
        parametros=0,
    )
    fila["fast_dev_run"] = False
    return [fila]


def cargar_fila_baseline_2d() -> list[dict[str, object]]:
    """Trae la mejor fila del pipeline 2D historico desde f4_metricas_val_2d_tta.csv."""
    candidatos = [
        DIR_METRICAS / "f4_metricas_val_2d_tta.csv",
        DIR_METRICAS / "f4_metricas_val_2d.csv",
    ]
    for ruta in candidatos:
        if ruta.exists():
            df_eval = pd.read_csv(ruta)
            tipo = "2D axial+TTA" if "tta" in ruta.name else "2D axial"
            return [
                fila_comparativa(
                    metodo=f"UNet2D-AxialContext+Priors-v4 (historico, {tipo})",
                    tipo="2D axial historico",
                    df_eval=df_eval,
                    tiempo_total_s=float("nan"),
                    parametros=0,
                )
            ]
    return []


# =============================================================================
# Pipeline
# =============================================================================


def main() -> None:
    asegurar_metricas_legado()

    data_dicts = construir_data_dicts()
    if FAST_DEV_RUN:
        data_dicts = data_dicts[:5]
    train_files, val_files = split_kfold(data_dicts, fold_idx=KFOLD_DEFAULT_INDEX)
    if FAST_DEV_RUN:
        train_files = train_files[:2]
        val_files = val_files[:1]

    print("\nFase 4 activa: SegResNet 3D patch-based (MONAI)")
    print(f"  train cases: {len(train_files)}")
    print(f"  val cases: {len(val_files)}")
    print(f"  augment_level: {SEGMENTATION_AUGMENT_LEVEL}")
    print(f"  TTA final: {SEGMENTATION_ENABLE_TTA_FINAL}")

    checkpoint_path = None if FAST_DEV_RUN else RUTA_MODELO_3D
    modelo, historia = entrenar_modelo(
        train_files=train_files,
        val_files=val_files,
        augment_level=SEGMENTATION_AUGMENT_LEVEL,
        num_epocas=NUM_EPOCAS_3D,
        checkpoint_path=checkpoint_path,
        descripcion=f"{MODELO_3D_NOMBRE} MONAI-first",
        fold_idx=KFOLD_DEFAULT_INDEX,
    )

    inferencias_single = inferir_todos(modelo, val_files, use_tta=False)
    df_std = evaluar_inferencias(inferencias_single, threshold=0.50)
    df_threshold, threshold_optimo, df_final, resumen_final = buscar_threshold(
        inferencias_single, mode="single"
    )

    df_threshold_tta = None
    df_final_tta = None
    threshold_tta = None
    resumen_tta = None
    if SEGMENTATION_ENABLE_TTA_FINAL:
        inferencias_tta = inferir_todos(modelo, val_files, use_tta=True)
        df_threshold_tta, threshold_tta, df_final_tta, resumen_tta = buscar_threshold(
            inferencias_tta, mode="tta"
        )
    else:
        inferencias_tta = None

    guardar_json(historia, DIR_METRICAS / "f4_historiales.json")
    guardar_csv(df_std, DIR_METRICAS / "f4_metricas_val_3d_std.csv")
    guardar_csv(df_threshold, DIR_METRICAS / "f4_threshold_search_3d.csv")
    guardar_csv(df_final, DIR_METRICAS / "f4_metricas_val_3d.csv")
    if df_threshold_tta is not None and df_final_tta is not None:
        guardar_csv(df_threshold_tta, DIR_METRICAS / "f4_threshold_search_3d_tta.csv")
        guardar_csv(df_final_tta, DIR_METRICAS / "f4_metricas_val_3d_tta.csv")

    resumen_global = {
        "fold_idx": KFOLD_DEFAULT_INDEX,
        "modelo": MODELO_3D_NOMBRE,
        "strategy": "MONAI 3D patch-based con SegResNet",
        "patch_size": list(historia.get("patch_size", [])),
        "samples_per_volume": historia.get("samples_per_volume"),
        "augment_level": SEGMENTATION_AUGMENT_LEVEL,
        "augment_regimen": NOMBRES_REGIMEN.get(SEGMENTATION_AUGMENT_LEVEL),
        "loss_fn": historia.get("loss_fn"),
        "threshold_optimo": threshold_optimo,
        "summary_std_threshold_0_50": resumir_eval(df_std),
        "summary_final": resumen_final,
        "tta_enabled": SEGMENTATION_ENABLE_TTA_FINAL,
        "threshold_tta": threshold_tta,
        "summary_tta": resumen_tta,
        "fast_dev_run": FAST_DEV_RUN,
    }
    guardar_json(resumen_global, DIR_METRICAS / "f4_resumen_3d.json")

    # =============================================================================
    # Comparativa
    # =============================================================================
    df_comp = pd.DataFrame(LEGACY_3D_METRICS)
    parametros = int(historia.get("n_params", contar_parametros(modelo)))
    tiempo_total_s = float(historia.get("tiempo_total_s", 0.0))
    nombre_regimen = NOMBRES_REGIMEN.get(
        SEGMENTATION_AUGMENT_LEVEL, str(SEGMENTATION_AUGMENT_LEVEL)
    ).lower()

    filas_3d = [
        fila_comparativa(
            metodo=f"{MODELO_3D_NOMBRE} (std thresh=0.50)",
            tipo="3D patch",
            df_eval=df_std,
            tiempo_total_s=tiempo_total_s,
            parametros=parametros,
        ),
        fila_comparativa(
            metodo=f"{MODELO_3D_NOMBRE} ({nombre_regimen}, thresh={threshold_optimo:.2f})",
            tipo="3D patch",
            df_eval=df_final,
            tiempo_total_s=tiempo_total_s,
            parametros=parametros,
        ),
    ]
    if df_final_tta is not None and threshold_tta is not None:
        filas_3d.append(
            fila_comparativa(
                metodo=f"{MODELO_3D_NOMBRE} ({nombre_regimen}+TTA, thresh={threshold_tta:.2f})",
                tipo="3D patch+TTA",
                df_eval=df_final_tta,
                tiempo_total_s=tiempo_total_s,
                parametros=parametros,
            )
        )
    df_comp = pd.concat([df_comp, pd.DataFrame(filas_3d)], ignore_index=True)

    filas_legacy = cargar_fila_baseline_2d() + cargar_fila_cascada_historica()
    if filas_legacy:
        df_comp = pd.concat([df_comp, pd.DataFrame(filas_legacy)], ignore_index=True)

    guardar_csv(df_comp, DIR_METRICAS / "f4_comparativa.csv")

    # =============================================================================
    # Figura 1 - Curvas
    # =============================================================================
    estilo_oscuro()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f172a")
    axes[0].plot(historia.get("train_loss", []), color=COLORES[MODELO_3D_NOMBRE], lw=2.5)
    axes[0].set_title(f"{MODELO_3D_NOMBRE} - perdida de entrenamiento")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    val_eps = historia.get("val_epochs", [])
    val_dices = historia.get("val_dice", [])
    if val_eps and val_dices:
        axes[1].plot(val_eps, val_dices, color="#fb8c00", lw=2.5, marker="o")
    axes[1].set_title(f"{MODELO_3D_NOMBRE} - Dice de validacion")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    guardar_figura(fig, DIR_FIGURAS / "f4_curvas_3d.png")

    # =============================================================================
    # Figura 2 - Threshold search
    # =============================================================================
    estilo_oscuro()
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="#0f172a")
    ax.plot(
        df_threshold["threshold"], df_threshold["dice_mean"], marker="o", lw=2.2, label="single"
    )
    ax.axvline(
        threshold_optimo, color="#f97316", lw=1.5, ls="--", label=f"single={threshold_optimo:.2f}"
    )
    if df_threshold_tta is not None and threshold_tta is not None:
        ax.plot(
            df_threshold_tta["threshold"],
            df_threshold_tta["dice_mean"],
            marker="s",
            lw=2.2,
            label="TTA",
        )
        ax.axvline(
            threshold_tta, color="#22c55e", lw=1.5, ls="--", label=f"TTA={threshold_tta:.2f}"
        )
    ax.set_title(f"Busqueda de threshold para {MODELO_3D_NOMBRE}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Dice medio")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    guardar_figura(fig, DIR_FIGURAS / "f4_threshold_search_3d.png")

    # =============================================================================
    # Figura 3 - Comparativa
    # =============================================================================
    estilo_oscuro()
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0f172a")
    color_por_tipo = {
        "3D legado": COLORES["3D legado"],
        "3D patch": COLORES[MODELO_3D_NOMBRE],
        "3D patch+TTA": "#22c55e",
        "2D axial historico": "#64748b",
        "Cascada historica": "#94a3b8",
    }
    colores = [color_por_tipo.get(str(tipo), "#94a3b8") for tipo in df_comp["tipo"]]
    bars = ax.bar(range(len(df_comp)), df_comp["dice_medio"], color=colores, alpha=0.9, width=0.65)
    for bar, valor in zip(bars, df_comp["dice_medio"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(valor) + 0.01,
            f"{float(valor):.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(range(len(df_comp)))
    ax.set_xticklabels(
        [str(m)[:34] for m in df_comp["metodo"]], rotation=20, ha="right", fontsize=9
    )
    ax.set_title("Comparativa: legado 3D, SegResNet 3D activo y referencias historicas")
    ax.set_ylabel("Dice medio")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    guardar_figura(fig, DIR_FIGURAS / "f4_comparativa_3d.png")

    # =============================================================================
    # Figura 4 - Predicciones top
    # =============================================================================
    inferencias_para_figura = inferencias_tta if inferencias_tta is not None else inferencias_single
    threshold_para_figura = threshold_tta if threshold_tta is not None else threshold_optimo
    df_final_para_figura = df_final_tta if df_final_tta is not None else df_final

    if not df_final_para_figura.empty and inferencias_para_figura is not None:
        top_ids = df_final_para_figura.sort_values("dice", ascending=False)["case_id"].tolist()[
            : min(3, len(df_final_para_figura))
        ]
        inf_por_id = {inf["case_id"]: inf for inf in inferencias_para_figura}
        estilo_oscuro()
        fig, axes = plt.subplots(
            len(top_ids),
            3,
            figsize=(15, 5 * len(top_ids)),
            squeeze=False,
            facecolor="#0f172a",
        )
        for fila, case_id in enumerate(top_ids):
            inf = inf_por_id[case_id]
            pred = aplicar_threshold(inf, threshold=threshold_para_figura, cleanup=True)
            label = inf.get("label")
            z_indices = (
                seleccionar_indices_tumor(label, n=1)
                if label is not None
                else [inf["image"].shape[0] // 2]
            )
            z = z_indices[0]
            paneles = [
                ("Slice axial", None),
                ("Ground truth", label),
                (f"Pred thresh={threshold_para_figura:.2f}", pred),
            ]
            for col, (titulo, mascara) in enumerate(paneles):
                ax_i = axes[fila, col]
                ax_i.imshow(np.clip(inf["image"][z], 0, 1), cmap="gray", origin="lower")
                if mascara is not None and np.sum(mascara[z] > 0) > 0:
                    ax_i.contour(
                        mascara[z] > 0,
                        levels=[0.5],
                        colors=["#ef4444"],
                        linewidths=1.8,
                        origin="lower",
                    )
                ax_i.set_title(f"{case_id} - {titulo}")
                ax_i.axis("off")
        guardar_figura(fig, DIR_FIGURAS / "f4_predicciones_3d.png")

    print("\nFase 4 SegResNet 3D completada.")
    print(f"  Threshold final: {threshold_optimo:.2f}")
    print(f"  Dice medio final: {float(resumen_final['dice_mean']):.4f}")
    print(f"  Dice mediano final: {float(resumen_final['dice_median']):.4f}")
    print(f"  Casos con Dice=0: {int(resumen_final['zero_dice_cases'])}/{len(df_final)}")
    if resumen_tta is not None and threshold_tta is not None:
        print(
            "  TTA final: "
            f"threshold={threshold_tta:.2f} | dice={float(resumen_tta['dice_mean']):.4f}"
        )


if __name__ == "__main__":
    main()

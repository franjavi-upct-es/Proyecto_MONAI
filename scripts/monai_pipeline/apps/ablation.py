"""
Fase 6 - Ablacion de datos y augmentacion para SegResNet 3D.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import pandas as pd

from monai_pipeline.core.config import (
    COLORES,
    DIR_FIGURAS,
    DIR_METRICAS,
    FAST_DEV_RUN,
    FRACCIONES_DATOS,
    KFOLD_DEFAULT_INDEX,
    LEGACY_3D_ABLATION,
    NOMBRES_REGIMEN,
    NUM_EPOCAS_ABLACION_3D,
)
from monai_pipeline.core.utils import (
    dice_score,
    estilo_oscuro,
    guardar_csv,
    guardar_figura,
)
from monai_pipeline.pipelines.patch3d import (
    aplicar_threshold,
    construir_data_dicts,
    entrenar_modelo,
    inferir_todos,
    seleccionar_subconjunto,
    split_kfold,
)

warnings.filterwarnings("ignore")


def asegurar_legacy_ablacion() -> None:
    ruta = DIR_METRICAS / "f6_ablacion_legacy_3d.csv"
    if not ruta.exists():
        guardar_csv(pd.DataFrame(LEGACY_3D_ABLATION), ruta)


def main() -> None:
    asegurar_legacy_ablacion()

    data_dicts = construir_data_dicts()
    if FAST_DEV_RUN:
        data_dicts = data_dicts[:5]
    train_files, val_files = split_kfold(data_dicts, fold_idx=KFOLD_DEFAULT_INDEX)

    if FAST_DEV_RUN:
        train_files = train_files[:3]
        val_files = val_files[:1]

    resultados: list[dict[str, object]] = []
    fracciones_eval = [1.0] if FAST_DEV_RUN else FRACCIONES_DATOS
    for fraccion in fracciones_eval:
        subset = seleccionar_subconjunto(train_files, fraccion)
        for regimen in [1, 2]:
            print(
                f"\nEntrenando ablacion: frac={fraccion:.0%} "
                f"regimen={NOMBRES_REGIMEN[regimen]} casos={len(subset)}"
            )
            modelo, historia = entrenar_modelo(
                train_files=subset,
                val_files=val_files,
                augment_level=regimen,
                num_epocas=NUM_EPOCAS_ABLACION_3D,
                checkpoint_path=None,
                descripcion=f"fraccion={fraccion:.0%} regimen={NOMBRES_REGIMEN[regimen]}",
            )
            inferencias = inferir_todos(modelo, val_files, use_tta=False)
            dices = []
            for inf in inferencias:
                if "label" not in inf:
                    continue
                pred = aplicar_threshold(inf, threshold=0.5, cleanup=True)
                dices.append(dice_score(pred, inf["label"]))
            dice_medio = round(float(sum(dices) / max(len(dices), 1)), 4)
            resultados.append(
                {
                    "fraccion": fraccion,
                    "regimen": regimen,
                    "nombre_regimen": NOMBRES_REGIMEN[regimen],
                    "n_train_casos": len(subset),
                    "mejor_dice": dice_medio,
                    "mejor_epoca": int(historia["mejor_epoca"]),
                    "tiempo_s": float(historia["tiempo_total_s"]),
                    "fast_dev_run": FAST_DEV_RUN,
                }
            )

    df_abl = pd.DataFrame(resultados)
    guardar_csv(df_abl, DIR_METRICAS / "f6_ablacion.csv")

    # =============================================================================
    # Figura 1 - Curvas
    # =============================================================================
    estilo_oscuro()
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#0f172a")
    for regimen in [1, 2]:
        sub = df_abl[df_abl["regimen"] == regimen].sort_values("fraccion")
        if sub.empty:
            continue
        ax.plot(
            sub["fraccion"] * 100,
            sub["mejor_dice"],
            marker="o",
            lw=2.5,
            ms=8,
            color=COLORES[NOMBRES_REGIMEN[regimen]],
            label=NOMBRES_REGIMEN[regimen],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['mejor_dice']:.3f}",
                (row["fraccion"] * 100, row["mejor_dice"]),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=9,
            )
    ax.set_title("Impacto de la augmentacion en SegResNet 3D")
    ax.set_xlabel("Fraccion de casos de entrenamiento (%)")
    ax.set_ylabel("Dice medio en validacion")
    ax.set_xticks([25, 50, 100])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    guardar_figura(fig, DIR_FIGURAS / "f6_curvas_aprendizaje.png")

    # =============================================================================
    # Figura 2 - Heatmap
    # =============================================================================
    estilo_oscuro()
    if not df_abl.empty:
        pivot = df_abl.pivot_table(
            values="mejor_dice",
            index="fraccion",
            columns="nombre_regimen",
            aggfunc="first",
        )
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0f172a")
        im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{int(f * 100)}%" for f in pivot.index])
        ax.set_xlabel("Regimen de augmentacion")
        ax.set_ylabel("Fraccion de datos")
        ax.set_title("Heatmap de resultados SegResNet 3D")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                valor = float(pivot.values[i, j])
                ax.text(j, i, f"{valor:.3f}", ha="center", va="center", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, label="Dice medio")
        guardar_figura(fig, DIR_FIGURAS / "f6_heatmap.png")

    print("Fase 6 completada.")


if __name__ == "__main__":
    main()

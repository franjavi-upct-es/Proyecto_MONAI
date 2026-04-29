# lungseg — Segmentación y Clasificación de Tumores de Pulmón

Segmentación 3D de tumores NSCLC en TC (MSD Task06_Lung, 64 casos) con MONAI
1.5+, además de un clasificador de características radiómicas / profundas de la
Fase 5 (LIDC-IDRI) y un estudio de ablación de la Fase 6.

Este repositorio reemplaza una base de código anterior `scripts/monai_pipeline/`
que alcanzó un máximo de DSC ≈ 0.14 con los síntomas diagnosticados en
[REPORT_DIAGNOSIS.md](REPORT_DIAGNOSIS.md). El nuevo paquete `src/lungseg/` es
impulsado por Hydra y está verificado con pruebas.

## Configuración

```bash
pip install -e .[dev]                  # núcleo + herramientas de desarrollo
# extras opcionales:
pip install -e .[radiomics]            # Fase 5: pyradiomics, pylidc, xgboost
pip install -e .[demo]                 # demo de gradio
pip install -e .[kaggle]               # wandb para ejecuciones en la nube
```

## Datos

Coloque el conjunto de datos MSD Task06_Lung en `data/raw/Task06_Lung/`. El tar
original de la descarga de MONAI reside en `data/raw/Task06_Lung.tar`. Las
particiones fijas a nivel de paciente son generadas por B2 en
`data/splits/fold_{0..4}.json` (incluidas en el repositorio).

`data/raw/` y `data/processed/` están en .gitignore. No edite los archivos bajo
`data/raw/`.

## Comandos

```bash
pytest tests/ -q                                              # pruebas
ruff check src/                                               # lint (análisis estático)
python -m lungseg.cli --help                                  # ayuda de la CLI

python -m lungseg.cli train --config-name=sanity              # sobreajuste de 1 lote
python -m lungseg.cli train experiment=phase4_full            # Fase 4 completa
python -m lungseg.cli ablate -m experiment=phase6_ablation \
    seed=0,1,2 data_fraction=0.25,0.5,1.0                     # multiejecución de Hydra
```

## Reglas estrictas

Vea [.claude/CLAUDE.md](.claude/CLAUDE.md) para la lista completa. Destacados:

- GroupKFold a nivel de paciente; nunca aleatorio por imagen.
- Recorte de HU `a_min=-1024, a_max=400` (desviación justificada de los 325 de
  nnU-Net).
- `RandFlipd spatial_axis=0` (LR) está prohibido en TC de tórax.
- MSD Task06 NO tiene etiquetas de benigno/maligno.

## Estructura

```
src/lungseg/                # el paquete
configs/                    # árbol de Hydra (data/, model/, training/, experiment/)
tests/                      # suite de pytest
data/{raw,processed,splits} # raw/ y processed/ en gitignore; splits en el repo
outputs/                    # checkpoints + ejecuciones de Hydra (en gitignore)
docs/legacy_metrics/        # CSVs citados en REPORT_DIAGNOSIS.md
notebooks/                  # B7 (Kaggle) aterriza aquí
```

# lungseg — Segmentación y Clasificación de Tumores de Pulmón

Pipeline de segmentación de tumores NSCLC en TC (MSD Task06_Lung, 64 casos)
construido sobre MONAI 1.5+, con extensiones para clasificación radiómica /
profunda en LIDC-IDRI (Fase 5) y un estudio de ablación (Fase 6).

El paquete `src/lungseg/` reemplaza la base de código previa
`scripts/monai_pipeline/` que alcanzó un máximo de DSC ≈ 0.14 con los síntomas
diagnosticados en [REPORT_DIAGNOSIS.md](REPORT_DIAGNOSIS.md). El nuevo paquete
es impulsado por Hydra y cubierto por tests.

## Arquitectura del segmentador (Phase 4)

`ViT25D` es un segmentador 2.5D con contexto de slices vecinos:

- **Pre-procesamiento por volumen**: re-orientación RAS, resampling a
  `target_spacing = (0.79, 0.79, 1.24) mm`, máscara pulmonar (lungmask R231)
  aplicada como prior duro (voxels no-pulmón → -1024 HU), tres ventanas HU
  (pulmón / mediastino / completa) normalizadas a `[0, 1]` como canales.
- **Modelo**: para cada slice axial `z`, se apilan los 5 vecinos
  `[z-2, ..., z+2]` × 3 ventanas = 15 canales y se pasan a un encoder
  `vit_base_patch16_224.mae` pre-entrenado (congelado por defecto). El primer
  conv del `patch_embed` se inicializa promediando los pesos RGB y replicándolos
  a 15 canales para preservar el conocimiento MAE.
- **Decoder**: 2D slice-wise con tres cabezas a resoluciones full / 1⁄2 / 1⁄4
  (deep supervision en XY; Z se conserva).
- **Pérdida**: `FocalTverskyLoss(α=0.3, β=0.7, γ=4/3)` — los falsos negativos
  se penalizan 7/3 veces más que los falsos positivos.

## Configuración

```bash
pip install -e .[dev]                  # núcleo + herramientas de desarrollo
# extras opcionales:
pip install -e .[radiomics]            # Fase 5: pyradiomics, pylidc, xgboost
pip install -e .[demo]                 # demo de gradio
pip install -e .[kaggle]               # wandb para ejecuciones en la nube
```

`lungmask` está incluido en el paquete principal (necesario para el prior
pulmonar). El primer uso descargará automáticamente los pesos R231 a
`~/.cache/torch/hub/`.

## Datos

Coloque el conjunto de datos MSD Task06_Lung en `data/raw/Task06_Lung/`. Las
particiones fijas a nivel de paciente viven en `data/splits/fold_{0..4}.json`
(GroupKFold por `patient_id`).

`data/raw/` y `data/processed/` están en `.gitignore`. **No edite los archivos
bajo `data/raw/`.**

Antes del primer entrenamiento hay que pre-computar las máscaras pulmonares
(idempotente: si ya están en caché no se recomputan):

```bash
python -m lungseg.cli precompute-lung-masks
# o con make:
make precompute-masks
```

Las máscaras se guardan en `data/cache/lung_masks/<patient_id>.nii.gz`.

## Comandos

Pipeline mínimo de Phase 4 desde cero:

```bash
make bootstrap              # uv sync con extras dev + radiomics
make splits                 # genera fold_{0..4}.json (idempotente)
make precompute-masks       # cachea las máscaras pulmonares
make phase4-fold FOLD=0     # entrena un fold con vit_25d_lung
```

Comandos sueltos vía CLI:

```bash
pytest tests/ -q                                              # pruebas
ruff check src tests                                          # lint
python -m lungseg.cli --help                                  # ayuda

python -m lungseg.cli precompute-lung-masks                   # cachea máscaras
python -m lungseg.cli train --config-name=sanity              # overfit 1 batch
python -m lungseg.cli train experiment=phase4_full            # Phase 4 fold 0
python -m lungseg.cli ablate -m experiment=phase6_ablation \
    seed=0,1,2 data_fraction=0.25,0.5,1.0                     # multirun Hydra
python -m lungseg.cli predict --checkpoint best.pt \
    --image volume.nii.gz --output pred.nii.gz                # inferencia
```

`make doctor` valida que `uv`, Task06, splits y máscaras pulmonares están listos.

## Reglas estrictas

Vea [.claude/CLAUDE.md](.claude/CLAUDE.md) para la lista completa. Destacados:

- GroupKFold a nivel de paciente; nunca aleatorio por imagen.
- Tres ventanas HU como canales (`-1000..0`, `-150..250`, `-1024..400`) en
  lugar de un único `ScaleIntensityRanged`.
- Máscara pulmonar obligatoria como prior anatómico antes de ventanear.
- `RandFlipd spatial_axis=0` (LR) está prohibido en TC de tórax — guard
  defensivo en `lungseg.data.transforms._check_no_lr_flip`.
- MSD Task06 NO tiene etiquetas de benigno/maligno; el pipeline de
  clasificación lanza un error claro si se invoca con `data=task06`.

## Estructura

```
src/lungseg/                # paquete principal
  data/                     #   carga, splits, transforms, lung_mask
  models/                   #   ViT25D
  training/                 #   trainer, losses (Focal Tversky), schedulers
  inference/                #   sliding window 2.5D, postprocesado
  radiomics/, classification/, ablation/, utils/
configs/                    # árbol de Hydra (data/, model/, training/, experiment/)
tests/                      # suite de pytest
data/{raw,processed,splits,cache} # raw/, processed/, cache/ en .gitignore
outputs/                    # checkpoints + ejecuciones de Hydra (en .gitignore)
docs/legacy_metrics/        # CSVs citados en REPORT_DIAGNOSIS.md
notebooks/                  # B7 (Kaggle) aterriza aquí
```

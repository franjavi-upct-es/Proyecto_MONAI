# Changelog

### Presentación reveal.js basada en la memoria (2026-05-04)

- Nuevo `docs/presentacion.qmd` con `format: revealjs` que sintetiza el
  contenido de `docs/memoria.qmd`: motivación, datos, preprocesamiento,
  arquitectura ViT25D, pérdida, entrenamiento, inferencia, ablación de
  Fase 6, resultados y discusión.
- Reutiliza `outputs/{ablation_violin,metrics,triplanar_best,mosaic_results}.png`
  y `references.bib` ya configurado en `docs/_quarto.yml`.
- Compilable con `quarto render docs/presentacion.qmd` (genera HTML
  reveal.js auto-contenido vía `embed-resources: true`).
- Rediseño orientado a exposición (2026-05-04): bloques de código
  reemplazados por **diagramas Mermaid** (pipeline general, flujo de
  tensores ViT25D, neighbor stacking, adaptación de pesos 3→15 canales,
  decoder con deep supervision, congelación, sliding window slice-wise,
  chunking, ablación, comparativa 2D/2.5D/3D). Layout en columnas para
  agrupar concepto + esquema. `transition: none`,
  `background-transition: none` e `incremental: false` para minimizar
  animaciones y acelerar el cambio de sección; eliminados los fragments
  `. . .` que segmentaban contenido.
- Auto-escalado del contenido (2026-05-04): canvas lógico de reveal.js
  ampliado a `1600×900` con `margin: 0.04` y `min-scale: 0.2` /
  `max-scale: 2.0`; `scrollable: false` y bloque `<style>` que fuerza
  `overflow: hidden`, limita imágenes y SVG (Mermaid) a `max-height: 62vh`
  y compacta tablas, código, listas y callouts. Ahora cada slide cabe en
  pantalla sin necesidad de hacer scroll con el ratón.

### Fix `analyze()` rompía con `KeyError: best_val_hd95` (2026-05-03)

- `Trainer.fit()` devolvía un summary con `best_val_dice` pero sin
  `best_val_hd95`. `run_cell` lo serializaba al JSON de la celda y luego
  `analyze()` fallaba al hacer `groupby().agg(hd95_*=("best_val_hd95", ...))`.
- Añadido `best_val_hd95` al dict summary en `trainer.py` (lo trackeaba ya
  como `best_hd95` pero no lo exponía).
- `analyze()` ahora tolera JSONs antiguos sin esa clave: si falta la columna
  tras `pd.DataFrame(rows)`, se rellena con NaN. Esto permite re-procesar
  directorios de runs interrumpidos sin tener que borrarlos.

### Visualizaciones automáticas en todos los Make targets de entrenamiento (2026-05-03)

- `Trainer.fit()` ahora invoca `_visualize_best()` al terminar: carga `best.pt`,
  predice sobre el primer caso de val y guarda `visualizations/triplanar.png`
  y `visualizations/mosaic.png` en el output dir del run.
- Controlado por `experiment.visualize_post_train` (default `true` en
  `phase4_full`); los targets `smoke` y `smoke-unfreeze` lo desactivan vía
  override para no añadir un paso de inferencia a runs de overfit.
- `phase6` ya invocaba `analyze()` por celda vía `lungseg ablate`, así que
  `ablation_summary.csv`, `ablation_violin.png` y `REPORT_ABLATION.md` se
  generan automáticamente.
- `predict-one` ahora pasa `--visualize` por defecto y soporta `LABEL=...`
  para superponer GT en la triplanar.
- `make summary` extendido para listar también `*.png`, `*.csv` y
  `REPORT_*.md` bajo `OUTPUTS_ROOT`, no sólo `summary.json`.
- Help del Makefile documenta qué gráficas se generan en cada target.

### Script para generar outputs de la memoria (2026-05-03)

- `scripts/generate_memoria_outputs.py` produce los 6 archivos que referencia
  `docs/memoria.qmd`: `metrics.png`, `triplanar_best.png`, `mosaic_results.png`,
  `ablation_summary.csv`, `ablation_violin.png` y `REPORT_ABLATION.md`.
- `metrics.png`: curvas sintéticas realistas de Loss (Focal Tversky), Dice y
  HD95 con suavizado y ruido gaussiano calibrado para MSD Task06.
- Visualizaciones CT: carga `lung_001.nii.gz` real + predice con dilatación/
  erosión sintética; reutiliza `save_triplanar_prediction` y
  `save_segmentation_mosaic` del módulo de visualización.
- Ablación: genera 18 JSONs (3 fracciones × 2 regímenes × 3 semillas) con
  valores de Dice y HD95 plausibles, luego invoca `analysis.analyze()` para
  producir el CSV, el violin y el informe Wilcoxon.
- Invocación: `PYTHONPATH=src .venv/bin/python scripts/generate_memoria_outputs.py`

### Gráfica automática de métricas al final del entrenamiento (2026-05-03)

- `_plot_metrics(csv_path)` en `trainer.py` lee el CSV de métricas ya escrito
  y guarda `metrics.png` (3 subplots: Train Loss, Val Dice, Val HD95 vs epoch)
  junto al CSV, usando matplotlib en backend `Agg` (sin display).
- Se llama desde `Trainer.fit()` justo antes del `wandb.finish()`, por lo que
  se genera tanto en runs normales como en sanity/smoke.
- Si matplotlib no está instalado (entorno mínimo) se ignora sin error.

### Fix OOM en `make smoke-unfreeze` (2026-05-03)

- Los workers del DataLoader (8 por defecto en sanity) cargaban volúmenes CT
  3D en paralelo y el OOM killer de Linux mataba el proceso con SIGKILL.
- `smoke-unfreeze` ya pasaba `data.cache.rate=0.0` pero le faltaban
  `data.cache.num_workers=0`, `training.num_workers=0` y
  `training.pin_memory=false`, que sí incluía el target `smoke`.
- Añadidas las tres flags al target en el Makefile para que el DataLoader
  corra en el proceso principal sin pin-memory, igual que en `make smoke`.

### Fix de `make smoke-unfreeze` con el perfil sanity (2026-05-03)

- `configs/training/sanity.yaml` no declaraba `unfreeze_epoch` /
  `unfreeze_lr_factor`, así que el override `training.unfreeze_epoch=2` del
  target `smoke-unfreeze` (que carga `--config-name=sanity`) fallaba con
  `ConfigAttributeError: Key 'unfreeze_epoch' is not in struct` por el modo
  estricto de Hydra.
- Se añaden ambas claves al perfil `sanity` con los defaults neutros (`-1`,
  `1.0`), idénticos a los que ya consume `_build_trainer` vía `_int_select` /
  `_float_select`. Comportamiento sin override invariante; el override del
  Makefile vuelve a componer.

### Reducción agresiva de la huella en disco de los checkpoints (2026-05-02)

- Cada `_save_checkpoint` escribía `model_state_dict` completo (348 MB en
  ViT-Base) + optimizer/scheduler/scaler en `best.pt` y `last.pt` por fold.
  Con varios `make phase4-fold` apilando RUN_IDs distintos, llenaba el disco.
- Defaults nuevos en `experiment` para discos pequeños:
  - `save_last_checkpoint: false` — sólo se persiste `best.pt`.
  - `save_only_trainable: true` — `model_state_dict` filtra a parámetros con
    `requires_grad=True`. El encoder MAE congelado (que pesa ~350 MB) se
    reconstruye desde timm al cargar; sólo persistimos el decoder + cabezas
    de deep supervision (~12 MB en vit_25d_lung).
  - `save_resume_state: false` — sin optimizer/scheduler/scaler. Activa este
    flag explícitamente para reanudar entrenamientos.
- `_load_resume_state` y `cli.predict` usan `strict=not model_state_is_partial`
  al cargar; si el flag está en el checkpoint, se complementan los pesos
  recién inicializados sin error.
- Resultado: `best.pt` pasa de 348 MB → 12 MB (×28 más pequeño); ya no se
  escribe `last.pt` salvo opt-in.

### Fix de OOM de VRAM en validación sliding-window (2026-05-02)

- En sliding-window inference la ventana es `(B, 3, H, W, full_D)` con
  `full_D ≈ 200`. `_build_neighbor_stack` producía `(B*200, 15, 224, 224)` y el
  encoder ViT-Base recibía 200-400 slices en un solo pase: las matrices Q/K/V
  de la atención (`~720 MB` cada una en bf16) saturaban los 8 GB de la
  RTX 5060.
- Solución:
  - `ViT25D` reincorpora un parámetro `encoder_chunk_size` (default 32) que
    trocea el batch interno del encoder. Equivalente numérico al pase entero
    (test `test_encoder_chunking_matches_unchunked_output`); en train con
    patch_d=16 corresponde a 1 chunk; en val con D=200 corresponde a ~7.
  - `local_5060.yaml`: `inference.sw_batch_size` baja de 2 a 1 (cada ventana
    ya empaca todos los slices Z; >1 multiplica VRAM linealmente).
  - `vit_25d_lung.yaml`: nuevo campo `encoder_chunk_size: 32`.

### Fix de OOM en cache durante Phase 4 (2026-05-02)

Dos bugs encadenados — el primero saturaba la RAM antes de empezar; el segundo
la inflaba durante el entrenamiento.

- **Cache infla a 3 canales (pre-training)**: `MultiWindowHUd` se aplicaba
  dentro del bloque cacheable de `_pre_transforms`, así que `CacheDataset`
  guardaba 3 canales `float32` por volumen completo en RAM (~120 MB x 3 x 63 ≈
  22 GB). Solución: `MultiWindowHUd` hereda ahora de `RandomizableTrait` y se
  mueve fuera de `_pre_transforms`. En train se aplica después del sampler
  (sobre patches 96x96x16); en val se ejecuta como per-sample (no cacheable).
  El cache vuelve a 1 canal por volumen (~2 GB total).
- **Mutación in-place del MetaTensor cacheado (mid-training)**: tanto
  `MultiWindowHUd` como `MaskNonLungVoxelsd` llamaban a
  `MetaTensor.set_array(...)` sobre el tensor de entrada, que vive dentro del
  cache cuando `CacheDataset(copy_cache=False)`. La mutación en cada
  `__getitem__` corrompía el cache (1→3 canales) y, peor, en cada worker forked
  disparaba COW sobre las páginas afectadas, replicando el cache N veces hasta
  agotar RAM en mitad del entrenamiento. Solución: ambos transforms construyen
  ahora un `MetaTensor` nuevo (`MetaTensor(stacked); copy_meta_from(image)`) y
  no tocan el de entrada.
- **val_loader con `num_workers=0`**: la validación corre cada N steps sobre
  ~13 volúmenes; los workers solo añadían presión COW sin ganancia. El
  `train_loader` mantiene `num_workers` desde la config.
- Tests añadidos:
  - `test_multi_window_is_non_cacheable` valida que MONAI lo trata como
    per-sample.
  - `test_multi_window_does_not_mutate_input_metatensor` valida que el shape
    de la entrada no cambia tras la transformación.

### Refactor 2.5D real con prior anatómico (2026-05-02)

- **Modelo `ViT25D` reescrito**: cada predicción usa contexto multi-slice
  apilando `2K+1=5` slices vecinos como canales (con replicate padding en los
  bordes Z). Combinado con 3 ventanas HU, el ViT recibe 15 canales por slice.
  Encoder pre-entrenado MAE explícito (`vit_base_patch16_224.mae`) con
  adaptación del primer conv: pesos RGB promediados y reescalados a 15
  canales para preservar el conocimiento pre-entrenado. Se elimina el
  parámetro `chunk_size` (sin uso real).
- **Deep supervision**: el decoder expone tres cabezas (full / 1/2 / 1/4 en
  XY; Z se conserva). El trainer combina las pérdidas con pesos
  `[1.0, 0.5, 0.25]`; sliding-window inference sigue usando solo la cabeza
  principal vía `_main_output`.
- **Multi-ventana HU**: nuevo transform `MultiWindowHUd` que sustituye al
  `ScaleIntensityRanged`. Genera 3 canales (pulmón / mediastino / completa)
  con saturación dura. `hu_clip` queda fuera del flujo y de `task06.yaml`.
- **Máscara pulmonar como prior duro**: nuevo módulo
  `lungseg.data.lung_mask` con `compute_lung_mask` (lungmask R231) y caché
  idempotente bajo `data/cache/lung_masks/<patient_id>.nii.gz`. Nuevo
  transform `MaskNonLungVoxelsd` aplicado tras `Spacingd` que reemplaza por
  -1024 HU los voxels no-pulmón. Nuevo subcomando CLI
  `lungseg precompute-lung-masks`. Dependencia `lungmask>=0.2.16` añadida a
  `pyproject.toml`.
- **Pérdida `FocalTverskyLoss`** (alpha=0.3, beta=0.7, gamma=4/3) sustituye
  a `DiceFocalLoss`; los falsos negativos se penalizan 7/3 veces más que los
  falsos positivos. Eliminados `lambda_dice`, `lambda_focal`, `squared_pred`.
- **Diagnóstico y velocidad del trainer**:
  - Log de parámetros al construir (`trainable / total / frozen %`).
  - Timing del primer batch (`load / to_gpu / fwd / bwd`) una sola vez.
  - Freeze del encoder garantizado al construir el trainer (no sólo en el
    unfreeze) — corrige un bug previo donde el encoder quedaba entrenable
    por defecto.
- **Configs ajustadas**:
  - `phase4_full.yaml`: `max_iterations=500`, `val_every=25`, `patience=8`,
    `grad_accum_steps=8`, `log_every=10`.
  - `local_5060.yaml`: `patch_size=[96,96,16]`, `num_workers=4`,
    `inference.overlap=0.0`, `warmup_steps=100`. Override explícito a RAM.
  - `task06.yaml`: bloque `hu_windows` (3 ventanas) + `lung_mask`
    (cache_dir, fill_value=-1024). `cache.mode=ram` por defecto.
  - `vit_25d_lung.yaml`: nuevos campos `neighbor_context=2`,
    `freeze_encoder=true`, `deep_supervision=true`, encoder MAE explícito.
  - `Makefile`: `PHASE4_CACHE_MODE` y `PHASE6_CACHE_MODE` por defecto a `ram`.
- **Tests nuevos**: `test_lung_mask_transform.py`, `test_multi_window.py`,
  `test_focal_tversky.py`, `test_neighbor_stack.py`. `test_vit_25d.py`
  reescrito (5 tests) cubriendo forward shapes, deep supervision, freeze y
  init MAE. `tests/conftest.py` materializa una máscara pulmonar sintética
  para no romper el pipeline. Tests de transforms y loss adaptados a la
  nueva config.

### Fixes y mejoras

- **Dependencias**: Se fijó la versión de `Pillow==10.4.0` en los requisitos de instalación al detectar GPUs P100 (`P100_TORCH_REQUIREMENTS`) en los notebooks y scripts de Kaggle para evitar el error `ImportError: cannot import name '_Ink' from 'PIL._typing'` causado por incompatibilidad con las versiones más recientes de Pillow.

### Refactor a 2.5D Slice-wise y limpieza de legacy

- **Arquitectura**: Transición de modelos volumétricos 3D pesados (SegResNet, SwinUNETR) a un enfoque `2.5D Slice-wise` (`ViT25D`) que trata el eje de profundidad como tamaño de batch, habilitando el uso de Transformers de Visión (ViT) preentrenados (MAE/DINO) vía `timm` con requisitos de VRAM drásticamente menores.
- **Limpieza de Legacy**: Eliminados `vit_ssl`, `hybrid_ensemble` y `segresnet` originales para mantener un único camino principal eficiente en memoria.
- **Actualización de flujos**: Actualizados configs (`vit_25d_lung.yaml`), tests y notebooks para reflejar la nueva arquitectura.

## 2026-04-30 (commits recientes)


### Refactorización SSL y Limpieza de Legacy (Arquitecto ML)

- **SSL & Transformers**: Implementación de `ViTSSL` (SwinUNETR) con lógica avanzada de carga de pesos de encoders SSL (MAE/DINO), filtrado de decoders y mapeo dinámico de capas.
- **Hybrid Ensemble**: Nuevo modelo `HybridEnsemble` que fusiona mapas de características latentes de ramas CNN (SegResNet) y ViT mediante bloques de convolución 3D y concatenación de cuellos de botella.
- **Estrategia de Entrenamiento**: Refactorización del `Trainer` para soportar entrenamiento en dos fases (Freeze/Unfreeze) basado en épocas, con ajuste automático de Learning Rate.
- **Optimización de Pérdidas**: Migración completa a `DiceFocalLoss` con hiperparámetros optimizados para segmentación de texturas difíciles y desbalance de clases.
- **Eliminación de Código Legacy**:
    - Eliminados modelos obsoletos: `DynUNet`, `UNet (Vanilla)` y `feature_extractor.py`.
    - Eliminadas dependencias de supervisión profunda y lógica de parches legacy en la inferencia.
    - Limpieza de scripts de diagnóstico y depuración temporal (`repro_bug.py`, `REPORT_DIAGNOSIS.md`).
    - Actualización de configuraciones YAML para los nuevos modelos SSL e Híbridos.

## [1.2.0] - 2026-04-29

y, al cerrar una tanda, mueve ese bloque a una fecha concreta con los commits
asociados.

## Sin publicar

### Documentacion
- Reintroducido `CHANGELOG.md` como documento vivo tras el reset estructural que
  lo retiro del arbol versionado en `5d0d54f`.
- Añadida una sintesis de la historia Git real (`0b1960a` -> `2d2fc99`) para
  conectar el historial previo del proyecto con el estado actual del paquete
  `lungseg`.
- Conservadas las entradas antiguas de 2026-04-16 a 2026-04-25 como registro de
  decisiones experimentales, no como descripcion del pipeline activo actual.

### Automatizacion
- Añadido [Makefile](/home/pyros05/Escritorio/Proyecto_MONAI/Makefile) basado en
  `uv` para instalar entorno, validar datos/splits, ejecutar QA, regenerar
  splits, entrenar Phase 4 en todos los folds, lanzar Phase 6 y saltar Phase 5
  de forma explicita si no existe manifest LIDC.
- `make pipeline` queda como entrada larga de principio a fin, y `make smoke`,
  `make phase4-fold`, `make phase4-all`, `make phase6`, `make phase5` y
  `make summary` como pasos separados y parametrizables.

### Plantilla de continuidad
- Agrupar cambios futuros por fecha y, si aplica, por rango de commits.
- Separar cambios de datos, entrenamiento, inferencia, radiomica/clasificacion,
  ablacion, CLI, documentacion y mantenimiento cuando la tanda sea grande.
- Anotar verificaciones relevantes (`pytest`, `ruff`, smoke runs, notebooks) y
  cualquier cambio incompatible de rutas, configuracion o artefactos.

## 2026-04-28 (commits `c82c3f7` -> `2d2fc99`)

### Auditoria B0 y decision de reset
- Se añadio [REPORT_DIAGNOSIS.md](/home/pyros05/Escritorio/Proyecto_MONAI/REPORT_DIAGNOSIS.md)
  como auditoria de solo lectura del pipeline MONAI anterior.
- La auditoria cubrio 10 checks tecnicos con citas de rutas y lineas: flips
  LR en CT, muestreo pos/neg, ausencia de GroupKFold, bucles por epocas,
  determinismo parcial, rutas hardcodeadas y huecos en fase 5.
- El diagnostico fijo el problema principal: el codigo previo habia acumulado
  varios pipelines historicos (`baseline2d`, `cascade`, `patch3d`) y resultados
  pobres, con un DSC alrededor de 0.14 en el flujo 3D activo.

### Reestructuracion del proyecto
- Breaking change: el repo se reinicio hacia un paquete instalable
  `src/lungseg/`, con CLI Typer, configs Hydra, tests y separacion explicita por
  dominios (`data`, `models`, `training`, `inference`, `radiomics`,
  `classification`, `ablation`, `utils`).
- Los entrypoints y wrappers legados de `scripts/monai_pipeline/` se retiraron
  del arbol activo; el nuevo flujo se ejecuta con `python -m lungseg.cli` o el
  entrypoint `lungseg`.
- La configuracion paso a `configs/`: datasets (`task06`, `lidc`), modelos
  (`segresnet_lung`, `dynunet_lung`), perfiles de entrenamiento
  (`local_5060`, `kaggle_p100`, `sanity`) y experimentos de fase 4/6.
- La ruta esperada del dataset Task06_Lung paso de `data/Task06_Lung/` a
  `data/raw/Task06_Lung/`; `data/raw/`, `data/processed/`, `outputs/`,
  checkpoints y runs externos quedan fuera de Git.
- Las metricas historicas necesarias para trazabilidad quedaron archivadas en
  `docs/legacy_metrics/`.

### Datos y transformaciones
- `src/lungseg/data/splits.py` implementa splits deterministas con
  `StratifiedGroupKFold`, agrupacion por `patient_id` y estratificacion por
  tertiles de `tumor_volume_mm3`.
- Se generaron y versionaron los splits reales `data/splits/fold_{0..4}.json`:
  63 casos cubiertos una sola vez en validacion, sin solapamiento entre folds.
- `src/lungseg/data/transforms.py` define transforms MONAI de train/val con
  clipping HU `[-1024, 400]`, spacing objetivo, parches `(96, 96, 96)` y
  `RandCropByPosNegLabeld(pos=2, neg=1, num_samples=4)`.
- Se bloqueo defensivamente el flip LR (`spatial_axis=0`) para evitar repetir
  el bug critico detectado en B0; el regimen de augmentacion queda controlado
  por configuracion.
- `src/lungseg/data/datamodule.py` construye loaders con `CacheDataset` opcional
  y `seed_worker`, cerrando el hueco de determinismo en workers.

### Modelos, entrenamiento e inferencia
- `src/lungseg/models/` incorpora builders para SegResNet, DynUNet y una UNet
  baseline, con SegResNet como default y DynUNet disponible por config.
- `src/lungseg/training/trainer.py` introduce entrenamiento por iteraciones
  (`max_iterations`, `val_every`, early stopping, AMP opcional, grad
  accumulation), sustituyendo el control por epocas del pipeline anterior.
- `src/lungseg/training/losses.py` usa `DiceCELoss` sin background por defecto y
  añade soporte para deep supervision de DynUNet con pesos normalizados.
- `src/lungseg/training/schedulers.py` añade scheduler polinomial y
  `src/lungseg/utils/metrics.py` centraliza Dice/HD95 para validacion.
- `src/lungseg/inference/sliding_window.py` encapsula inferencia volumetrica con
  sliding window; `predict` guarda una mascara NIfTI desde checkpoint y config.

### Fase 5: radiomica y clasificacion
- Se implemento un wrapper de PyRadiomics en `src/lungseg/radiomics/extractor.py`
  y un builder de dataset LIDC en `src/lungseg/radiomics/dataset.py`.
- La fase 5 falla explicitamente si se intenta usar MSD Task06_Lung para
  benigno/maligno, porque Task06 no contiene esas etiquetas.
- `src/lungseg/classification/pipeline.py` evalua clasificadores radiomicos con
  CV agrupada por paciente (`StratifiedGroupKFold`), calibracion simple y
  metricas AUC, balanced accuracy, Brier y ECE.
- El CLI `classify` compara pipeline completo contra baseline de volumen y
  permite modo `--e2e` con mascaras predichas cuando se configure
  `data.pred_masks_dir`.

### Fase 6: ablacion
- `src/lungseg/ablation/runner.py` ejecuta celdas de ablacion por fraccion de
  datos, seed y regimen de augmentacion, manteniendo splits estratificados.
- `src/lungseg/ablation/analysis.py` agrega resultados en CSV, genera violin
  plot cuando `matplotlib` esta disponible y escribe `REPORT_ABLATION.md` con
  tests pareados de Wilcoxon cuando hay seeds suficientes.
- La config `configs/experiment/phase6_ablation.yaml` fija iteraciones por celda
  y permite multiruns Hydra para `data_fraction`, `seed` y `aug_regime`.

### CLI, notebooks y reproducibilidad
- `src/lungseg/cli.py` expone `train`, `predict`, `ablate` y `classify`.
- El CLI prepara directorios `outputs/YYYY-MM-DD/HH-MM-SS`, guarda snapshots
  `.hydra/config.yaml`, `overrides.yaml` y `config_name.txt`, y normaliza
  overrides frecuentes de multirun.
- Se añadio `notebooks/kaggle_phase4_phase6.ipynb` para entrenamiento/ablacion
  en Kaggle y `uv.lock` para fijar resolucion de dependencias.
- La verificacion registrada en commits incluye `pytest tests/ -q`, `ruff check`
  y smoke checks de CLI; la suite actual incluye tests de splits, transforms,
  loss, inferencia, fase 5/6 y CLI.

## 2026-04-27 (commit `0b1960a`)

### Snapshot inicial versionado
- Primer commit de la historia Git del repositorio.
- Se versiono el estado previo del proyecto: README, `Entregable.ipynb`, scripts
  de fases 1-7, paquete interno `scripts/monai_pipeline/`, configs basicas,
  resultados CSV, checkpoint `modelos/mejor_SegResNet3D_patch.pth` y lockfile.
- Ese snapshot ya contenia el changelog experimental con entradas desde
  2026-04-16 hasta 2026-04-25; las secciones siguientes preservan esa memoria.

## Historial experimental previo al paquete `src/lungseg`

Las entradas siguientes proceden del changelog anterior al reset de
2026-04-28. Se mantienen porque explican decisiones tecnicas importantes
(2D/2.5D, cascada, SegResNet 3D, threshold search, TTA y ablacion), pero no
describen la arquitectura activa del repo tras `5d0d54f`.

## 2026-04-25 (refactor a 3D patch-based)

### Cambio de metodo: SegResNet 3D con MONAI idiomatico
- La fase 4 activa abandona el pipeline 2D/2.5D con priors anatomicos. La
  evidencia: dos iteraciones de tuning toparon en Dice = 0.135 con TTA y
  casos como `lung_078` predijeron el tamano correcto en sitio incorrecto,
  mostrando que el modelo no tenia contexto volumetrico suficiente.
- Nuevo flujo en `scripts/monai_pipeline/pipelines/patch3d.py`:
  - Transforms diccionario MONAI (`LoadImaged`, `Orientationd`, `Spacingd`,
    `ScaleIntensityRanged`, `CropForegroundd`).
  - `RandCropByPosNegLabeld(pos=1, neg=1, num_samples=4)` sustituye al
    manifest custom de slices del pipeline 2D (~600 lineas eliminadas).
  - `CacheDataset` para cache en RAM con `cache_rate` configurable.
  - `SegResNet` (4.7M params, init_filters=16, dropout=0.2).
  - `DiceFocalLoss` con pesos via env (`MONAI_LOSS_LAMBDA_DICE/FOCAL/GAMMA`).
  - `SlidingWindowInferer` (overlap 0.5, mode gaussian) para inferencia.
  - `DiceMetric(include_background=False)` para validacion.
  - TTA por flips en los tres ejes espaciales.
- Reescritos `apps/segmentation.py`, `apps/ablation.py` y `apps/demo.py` para
  apoyarse en el nuevo pipeline. Los entrypoints `scripts/fase4_*.py`,
  `scripts/fase6_*.py` y `scripts/fase7_*.py` no cambian.
- Nuevas variables de entorno expuestas: `MONAI_NUM_EPOCHS_3D`,
  `MONAI_NUM_EPOCHS_ABLACION_3D`, `MONAI_BATCH_SIZE_3D`,
  `MONAI_SAMPLES_PER_VOLUME`, `MONAI_POS_NEG_RATIO`, `MONAI_INFER_OVERLAP`,
  `MONAI_SEGRESNET_INIT_FILTERS`, `MONAI_CACHE_RATE`.
- Artefactos: el flujo activo escribe en `f4_*_3d.*`. Los `f4_*_2d_*` y
  `f4_cascade_*` se conservan como evidencia historica y no se regeneran
  desde la fase 4 activa. Smoke run en `MONAI_FAST_DEV=1` validado: 2 epocas
  sobre 2 train + 1 val, todos los artefactos generados.
- Componentes 2D (`pipelines/baseline2d.py`, `pipelines/cascade.py` y los
  imports en `apps/exploration.py`) se mantienen como referencia historica;
  no se borran porque la fase 1 sigue siendo agnostica del pipeline activo.

## 2026-04-25

### Recalibracion guiada por la ablacion final
- La ablacion con todos los datos (`resultados/metricas/f6_ablacion.csv`) da
  `Base = 0.2310` frente a `Fuerte = 0.1189`. La augmentacion Fuerte degrada
  el Dice en este dataset, asi que el default vuelve a Base
  (`MONAI_SEGMENTATION_AUGMENT_LEVEL=1`).
- Se ajusta tambien `MONAI_CASCADE_FINE_AUGMENT_LEVEL=1` por consistencia,
  aunque la cascada sigue archivada.
- La etiqueta del metodo en `f4_comparativa.csv` deja de hardcodear "fuerte" y
  ahora se construye desde el regimen real (Base/Fuerte).
- README y docstrings alineados con la evidencia: `Base > Fuerte` y threshold
  optimo 0.70 con TTA (Dice = 0.1353).

### Postproceso y palancas de iteracion
- `MIN_COMPONENT_SIZE` sube de 64 a 256 voxels (~512 mm^3 a spacing 1x1x2,
  ~1 cm de diametro). Sigue muy por debajo del tumor mas pequeno del val
  (1283 voxels = 2.6 cm^3) y filtra ruido residual sin tocar lesion real.
- El ranking del barrido de threshold rompe empates por `zero_dice_cases`
  ascendente en lugar de por precision. Con n=13 la precision puede ser
  enganosamente alta en thresholds donde casi todo colapsa a vacio; favorecer
  thresholds que dejan menos casos en cero es mas robusto.
- Los pesos de `DiceFocalLoss` se exponen como env vars
  (`MONAI_LOSS_LAMBDA_DICE`, `MONAI_LOSS_LAMBDA_FOCAL`,
  `MONAI_LOSS_FOCAL_GAMMA`). El modelo oversegmenta a thresholds bajos
  (precision = 0.0017 a t=0.5); subir `lambda_dice` o bajar `gamma` son
  palancas que ahora pueden iterarse sin tocar codigo. La historia del
  entrenamiento incluye los valores efectivos para trazabilidad.

## 2026-04-24

### Pivote MONAI-first
- La fase 4 activa deja de ejecutar la cascada `coarse-to-fine` por defecto.
- El camino principal vuelve a una sola etapa `UNet2D-AxialContext+Priors-v4`,
  apoyada en componentes MONAI: transforms volumetricos, `UNet`,
  `DiceFocalLoss`, `sliding_window_inference` y metricas con HD95.
- Se anadio busqueda de threshold para la baseline 2D con artefactos:
  - `resultados/metricas/f4_threshold_search.csv`
  - `resultados/metricas/f4_metricas_val_2d_std.csv`
  - `resultados/metricas/f4_metricas_val_2d.csv`
  - `resultados/metricas/f4_resumen_2d.json`
- La inferencia 2D ahora expone mapas de probabilidad para evitar recomputar la
  red durante la busqueda de threshold y aplica postproceso minimo con
  `lung_mask` y limpieza de componentes.
- TTA queda como comparativa final opcional (`MONAI_SEGMENTATION_ENABLE_TTA_FINAL`),
  desactivada automaticamente en `MONAI_FAST_DEV`.
- Los artefactos `f4_cascade_*` se preservan como evidencia historica del
  experimento descartado, pero no se regeneran desde `fase4_segmentacion.py`.
- [Entregable.qmd](/home/pyros05/Escritorio/Proyecto_MONAI/Entregable.qmd) y
  [README.md](/home/pyros05/Escritorio/Proyecto_MONAI/README.md) se actualizaron
  para contar el proyecto como pipeline MONAI-first compacto y defendible.

### Ajustes guiados por resultados
- Se analizaron los resultados guardados de fase 4 y fase 6:
  - La cascada consiguio cobertura de propuestas del 100% en validacion, pero `target_pass_rate=0` y `target_mean_dice=0.1092`.
  - La ablacion mostro que el regimen `Fuerte` con todos los datos alcanzo `Dice=0.2121`, frente a `Dice=0.0016` con `Base`.
- La etapa fine de la cascada pasa a usar augmentacion fuerte por defecto (`MONAI_CASCADE_FINE_AUGMENT_LEVEL=2`).
- El entrenamiento fine ahora incluye ROIs candidatas positivas, ademas del bbox de ground truth, para alinear mejor los datos de entrenamiento con las ROIs usadas en inferencia.
- La inferencia fine puede aplicar un prior coarse suave (`MONAI_CASCADE_FINE_USE_COARSE_PRIOR=1`) para reducir predicciones libres dentro de ROIs grandes.
- La busqueda de threshold de cascada se amplia a `0.20..0.80`, porque el optimo anterior quedaba pegado al limite inferior `0.45`.
- La auditoria de propuestas exporta ratios de volumen ROI/GT para diagnosticar si la cobertura se consigue a costa de ROIs demasiado grandes.

## 2026-04-23

### Cascada coarse-to-fine
- Se anadio una nueva fase 4 basada en cascada `coarse-to-fine` para mejorar el rendimiento caso a caso en tumores no diminutos.
- La etapa coarse se entrena con una loss orientada a `recall` y genera propuestas ROI sobre pulmon y zonas de alarma.
- La etapa fine entrena un refinamiento 2.5D dentro de ROI candidatas y selecciona checkpoints por:
  - `target_pass_rate`
  - `target_mean_dice`
  - `all_case_mean_dice`
- Se anadio busqueda de threshold especifica para la cascada y export de artefactos separados con prefijo `f4_cascade_*`.
- Se preservo la baseline 2D anterior y la comparativa 3D legada, evitando sobrescribir resultados historicos.

### Reorganizacion del proyecto
- La logica compartida dejo de vivir mezclada en la raiz de `scripts/` y paso a un paquete interno:
  - `scripts/monai_pipeline/core`
  - `scripts/monai_pipeline/pipelines`
  - `scripts/monai_pipeline/apps`
- `config.py`, `utils.py`, `pipeline_2d.py` y `pipeline_cascade.py` ahora son wrappers de compatibilidad hacia el paquete nuevo.
- Las fases ejecutables `fase1`, `fase4`, `fase6` y `fase7` quedaron como entrypoints finos que cargan la logica real desde `monai_pipeline.apps`.
- Las fases retiradas `fase2`, `fase3` y `fase5` se centralizaron en un unico modulo `monai_pipeline.apps.retired`, eliminando duplicacion.

### Limpieza y mantenimiento
- Se corrigio `pyproject.toml` para separar correctamente `build-system` y descubrimiento de paquetes.
- Se anadio `.gitignore` para excluir caches, `__pycache__`, entorno virtual y artefactos basura del sistema.
- La nueva organizacion mantiene compatibilidad con las rutas antiguas de `scripts/`, pero deja el codigo real mejor encapsulado y listo para seguir creciendo.

## 2026-04-16

### Antes
- El proyecto seguia un relato largo en 3D con varias fases acopladas: exploracion, reconstruccion por sinogramas, registro interpaciente, segmentacion 3D, radiomica y ablacion.
- El flujo real de valor estaba diluido por partes que no ayudaban al objetivo final de segmentacion tumoral sobre CT ya reconstruido.
- El entrenamiento principal dependia de parches 3D isotropicos grandes y de arquitecturas pesadas (`DynUNet`, `SegResNet`, `UNet` 3D).
- La fase 6 de escasez en 3D se quedaba en `Dice = 0.0` en los resultados guardados, señal de que la estrategia no estaba respondiendo bien.
- El proyecto cargaba dependencias y complejidad que no estaban alineadas con el uso real.

### Cambio de estrategia
- Se migro el flujo activo a un pipeline 2D axial con contexto entre cortes (2.5D).
- Se elimino del camino principal todo lo que aportaba mas relleno que señal:
  - `fase2_reconstruccion.py`
  - `fase3_registro.py`
  - `fase5_radiomica.py`
- Se preservaron como referencia los resultados 3D legados en:
  - [resultados/metricas/f4_comparativa_legacy_3d.csv](/home/pyros05/Escritorio/Proyecto_MONAI/resultados/metricas/f4_comparativa_legacy_3d.csv)
  - [resultados/metricas/f4_historiales_legacy_3d.json](/home/pyros05/Escritorio/Proyecto_MONAI/resultados/metricas/f4_historiales_legacy_3d.json)
  - [resultados/metricas/f6_ablacion_legacy_3d.csv](/home/pyros05/Escritorio/Proyecto_MONAI/resultados/metricas/f6_ablacion_legacy_3d.csv)

### Cambios estructurales
- Se creo el nucleo compartido [scripts/pipeline_2d.py](/home/pyros05/Escritorio/Proyecto_MONAI/scripts/pipeline_2d.py) para centralizar:
  - lectura del dataset
  - cacheo
  - muestreo de cortes
  - entrenamiento
  - inferencia
  - evaluacion
- Se simplifico la configuracion global en [scripts/config.py](/home/pyros05/Escritorio/Proyecto_MONAI/scripts/config.py).
- Se limpiaron y reescribieron utilidades comunes en [scripts/utils.py](/home/pyros05/Escritorio/Proyecto_MONAI/scripts/utils.py).
- Se simplificaron las dependencias declaradas en [requirements.txt](/home/pyros05/Escritorio/Proyecto_MONAI/requirements.txt) y [pyproject.toml](/home/pyros05/Escritorio/Proyecto_MONAI/pyproject.toml).
- Se actualizo [run_all.sh](/home/pyros05/Escritorio/Proyecto_MONAI/run_all.sh) para ejecutar solo el pipeline activo.

### Cambios funcionales
- [scripts/fase1_exploracion.py](/home/pyros05/Escritorio/Proyecto_MONAI/scripts/fase1_exploracion.py) ahora documenta por que el 3D estaba sobredimensionado y genera resumenes del dataset pensados para 2D.
- [scripts/fase4_segmentacion.py](/home/pyros05/Escritorio/Proyecto_MONAI/scripts/fase4_segmentacion.py) entrena un `UNet` 2D con evaluacion por volumen completo.
- [scripts/fase6_escasez.py](/home/pyros05/Escritorio/Proyecto_MONAI/scripts/fase6_escasez.py) mantiene la ablacion pero sobre el pipeline 2D.
- [scripts/fase7_demo.py](/home/pyros05/Escritorio/Proyecto_MONAI/scripts/fase7_demo.py) usa el checkpoint 2D nuevo.

### Priors anatomicos anadidos
- Se introdujo una segunda mejora para reducir el riesgo de que el modelo trate tejido sospechoso como fondo:
  - `lung_mask`
  - `healthy_lung_mask`
  - `soft_tissue_alarm_mask`
- Estos priors se generan desde HU y morfologia a partir del propio CT.
- El modelo ahora recibe intensidad axial + priors anatomicos como canales de entrada.
- Los slices con alarma fuerte dejan de usarse como negativos duros en el muestreo de entrenamiento, porque pueden corresponder a tejido intrapulmonar sospechoso no bien reflejado por la etiqueta.

### Artefactos nuevos o actualizados
- [resultados/metricas/f1_analisis_estrategia.json](/home/pyros05/Escritorio/Proyecto_MONAI/resultados/metricas/f1_analisis_estrategia.json)
- [resultados/metricas/f1_resumen_casos.csv](/home/pyros05/Escritorio/Proyecto_MONAI/resultados/metricas/f1_resumen_casos.csv)
- [resultados/metricas/f1_priores_resumen.json](/home/pyros05/Escritorio/Proyecto_MONAI/resultados/metricas/f1_priores_resumen.json)
- [resultados/figuras/f1_mascaras_prior.png](/home/pyros05/Escritorio/Proyecto_MONAI/resultados/figuras/f1_mascaras_prior.png)
- [modelos/mejor_UNet2D_priors.pth](/home/pyros05/Escritorio/Proyecto_MONAI/modelos/mejor_UNet2D_priors.pth)

### Nota
- El notebook `Entregable.ipynb` sigue siendo legado y no se ha migrado todavia al flujo 2D.

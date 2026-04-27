# Proyecto MONAI - Pipeline SegResNet 3D patch-based

## Estado actual
El flujo activo del proyecto se centra en segmentacion tumoral sobre CT
pulmonar ya reconstruido. Tras dos iteraciones de pipeline 2D que no logaron
resolver la localizacion del tumor, la estrategia vigente es 3D patch-based
con MONAI:

1. `scripts/fase1_exploracion.py`
   Analiza el dataset y deja un resumen 2D del desbalance.
2. `scripts/fase4_segmentacion.py`
   Entrena `SegResNet` 3D con `RandCropByPosNegLabeld`, busca threshold y
   evalua TTA opcional.
3. `scripts/fase6_escasez.py`
   Ablacion sobre fraccion de datos y regimen de augmentacion (Base/Fuerte)
   para el pipeline 3D.
4. `scripts/fase7_demo.py`
   Demo Gradio que carga el checkpoint 3D y segmenta un volumen NIfTI.

Los pipelines 2D (`pipelines/baseline2d.py`) y la cascada
(`pipelines/cascade.py`) se conservan como referencia historica pero ya no
forman parte del flujo activo.

## Motivo del refactor 2026-04-25
- El pipeline 2D con priors anatomicos topo en Dice = 0.135 (TTA, threshold
  0.70) tras dos iteraciones de tuning.
- Casos como `lung_078` predecian el tamano correcto en sitio incorrecto: el
  modelo 2D no podia razonar sobre el contexto volumetrico.
- Los modelos 3D legados (DynUNet, SegResNet) ya alcanzaban Dice = 0.32-0.33
  sin priors. La nueva fase 4 capitaliza ese resultado con una implementacion
  compacta y idiomatica de MONAI.
- El balance pos/neg, antes gestionado con un manifest custom de slices y mas
  de 600 lineas de codigo, ahora se delega a `RandCropByPosNegLabeld`.

## Componentes MONAI activos
- Transforms diccionario para preprocesado e I/O: `LoadImaged`,
  `EnsureChannelFirstd`, `Orientationd`, `Spacingd`, `ScaleIntensityRanged`,
  `CropForegroundd`.
- Augmentacion: `RandFlipd`, `RandRotate90d`, `RandShiftIntensityd`,
  `RandScaleIntensityd`, mas `RandGaussianNoised`, `RandGaussianSmoothd` y
  `RandAdjustContrastd` en regimen Fuerte.
- Muestreo balanceado: `RandCropByPosNegLabeld(pos=1, neg=1, num_samples=4)`.
- Caching: `CacheDataset` (cache_rate configurable).
- Modelo: `monai.networks.nets.SegResNet` (~4.7M parametros, init_filters=16).
- Loss: `DiceFocalLoss(sigmoid=True)` con pesos via env vars.
- Inferencia: `SlidingWindowInferer(roi_size=PATCH_SIZE_3D, sw_batch_size=4,
  overlap=0.5, mode="gaussian")`.
- Metrica de validacion: `DiceMetric(include_background=False)`.

## Fases retiradas
Se mantienen los scripts `fase2`, `fase3` y `fase5`, pero solo como aviso de
retirada:
- `fase2_reconstruccion.py`: el dataset ya esta reconstruido.
- `fase3_registro.py`: no mejora el objetivo principal de segmentacion.
- `fase5_radiomica.py`: se pospone hasta tener mascaras fiables y una tarea
  clinica definida.

## Ejecucion
Pipeline principal:

```bash
./run_all.sh
```

Con modo rapido para validar arranque:

```bash
MONAI_FAST_DEV=1 ./run_all.sh
```

Demo:

```bash
.venv/bin/python scripts/fase7_demo.py
```

## Variables utiles
- `MONAI_FAST_DEV=1`: smoke run corto para validar que todo arranca.
- `MONAI_NUM_EPOCHS_3D=...`: epocas de fase 4 (default 200).
- `MONAI_NUM_EPOCHS_ABLACION_3D=...`: epocas de fase 6 (default 80).
- `MONAI_BATCH_SIZE_3D=...`: batch size de entrenamiento (default 2 con AMP).
- `MONAI_SAMPLES_PER_VOLUME=...`: parches por volumen en
  `RandCropByPosNegLabeld` (default 4).
- `MONAI_POS_NEG_RATIO=...`: ratio pos/neg en el muestreo (default 1.0).
- `MONAI_INFER_OVERLAP=...`: overlap del sliding window (default 0.5).
- `MONAI_SEGRESNET_INIT_FILTERS=...`: filtros base de SegResNet (default 16).
- `MONAI_SEGMENTATION_AUGMENT_LEVEL=1|2`: regimen de augmentacion (Base por
  defecto; ver CHANGELOG sobre el porque).
- `MONAI_SEGMENTATION_THRESHOLD_MIN/MAX/STEP`: rango de busqueda de threshold.
- `MONAI_SEGMENTATION_ENABLE_TTA_FINAL=0/1`: activa o desactiva TTA final.
- `MONAI_LOSS_LAMBDA_DICE/FOCAL`, `MONAI_LOSS_FOCAL_GAMMA`: pesos y gamma de
  `DiceFocalLoss` (defaults `0.5/0.5`, `gamma=2.0`).
- `MONAI_CACHE_RATE=0..1`: fraccion de casos cacheados en RAM por
  `CacheDataset` (default 1.0; baja a 0.0 con FAST_DEV).

## Artefactos principales
- `resultados/metricas/f4_resumen_3d.json`
- `resultados/metricas/f4_metricas_val_3d.csv` (mejor threshold)
- `resultados/metricas/f4_metricas_val_3d_std.csv` (threshold 0.5)
- `resultados/metricas/f4_metricas_val_3d_tta.csv` (TTA si esta activo)
- `resultados/metricas/f4_threshold_search_3d.csv` (y `_tta`)
- `resultados/metricas/f4_comparativa.csv`
- `resultados/metricas/f6_ablacion.csv`
- `resultados/figuras/f4_curvas_3d.png`,
  `f4_threshold_search_3d.png`,
  `f4_comparativa_3d.png`,
  `f4_predicciones_3d.png`

Los artefactos `f4_*_2d_*` y `f4_cascade_*` se preservan como evidencia
historica de los pipelines 2D y de la cascada y no se regeneran desde el
flujo activo.

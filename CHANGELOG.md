# Changelog

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

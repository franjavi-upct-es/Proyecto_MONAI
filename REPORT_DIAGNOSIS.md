# REPORT_DIAGNOSIS — Lung Tumor Segmentation (MSD Task06_Lung) on MONAI

Fecha: 2026-04-28
Bloque: B0 (solo lectura/diagnóstico)
Repo: `/home/pyros05/Escritorio/Proyecto_MONAI`

---

## §1. Mapa real del repositorio

### 1.1 Árbol resumido

```
Proyecto_MONAI/
├── CHANGELOG.md, CHANGELOG.pdf, README.md
├── Entregable.ipynb              # 1.4 MB, legado, no runtime
├── pyproject.toml                # plano, sin extras [dev,kaggle,radiomics]
├── requirements.txt
├── run_all.sh                    # cd scripts && python fase{1,4,6}_*.py
├── uv.lock
├── data/
│   ├── Task06_Lung/{imagesTr,imagesTs,labelsTr,dataset.json}
│   └── cache_2d/                 # cache 2D (legado)
├── modelos/
│   └── mejor_SegResNet3D_patch.pth   # único checkpoint activo
├── resultados/
│   ├── figuras/                  # vacío en runtime actual
│   └── metricas/{f1_resumen_casos.csv, f6_ablacion.csv}
└── scripts/
    ├── fase{1..7}_*.py           # entrypoints finos
    └── monai_pipeline/
        ├── core/{config.py, utils.py}
        ├── pipelines/{patch3d.py, baseline2d.py, cascade.py}
        └── apps/{exploration.py, segmentation.py, ablation.py, demo.py, retired.py}
```

**Discrepancias estructurales con la plantilla del refactor:**
no existen `src/lungseg/`, `notebooks/`, `configs/`, `tests/`, `data/raw/`,
`outputs/`. Los datos están directamente en `data/Task06_Lung/`, no en
`data/raw/Task06_Lung/`. El directorio de modelos se llama `modelos/`, no
`outputs/`.

### 1.2 Inventario de entrypoints

| Script | Estado | Dispatch | Responsabilidad |
|---|---|---|---|
| `scripts/fase1_exploracion.py` | ACTIVO | `apps.exploration` | Análisis del dataset, estadísticas por caso, resumen de desbalance |
| `scripts/fase2_reconstruccion.py` | RETIRADO | `apps.retired.run_reconstruction_phase` | (suspendido) reconstrucción CT por sinogramas |
| `scripts/fase3_registro.py` | RETIRADO | `apps.retired.run_registration_phase` | (suspendido) registro interpaciente |
| `scripts/fase4_segmentacion.py` | ACTIVO | `apps.segmentation.main` | Entrenamiento SegResNet 3D + búsqueda de threshold + TTA + figuras |
| `scripts/fase5_radiomica.py` | RETIRADO | `apps.retired.run_radiomics_phase` | (suspendido) extracción de features radiómicas |
| `scripts/fase6_escasez.py` | ACTIVO | `apps.ablation.main` | Ablación 3 fracciones × 2 regímenes (1 seed, 80 épocas/celda) |
| `scripts/fase7_demo.py` | ACTIVO | `apps.demo.launch` | Demo Gradio que carga `mejor_SegResNet3D_patch.pth` y segmenta NIfTI |

### 1.3 Estado del notebook `Entregable.ipynb`

- 1.4 MB; legado (no migrado al flujo 3D activo, según README línea 180).
- Duplica lógica que ahora vive en `apps/segmentation.py`, `apps/ablation.py` y
  `pipelines/patch3d.py` (env detection, transforms MONAI, `KFold` de sklearn,
  threshold search, ablación). No se ejecuta desde `run_all.sh`.
- Se conserva como deliverable documental.

---

## §2. Bugs y antipatrones por fase, con cita exacta

### (a) `ScaleIntensityRange` con `a_min/a_max` fuera del rango torácico

- **Hallazgo único** —
  `scripts/monai_pipeline/core/config.py:126` →
  `CT_CLIP_RANGE = (-1024.0, 400.0)`. Aplicado en:
  - `scripts/monai_pipeline/pipelines/patch3d.py:174-181`
    (`ScaleIntensityRanged(a_min=CT_CLIP_RANGE[0], a_max=CT_CLIP_RANGE[1], …)`).
  - `scripts/monai_pipeline/pipelines/baseline2d.py:163-170` (mismo patrón).
- El rango cumple lo "permisivo" `[-1024..-500]/[200..400]` pedido por el
  prompt, pero **viola la regla dura de `CLAUDE.md`** que fija `a_max = 325`.
  Subir hasta 400 HU incluye señal de hueso/calcificación; defendible para
  reducir saturación, pero no documentado como justificación técnica en el
  código.
- **Severidad: MEDIA** — no es un fallo de pipeline, es una desviación
  contractual con `CLAUDE.md`.

### (b) `RandCropByPosNegLabeld` no balanceado (pos=neg=1)

- `scripts/monai_pipeline/pipelines/patch3d.py:199-209` →
  ```python
  RandCropByPosNegLabeld(
      keys=KEYS_BOTH,
      label_key="label",
      spatial_size=PATCH_SIZE_3D,            # (96,96,96)
      pos=POS_NEG_RATIO_3D,                  # default 1.0
      neg=1.0,
      num_samples=SAMPLES_PER_VOLUME_3D,     # default 4
      …
  )
  ```
  con defaults definidos en `core/config.py:186-190`
  (`SAMPLES_PER_VOLUME_3D=4`, `POS_NEG_RATIO_3D=1.0`).
- Ratio 1:1 sobre tumores que ocupan ~10⁻⁴–10⁻² del volumen ⇒ los parches
  "positivos" siguen estando dominados por fondo en el centro. La plantilla
  pide `pos=2, neg=1, num_samples=4` para reforzar foreground.
- **Severidad: ALTA**.

### (c) `RandFlipd` con `spatial_axis=0` (LR)

- `scripts/monai_pipeline/pipelines/patch3d.py:211` →
  `RandFlipd(keys=KEYS_BOTH, prob=0.5, spatial_axis=0)`.
- Tras `Orientationd(axcodes="RAS")` (línea 172) +
  `EnsureChannelFirstd` (línea 171), el primer eje espacial es R/L. Flipear
  L↔R en CT torácico es **PROHIBIDO por `CLAUDE.md`** (rompe simetría
  anatómica clínica: el lóbulo medio sólo existe en el pulmón derecho).
- Las líneas siguientes 212 (`spatial_axis=1`, AP) y 213 (`spatial_axis=2`, IS)
  son aceptables si se quiere conservar augmentación geométrica; la plantilla
  del refactor admite sólo `spatial_axis=2`.
- **Severidad: CRÍTICA**.

### (d) Loop por épocas en lugar de iteraciones

- `scripts/monai_pipeline/pipelines/patch3d.py:529` →
  `for epoca in range(1, num_epocas + 1):`.
  - `num_epocas` proviene de
    `core/config.py:207` (`NUM_EPOCAS_3D = 200` en producción, 2 en
    `MONAI_FAST_DEV`).
  - Validación en `patch3d.py:553` cada `VAL_INTERVAL_3D = 2` épocas
    (`config.py:216`), no cada N iteraciones.
  - Early stopping por épocas (línea 589, `patience = 25`).
- Idem en legado:
  - `scripts/monai_pipeline/pipelines/baseline2d.py:1116`
    (`for epoca in range(1, num_epocas + 1):`).
  - `scripts/monai_pipeline/pipelines/cascade.py:826`
    (`for epoca in range(1, num_epocas + 1):`).
- Incompatible con la receta nnU-Net-like del refactor (`max_iterations=50_000`,
  `val_every=500` iteraciones, paciencia en iteraciones).
- **Severidad: MEDIA** (decisión arquitectónica, no es un bug funcional).

### (e) Uso de máscaras predichas en PyRadiomics (Phase 5)

- **NO APLICA EN EL ESTADO ACTIVO.** PyRadiomics está ausente:
  - `scripts/monai_pipeline/apps/retired.py:45-54` — `run_radiomics_phase`
    sólo escribe un JSON `{"estado": "retirada", …}`.
  - Grep sobre todo el repo: 0 imports de `radiomics` /
    `RadiomicsFeatureExtractor`.
  - `pyproject.toml:19,52` — pyradiomics declarado como dependencia desde
    git, pero **nadie lo usa**.
- Hueco a cubrir en B5 (decidir GT vs predicted, definir `--e2e` flag).
- **Severidad: N/A para diagnóstico; ALTA como gap de B5**.

### (f) Splits aleatorios sin GroupKFold por paciente

- `scripts/monai_pipeline/pipelines/patch3d.py:133` →
  `kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)`.
- `scripts/monai_pipeline/pipelines/baseline2d.py:438` →
  `kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)`.
- `scripts/monai_pipeline/pipelines/patch3d.py:60` y
  `scripts/monai_pipeline/pipelines/baseline2d.py` importan únicamente
  `from sklearn.model_selection import KFold`. **Ningún `GroupKFold` en todo
  el repo.**
- Mitigado de facto en Task06 porque cada `case_id` tiene un único volumen,
  pero romperá silenciosamente al integrar LIDC-IDRI (varios scans por
  paciente) en B5. La plantilla pide `GroupKFold(groups=patient_id)` con
  estratificación aproximada por volumen tumoral.
- **Severidad: ALTA** (decisión arquitectónica que debe corregirse en B2 antes
  de tocar Phase 5).

### (g) `DynUNet` sin `deep_supervision` o sin manejar el stacked output

- **NO APLICA AL ESTADO ACTIVO.** El modelo activo es `SegResNet`
  (`patch3d.py:38, 242-254`) instanciado con
  `init_filters=16, blocks_down=(1,2,2,4), blocks_up=(1,1,1), norm="instance"`.
- Las únicas menciones de DynUNet son métricas hardcoded de runs históricos
  en `core/config.py:313-371` (`LEGACY_3D_METRICS`, `LEGACY_3D_HISTORIES`,
  Dice 0.331). No se instancia ni entrena.
- Relevante para B3 cuando se introduzca DynUNet como modelo principal de la
  receta nnU-Net-like.

### (h) `DiceCELoss(include_background=True)` en binario con FG diminuto

- `scripts/monai_pipeline/pipelines/cascade.py:957-959` →
  ```python
  criterio = DiceCELoss(
      sigmoid=True, lambda_dice=0.7, lambda_ce=0.3, squared_pred=True
  ).to(DEVICE)
  ```
  Sin `include_background` explícito ⇒ MONAI usa el default `True`. Con
  `sigmoid=True` y output de 1 canal el efecto del flag no es el clásico
  (no hay canal 0 = background separado), pero la API es ambigua y la elección
  no está justificada en código.
- En el activo `patch3d.py:490-495` se usa
  `DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=0.5, lambda_focal=0.5)`
  — esta pérdida no expone `include_background`. La nota del propio config
  (`config.py:234-235`) reconoce que el modelo oversegmenta
  (precision ≈ 0.0017 a t=0.5).
- La plantilla del refactor pide
  `DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)`,
  más adecuado para 2-canal con foreground diminuto.
- **Severidad: MEDIA** (cascade.py es legado; el activo usa otra loss).

### (i) Seeds no fijadas / `set_determinism` ausente

- `set_determinism` SÍ se llama:
  - `scripts/monai_pipeline/core/config.py:81-82` →
    `SEMILLA = 42; set_determinism(seed=SEMILLA)` (al import).
- Pero el determinismo **se rompe localmente**:
  - `scripts/monai_pipeline/pipelines/baseline2d.py:653, 657` —
    `np.random.randint(...)` sin RNG local en
    `_sample_patch_origin_from_mask`.
  - `scripts/monai_pipeline/pipelines/baseline2d.py:714, 717, 720, 727, 729,
    733-738, 743-746, 751-758` — `np.random.rand/randint/uniform/normal/randn`
    sin RNG local en `SliceDataset._augment` (incluyendo deformación elástica
    con `np.random.randn(H, W)`).
  - `CacheDataset` en `patch3d.py:430-442` se construye con
    `num_workers=N_WORKERS_PREPROCESADO` (>0) sin `worker_init_fn`. El
    estado de `set_determinism` no se propaga a los workers de
    multiprocessing.
- **Severidad: MEDIA** (parcialmente mitigado pero no determinista en
  augmentación 2D ni en workers).

### (j) Hardcoded paths fuera de configs

- `scripts/monai_pipeline/core/config.py:54-65` — todas las rutas
  (`DIR_DATOS`, `DIR_MODELOS`, `DIR_RESULTADOS`, `DIR_FIGURAS`,
  `DIR_METRICAS`, `DIR_CACHE_2D`) son `Path(__file__).resolve().parent…/`
  derivadas; **no hay literales absolutos hardcoded**.
- Sin embargo, la configuración global es **env-var-driven** vía
  `_env_bool/_env_int/_env_float` (`config.py:34-48`) parseando
  >30 variables `MONAI_*` (líneas 147-238, 244-281, 293-298). No hay Hydra ni
  Pydantic ni dataclasses tipadas. La "Hard rule" del refactor exige
  Hydra-only.
- **Severidad: MEDIA** — pasa el filtro literal del prompt pero choca con la
  arquitectura objetivo de `CLAUDE.md` y B1.

### Hallazgos extra relevantes

| # | Hallazgo | Cita | Severidad |
|---|---|---|---|
| K1 | No existe `tests/`. Cero cobertura | (ausencia) | ALTA |
| K2 | `pyproject.toml:5-27` plano, sin extras `[dev,kaggle,radiomics]` | `pyproject.toml` | MEDIA |
| K3 | `run_all.sh:13-23` hace `cd scripts && python fase*.py` (acoplado a CWD, no soporta `pip install -e .`) | `run_all.sh:13` | MEDIA |
| K4 | Sin W&B / MLflow; sólo CSV/JSON locales (`segmentation.py:288-294`) | `apps/segmentation.py:288` | MEDIA |
| K5 | Augmentación duplicada entre `baseline2d.py:702-776` y `cascade.py:281-313` (~75 líneas idénticas) | DRY violation | BAJA |
| K6 | Validación correcta: `DiceMetric(include_background=False)` en `patch3d.py:499` ✓ | (positivo) | — |
| K7 | Sin checkpoints intermedios; sólo `mejor_*.pth` (`patch3d.py:578-579`); sin recovery | `patch3d.py:579` | BAJA |
| K8 | Ablación con 1 sola seed, 80 épocas/celda y `threshold=0.5` hardcoded en `ablation.py:81-82` (no busca threshold óptimo por celda) | `apps/ablation.py:60-95` | ALTA |
| K9 | `f6_ablacion.csv` filas 0.25 → Dice = **0.0** ambos regímenes; el sistema **colapsa en escasez de datos** | `resultados/metricas/f6_ablacion.csv:2-3` | ALTA |
| K10 | `seleccionar_subconjunto` en `patch3d.py:144-156` muestrea por **caso** ya (no por imagen suelta) ✓ | (positivo) | — |
| K11 | `f4_curvas_3d.png`, `f4_threshold_search_3d.png`, `f4_predicciones_3d.png` declarados como output pero `resultados/figuras/` está vacío ⇒ no hay run completo reciente del activo | listado dir | INFO |
| K12 | `LEGACY_3D_ABLATION` en `config.py:373-428` reporta 0.0 en TODAS las celdas — evidencia de que la fase 6 ya colapsó antes; el código actual repite el patrón sin cambios estructurales | `core/config.py:373` | ALTA |

---

## §3. Tabla DSC objetivo vs DSC actual

| Fase | Pipeline | DSC objetivo (refactor) | DSC actual | Fuente |
|---|---|---|---|---|
| F4 — 3D activo | DynUNet + nnU-Net recipe (objetivo) **vs.** SegResNet 3D + DiceFocalLoss (real) | **≥ 0.45** fold 0, AMP, RTX 5060 8 GB | **0.1272 (Base) / 0.138 (Fuerte)** con full data | `resultados/metricas/f6_ablacion.csv` líneas 6-7 |
| F4 — 3D legado (DynUNet) | UNet-style 3D + sin priors | (referencia) | 0.331 | `core/config.py:313-325` |
| F4 — 2D + priors (retirada) | UNet 2D AxialContext + priors | — | 0.135 (TTA, t=0.70) | README:25-31, CHANGELOG 2026-04-25 |
| F4 — Cascada coarse-to-fine (retirada) | 2D coarse → 2D fine | — | 0.1092 mean | CHANGELOG 2026-04-23 |
| F5 — Radiómica + clasificación | Pipeline cerrado: RobustScaler→MI(SelectKBest k=20)→{RF,XGB,LASSO,MLP}; baseline VoxelVolume; AUC + bAcc + Brier + ECE; GroupKFold por paciente | **N/A** (PyRadiomics ausente; fase retirada) | `apps/retired.py:45-54` |
| F6 — Ablación | 3 fracs × 2 augs × 3 seeds, 50 000 iters fijos por celda, **DSC > 0 en todas las celdas** | 1 seed, 80 épocas/celda; **Dice = 0.0 a fracción 0.25 (ambos regímenes)** | `f6_ablacion.csv:2-3` |

**Conclusiones cuantitativas:**
- El pipeline activo (F4) está a **~0.34× del objetivo** (0.138 / 0.45).
- F6 **falla el criterio de aceptación** del refactor: a fracción 0.25 ambos
  regímenes colapsan a Dice = 0.0 (no es ruido aleatorio: el mismo patrón se
  observó en `LEGACY_3D_ABLATION` con el modelo 3D legado, lo que sugiere un
  problema estructural — probablemente sampler, loss y/o épocas insuficientes
  para que el modelo aprenda con 12 casos).

---

## §4. Resumen de severidades y trazabilidad al refactor

| ID | Hallazgo | Severidad | Bloque que lo aborda |
|---|---|---|---|
| (c) | `RandFlipd spatial_axis=0` (LR flip prohibido) | **CRÍTICA** | B2 (transforms) |
| (b) | `RandCropByPosNegLabeld pos=neg=1` | **ALTA** | B2 (transforms) |
| (f) | KFold sin GroupKFold por paciente | **ALTA** | B2 (splits) |
| K1 | Sin `tests/` | **ALTA** | B1 (scaffold) |
| K8 | Ablación con 1 seed + threshold hardcoded | **ALTA** | B6 |
| K9/K12 | Colapso a Dice = 0 a fracción 0.25 | **ALTA** | B6 (estructural; consecuencia de b/c/f/K8 + épocas insuficientes) |
| (e) | Radiómica ausente | **ALTA** (gap) | B5 |
| (a) | `CT_CLIP_RANGE = (-1024, 400)` viola CLAUDE.md (`a_max=325`) | MEDIA | B2 |
| (d) | Loop por épocas, no iteraciones | MEDIA | B4 (trainer) |
| (g) | DynUNet ausente | MEDIA (gap) | B3 |
| (h) | `DiceCELoss(include_background=True)` en cascade.py | MEDIA (legado) | B3 (loss) |
| (i) | Determinismo roto en `_augment` y workers | MEDIA | B2/B4 |
| (j) | Config env-var en lugar de Hydra | MEDIA | B1 (Hydra) |
| K2 | `pyproject` sin extras | MEDIA | B1 |
| K3 | `run_all.sh` acoplado a CWD | MEDIA | B1/B4 (CLI) |
| K4 | Sin W&B/MLflow | MEDIA | B4 |
| K5 | Aug duplicada baseline2d/cascade | BAJA | (legado, no urgente) |
| K7 | Sin checkpoints intermedios | BAJA | B4 |
| K6/K10 | `DiceMetric(include_background=False)` y subset por caso ✓ | (positivos) | — |

---

## §5. Divergencias plantilla ↔ repo (preguntas a resolver antes de B1)

Estas decisiones se necesitan ANTES de empezar B1 — la plantilla del refactor
asume un repo distinto al actual y hay que fijar criterios:

1. **Coexistencia o reemplazo.** ¿La nueva estructura `src/lungseg/` reemplaza
   por completo a `scripts/monai_pipeline/`, o conviven durante el refactor con
   `scripts/monai_pipeline/` marcado como `legacy/` y los entrypoints
   `scripts/fase*.py` redireccionados a la nueva CLI? Recomendación por
   defecto: **reemplazo gradual** — mover `scripts/monai_pipeline/` a
   `legacy/monai_pipeline/` y dejar los `scripts/fase*.py` como shims a
   `python -m lungseg.cli ...` durante una transición.

2. **Notebooks `f1..f6`.** No existen (sólo `Entregable.ipynb`). Opciones:
   (i) mover `Entregable.ipynb` a `notebooks/legacy/` y dejar B7 como único
   notebook moderno (Kaggle); (ii) generar f1..f6 finos nuevos. La plantilla
   sugiere (i) explícitamente — confirmar.

3. **Datos.** Path real: `data/Task06_Lung/`, no `data/raw/Task06_Lung/`.
   Opciones: (a) **renombrar** `data/Task06_Lung/` → `data/raw/Task06_Lung/`
   (afecta a `core/config.py:107`, `dataset.json` y a un checkpoint con
   manifests cacheados); (b) mantener `data/Task06_Lung/` y ajustar el config
   Hydra. Recomendación: (a) por coherencia con la regla "Never edit files in
   `data/raw/`" de `CLAUDE.md`.

4. **Modelos.** Directorio actual `modelos/` vs. plantilla `outputs/`.
   Recomendación: mantener `modelos/` para checkpoints publicados y usar
   `outputs/` (gitignored) para artefactos de cada run de Hydra. **Confirmar.**

5. **Ventana HU.** `CLAUDE.md` dice `a_max=325`; el código actual usa
   `CT_CLIP_RANGE=(-1024, 400)`. La regla añade *"do NOT change without
   justification"*. Subir a 400 incluye hueso/calcificaciones (defendible para
   tumor sólido + invasión costal). **Decisión necesaria:** mantener 325
   estricto (rompe modelo entrenado) o documentar 400 como excepción
   justificada.

6. **DynUNet vs SegResNet.** El criterio F4 del refactor pide
   "DynUNet + nnU-Net-like recipe". El checkpoint en producción es
   `mejor_SegResNet3D_patch.pth` (~4.7 M params, init_filters=16). Opciones:
   (a) **DynUNet pasa a default** y `SegResNet` queda como wrapper opcional —
   significa que el modelo entrenado en `modelos/` se reentrena desde cero;
   (b) **SegResNet sigue siendo default** y DynUNet se añade como factory
   alternativa. La plantilla orienta a (a) por la promesa de DSC ≥ 0.45.
   **Decisión necesaria.**

7. **Ablación: épocas vs iteraciones.** B6 fija
   `max_iterations=50_000` por celda × 18 celdas (3 fracs × 2 augs × 3 seeds).
   Estimación grosera: 18 × 50 000 / TR_BS ≈ varios días en una RTX 5060 8 GB.
   ¿Mantener 50 000 estricto o ajustar a un valor más realista (p. ej. 20 000)
   manteniendo la regla "fijo por celda" que es el aspecto científicamente
   crítico?

---

## Apéndice A — Cobertura de los 10 puntos solicitados

| Punto | Estado | Sección |
|---|---|---|
| (a) ScaleIntensityRange fuera de rango torácico | **Hallazgo** (a_max=400 vs CLAUDE.md=325) | §2 (a) |
| (b) RandCropByPosNegLabeld pos=neg=1 | **Hallazgo** | §2 (b) |
| (c) RandFlipd spatial_axis=0 (LR) | **Hallazgo CRÍTICO** | §2 (c) |
| (d) Loop de épocas en lugar de iteraciones | **Hallazgo** | §2 (d) |
| (e) Máscaras predichas en PyRadiomics | **No aplica** (PyRadiomics ausente) | §2 (e) |
| (f) Splits sin GroupKFold por paciente | **Hallazgo** | §2 (f) |
| (g) DynUNet sin deep_supervision | **No aplica** (DynUNet ausente) | §2 (g) |
| (h) DiceCELoss(include_background=True) | **Hallazgo** (cascade.py legado) | §2 (h) |
| (i) Seeds no fijadas / `set_determinism` ausente | **Hallazgo parcial** (set_determinism presente, pero `np.random` global en `_augment`) | §2 (i) |
| (j) Hardcoded paths fuera de configs | **No aplica literal** (rutas derivadas), pero env-var-driven en lugar de Hydra | §2 (j) |

"""
Configuracion central del pipeline.

Estrategia activa (refactor 2026-04-25):
  Pipeline 3D patch-based con SegResNet de MONAI. Reemplaza al pipeline
  2D/2.5D con priors anatomicos, que tope en Dice=0.135 con TTA y no resolvia
  el problema de localizacion (caso lung_078 predijo el tamano correcto en
  sitio incorrecto). El modelo 3D aprovecha el contexto volumetrico completo
  y resolvia 0.32-0.33 Dice en los experimentos legados, sin priors.

Componentes MONAI clave del flujo activo:
  - LoadImaged, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd
  - RandCropByPosNegLabeld (balance pos/neg automatico, sustituye al manifest
    custom de slices)
  - SegResNet (~4.7M params, mejor 3D legado)
  - DiceFocalLoss
  - SlidingWindowInferer con overlap=0.5 mode=gaussian
  - DiceMetric para validacion

Constantes 2D (PATCH_SIZE_2D, CONTEXT_SLICES, INPUT_CHANNELS_2D, RUTA_MODELO_2D,
MODELO_2D_NOMBRE, CACHE_VERSION_2D...) se conservan para que el pipeline legado
en `pipelines/baseline2d.py` siga importable como referencia historica.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from monai.utils.misc import set_determinism


def _env_bool(nombre: str, por_defecto: bool = False) -> bool:
    valor = os.getenv(nombre)
    if valor is None:
        return por_defecto
    return valor.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(nombre: str, por_defecto: int) -> int:
    valor = os.getenv(nombre)
    return por_defecto if valor is None else int(valor)


def _env_float(nombre: str, por_defecto: float) -> float:
    valor = os.getenv(nombre)
    return por_defecto if valor is None else float(valor)


# =============================================================================
# Rutas
# =============================================================================
RAIZ = Path(__file__).resolve().parent.parent.parent.parent

DIR_DATOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "modelos"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"
DIR_METRICAS = DIR_RESULTADOS / "metricas"
DIR_CACHE_2D = DIR_DATOS / "cache_2d"

# Incrementar version invalida el cache anterior y fuerza regeneracion.
# v4_ctx2: context_slices ampliado a 2 -> INPUT_CHANNELS_2D = 8
CACHE_VERSION_2D = "v4_ctx2"

for directorio in [
    DIR_DATOS,
    DIR_MODELOS,
    DIR_RESULTADOS,
    DIR_FIGURAS,
    DIR_METRICAS,
    DIR_CACHE_2D,
]:
    directorio.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Reproducibilidad
# =============================================================================
SEMILLA = 42
set_determinism(seed=SEMILLA)


# =============================================================================
# CPU / GPU
# =============================================================================
N_CPU = os.cpu_count() or 1
N_WORKERS_PREPROCESADO = max(1, min(4, N_CPU - 1))
N_WORKERS_DATALOADER = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    torch.set_num_threads(max(1, N_CPU - 1))


# =============================================================================
# Dataset
# =============================================================================
TASK = "Task06_Lung"
TASK_DIR = DIR_DATOS / TASK
RUTA_DATASET_JSON = TASK_DIR / "dataset.json"

# Tamano del parche 2D entrenado (no un redimensionado global).
PATCH_SIZE_2D = (256, 256)
FINE_ROI_SIZE_2D = (320, 320)
# Espaciado fisico objetivo tras Spacingd (mm). XY isotropico, Z estable
# para que el contexto 2.5D tenga significado fisico consistente.
TARGET_SPACING_3D = (1.0, 1.0, 2.0)

# MEJORA: CONTEXT_SLICES 1 -> 2.
# Con 2 slices de contexto el modelo recibe (z-2, z-1, z, z+1, z+2):
# 5 cortes de intensidad CT + 3 canales de priors = 8 canales totales.
# Esto mejora la discriminacion de tumor vs tejido denso que aparece
# y desaparece en pocos cortes, a costa de requerir regenerar el cache.
CONTEXT_SLICES = 2
PRIOR_CHANNELS_2D = 3
INPUT_CHANNELS_2D = 2 * CONTEXT_SLICES + 1 + PRIOR_CHANNELS_2D  # = 8

CT_CLIP_RANGE = (-1024.0, 400.0)
N_FOLDS_KFOLD = 5
KFOLD_DEFAULT_INDEX = 0
POSITIVE_SLICE_MIN_PIXELS = 10
NEGATIVE_TO_POSITIVE_RATIO = 1.5
HARD_NEGATIVE_MARGIN = 3
# 256 voxels a spacing (1,1,2) ~ 512 mm^3 ~ 1 cm de diametro. Mas selectivo
# que el 64 anterior (~0.13 cm^3) y aun bien por debajo del tumor mas pequeno
# del val (1283 voxels ~ 2.6 cm^3). Filtra ruido sin tocar lesiones reales.
MIN_COMPONENT_SIZE = 256
HU_LUNG_CANDIDATE_MAX = -320.0
HU_HEALTHY_LUNG_RANGE = (-950.0, -500.0)
HU_SOFT_TISSUE_RANGE = (-500.0, 180.0)
ALARM_COMPONENT_MIN_SIZE = 20
ALARM_SLICE_MIN_PIXELS = 64
ALARM_NEGATIVE_EXCLUSION_PIXELS = 96


# =============================================================================
# Entrenamiento
# =============================================================================
FAST_DEV_RUN = _env_bool("MONAI_FAST_DEV", False)

NUM_EPOCAS = _env_int("MONAI_NUM_EPOCHS", 30 if not FAST_DEV_RUN else 2)
NUM_EPOCAS_ABLACION = _env_int(
    "MONAI_NUM_EPOCHS_ABLACION",
    15 if not FAST_DEV_RUN else 1,
)
EARLY_STOPPING_PATIENCE = _env_int(
    "MONAI_EARLY_STOPPING_PATIENCE",
    8 if not FAST_DEV_RUN else 2,
)
VAL_INTERVAL = 1

LR = _env_float("MONAI_LR", 3e-4)
WEIGHT_DECAY = _env_float("MONAI_WEIGHT_DECAY", 1e-5)
TRAIN_BATCH_SIZE = _env_int("MONAI_BATCH_SIZE", 16 if AMP_ENABLED else 4)
INFER_BATCH_SIZE = _env_int("MONAI_INFER_BATCH_SIZE", 64 if AMP_ENABLED else 8)
TRAIN_BATCH_SIZE_FINE = _env_int("MONAI_BATCH_SIZE_FINE", 8 if AMP_ENABLED else 2)
INFER_BATCH_SIZE_FINE = _env_int("MONAI_INFER_BATCH_SIZE_FINE", 16 if AMP_ENABLED else 4)

# Epocas de warmup lineal antes del cosine annealing.
LR_WARMUP_EPOCAS = _env_int("MONAI_LR_WARMUP_EPOCAS", 3 if not FAST_DEV_RUN else 1)

MODELO_2D_NOMBRE = "UNet2D-AxialContext+Priors-v4"
RUTA_MODELO_2D = DIR_MODELOS / "mejor_UNet2D_priors_v4.pth"

# =============================================================================
# Pipeline 3D activo (patch-based con SegResNet)
# =============================================================================
MODELO_3D_NOMBRE = "SegResNet3D-Patch"
RUTA_MODELO_3D = DIR_MODELOS / "mejor_SegResNet3D_patch.pth"

# Tamano del parche 3D para entrenamiento e inferencia con sliding window.
# 96^3 a spacing (1,1,2) cubre ~9.6 x 9.6 x 19.2 cm, suficiente para encajar
# tumor + contexto pulmonar para la mayoria de casos del Task06_Lung.
PATCH_SIZE_3D = (96, 96, 96)

# RandCropByPosNegLabeld: 1 patch positivo y 1 negativo por muestreo, 4
# muestras por volumen en produccion (2 en FAST_DEV para smoke en GPUs chicas).
SAMPLES_PER_VOLUME_3D = _env_int(
    "MONAI_SAMPLES_PER_VOLUME",
    4 if not FAST_DEV_RUN else 2,
)
POS_NEG_RATIO_3D = _env_float("MONAI_POS_NEG_RATIO", 1.0)

TRAIN_BATCH_SIZE_3D = _env_int(
    "MONAI_BATCH_SIZE_3D",
    (2 if AMP_ENABLED else 1) if not FAST_DEV_RUN else 1,
)
INFER_SW_BATCH_3D = _env_int(
    "MONAI_INFER_SW_BATCH",
    (4 if AMP_ENABLED else 1) if not FAST_DEV_RUN else 1,
)
INFER_OVERLAP_3D = _env_float("MONAI_INFER_OVERLAP", 0.5)

SEGRESNET_INIT_FILTERS = _env_int("MONAI_SEGRESNET_INIT_FILTERS", 16)
SEGRESNET_DROPOUT = _env_float("MONAI_SEGRESNET_DROPOUT", 0.2)

LR_3D = _env_float("MONAI_LR_3D", 1e-4)
WEIGHT_DECAY_3D = _env_float("MONAI_WEIGHT_DECAY_3D", 1e-5)
NUM_EPOCAS_3D = _env_int("MONAI_NUM_EPOCHS_3D", 200 if not FAST_DEV_RUN else 2)
EARLY_STOPPING_PATIENCE_3D = _env_int(
    "MONAI_EARLY_STOPPING_PATIENCE_3D",
    25 if not FAST_DEV_RUN else 2,
)
NUM_EPOCAS_ABLACION_3D = _env_int(
    "MONAI_NUM_EPOCHS_ABLACION_3D",
    80 if not FAST_DEV_RUN else 1,
)
VAL_INTERVAL_3D = _env_int("MONAI_VAL_INTERVAL_3D", 2 if not FAST_DEV_RUN else 1)
CACHE_RATE_3D = _env_float("MONAI_CACHE_RATE", 1.0 if not FAST_DEV_RUN else 0.0)

# Fase 4 activa: pipeline MONAI 2D/2.5D compacto.
# Default Base (1) tras la ablacion del 2026-04-25: Fuerte degrada el Dice de
# 0.2310 a 0.1189 con todos los datos. Se puede forzar Fuerte exportando
# MONAI_SEGMENTATION_AUGMENT_LEVEL=2 si se quiere reproducir el experimento.
SEGMENTATION_AUGMENT_LEVEL = _env_int("MONAI_SEGMENTATION_AUGMENT_LEVEL", 1)
SEGMENTATION_THRESHOLD_MIN = _env_float("MONAI_SEGMENTATION_THRESHOLD_MIN", 0.20)
SEGMENTATION_THRESHOLD_MAX = _env_float("MONAI_SEGMENTATION_THRESHOLD_MAX", 0.80)
SEGMENTATION_THRESHOLD_STEP = _env_float("MONAI_SEGMENTATION_THRESHOLD_STEP", 0.05)
SEGMENTATION_ENABLE_TTA_FINAL = _env_bool(
    "MONAI_SEGMENTATION_ENABLE_TTA_FINAL",
    not FAST_DEV_RUN,
)

# Pesos y gamma de DiceFocalLoss expuestos para iterar sin tocar codigo.
# Defaults preservan el comportamiento previo (lambda 0.5/0.5, gamma=2.0).
# El modelo oversegmenta (precision=0.0017 a t=0.5): subir lambda_dice o bajar
# gamma son palancas razonables para la siguiente corrida.
LOSS_LAMBDA_DICE = _env_float("MONAI_LOSS_LAMBDA_DICE", 0.5)
LOSS_LAMBDA_FOCAL = _env_float("MONAI_LOSS_LAMBDA_FOCAL", 0.5)
LOSS_FOCAL_GAMMA = _env_float("MONAI_LOSS_FOCAL_GAMMA", 2.0)


# =============================================================================
# Cascada coarse-to-fine
# =============================================================================
TARGET_CASE_MIN_VOXELS = _env_int("MONAI_TARGET_CASE_MIN_VOXELS", 2500)
TARGET_CASE_DICE_THRESHOLD = _env_float("MONAI_TARGET_CASE_DICE_THRESHOLD", 0.7)
TARGET_CASE_SUCCESS_RATE = _env_float("MONAI_TARGET_CASE_SUCCESS_RATE", 0.8)
TARGET_CASE_NEAR_SUCCESS_MARGIN = _env_float("MONAI_TARGET_CASE_NEAR_SUCCESS_MARGIN", 0.05)

NUM_EPOCAS_CASCADE_COARSE = _env_int(
    "MONAI_NUM_EPOCHS_CASCADE_COARSE",
    60 if not FAST_DEV_RUN else 2,
)
NUM_EPOCAS_CASCADE_FINE = _env_int(
    "MONAI_NUM_EPOCHS_CASCADE_FINE",
    100 if not FAST_DEV_RUN else 2,
)
LR_CASCADE_COARSE = _env_float("MONAI_LR_CASCADE_COARSE", 1e-4)
LR_CASCADE_FINE = _env_float("MONAI_LR_CASCADE_FINE", 2e-4)
EARLY_STOPPING_PATIENCE_CASCADE_COARSE = _env_int(
    "MONAI_EARLY_STOPPING_PATIENCE_CASCADE_COARSE",
    12 if not FAST_DEV_RUN else 2,
)
EARLY_STOPPING_PATIENCE_CASCADE_FINE = _env_int(
    "MONAI_EARLY_STOPPING_PATIENCE_CASCADE_FINE",
    15 if not FAST_DEV_RUN else 2,
)
COARSE_SUPPORT_THRESHOLD = _env_float("MONAI_COARSE_SUPPORT_THRESHOLD", 0.35)
CASCADE_THRESHOLD_MIN = _env_float("MONAI_CASCADE_THRESHOLD_MIN", 0.20)
CASCADE_THRESHOLD_MAX = _env_float("MONAI_CASCADE_THRESHOLD_MAX", 0.80)
CASCADE_THRESHOLD_STEP = _env_float("MONAI_CASCADE_THRESHOLD_STEP", 0.05)
CASCADE_MAX_CANDIDATES = _env_int("MONAI_CASCADE_MAX_CANDIDATES", 3)
CASCADE_CANDIDATE_MARGIN_ZYX = (6, 48, 48)
CASCADE_FALLBACK_MARGIN_ZYX = (2, 16, 16)
CASCADE_DILATION_SHAPE = (3, 7, 7)
CASCADE_POSTPROCESS_CLOSING = (1, 3, 3)
CASCADE_FINE_POSITIVE_RATIO = _env_float("MONAI_CASCADE_FINE_POSITIVE_RATIO", 0.7)
CASCADE_FINE_AUGMENT_LEVEL = _env_int("MONAI_CASCADE_FINE_AUGMENT_LEVEL", 1)
CASCADE_ENABLE_TTA_FINAL = _env_bool("MONAI_CASCADE_ENABLE_TTA_FINAL", False)
CASCADE_FINE_USE_COARSE_PRIOR = _env_bool("MONAI_CASCADE_FINE_USE_COARSE_PRIOR", True)
CASCADE_FINE_COARSE_PRIOR_FLOOR = _env_float("MONAI_CASCADE_FINE_COARSE_PRIOR_FLOOR", 0.35)
CASCADE_COARSE_EVAL_USE_ALARM = _env_bool("MONAI_CASCADE_COARSE_EVAL_USE_ALARM", False)

MODELO_CASCADE_COARSE_NOMBRE = "UNet2D-Cascade-Coarse"
MODELO_CASCADE_FINE_NOMBRE = "UNet2D-Cascade-FineROI"
RUTA_MODELO_CASCADE_COARSE = DIR_MODELOS / "mejor_UNet2D_cascade_coarse.pth"
RUTA_MODELO_CASCADE_FINE = DIR_MODELOS / "mejor_UNet2D_cascade_fine.pth"
RUTA_MODELO_CASCADE_FINE_RANK2 = DIR_MODELOS / "mejor_UNet2D_cascade_fine_rank2.pth"


# =============================================================================
# Fase 6 - Ablacion
# =============================================================================
FRACCIONES_DATOS = [0.25, 0.50, 1.00]
NOMBRES_REGIMEN = {
    1: "Base",
    2: "Fuerte",
}


# =============================================================================
# Ventanas CT
# =============================================================================
VENTANAS_CT = {
    "pulmon": (1500, -600),
    "mediastinica": (400, 40),
    "tumor": (600, 40),
}


# =============================================================================
# Resultados 3D legados preservados para comparacion
# =============================================================================
LEGACY_3D_METRICS = [
    {
        "metodo": "DynUNet",
        "tipo": "3D legado",
        "dice_medio": 0.33071640133857727,
        "dice_std": 0.0,
        "hd95_medio_mm": 0.0,
        "sensibilidad": 0.0,
        "especificidad": 0.0,
        "n_eval": 10,
        "parametros": 16539432,
        "tiempo_total_s": 4561.7,
    },
    {
        "metodo": "SegResNet",
        "tipo": "3D legado",
        "dice_medio": 0.31637245416641235,
        "dice_std": 0.0,
        "hd95_medio_mm": 0.0,
        "sensibilidad": 0.0,
        "especificidad": 0.0,
        "n_eval": 10,
        "parametros": 4700914,
        "tiempo_total_s": 6255.7,
    },
    {
        "metodo": "UNet",
        "tipo": "3D legado",
        "dice_medio": 0.21313409507274628,
        "dice_std": 0.0,
        "hd95_medio_mm": 0.0,
        "sensibilidad": 0.0,
        "especificidad": 0.0,
        "n_eval": 10,
        "parametros": 4806481,
        "tiempo_total_s": 1234.8,
    },
]

LEGACY_3D_HISTORIES = {
    "DynUNet": {
        "mejor_dice": 0.33071640133857727,
        "mejor_epoca": 300,
        "n_params": 16539432,
        "tiempo_total_s": 4561.7,
    },
    "SegResNet": {
        "mejor_dice": 0.31637245416641235,
        "mejor_epoca": 290,
        "n_params": 4700914,
        "tiempo_total_s": 6255.7,
    },
    "UNet": {
        "mejor_dice": 0.21313409507274628,
        "mejor_epoca": 220,
        "n_params": 4806481,
        "tiempo_total_s": 1234.8,
    },
}

LEGACY_3D_ABLATION = [
    {
        "fraccion": 0.25,
        "regimen": 1,
        "nombre_regimen": "Sin augmentacion",
        "n_train": 10,
        "mejor_dice": 0.0,
        "loss_final": 0.7722,
        "tiempo_s": 893.6,
    },
    {
        "fraccion": 0.25,
        "regimen": 3,
        "nombre_regimen": "Agresiva",
        "n_train": 10,
        "mejor_dice": 0.0,
        "loss_final": 0.7723,
        "tiempo_s": 858.4,
    },
    {
        "fraccion": 0.50,
        "regimen": 1,
        "nombre_regimen": "Sin augmentacion",
        "n_train": 20,
        "mejor_dice": 0.0,
        "loss_final": 0.7230,
        "tiempo_s": 1709.5,
    },
    {
        "fraccion": 0.50,
        "regimen": 3,
        "nombre_regimen": "Agresiva",
        "n_train": 20,
        "mejor_dice": 0.0,
        "loss_final": 0.7229,
        "tiempo_s": 1756.1,
    },
    {
        "fraccion": 1.0,
        "regimen": 1,
        "nombre_regimen": "Sin augmentacion",
        "n_train": 41,
        "mejor_dice": 0.0,
        "loss_final": 0.6458,
        "tiempo_s": 3134.1,
    },
    {
        "fraccion": 1.0,
        "regimen": 3,
        "nombre_regimen": "Agresiva",
        "n_train": 41,
        "mejor_dice": 0.0,
        "loss_final": 0.6458,
        "tiempo_s": 3278.2,
    },
]


# =============================================================================
# Colores
# =============================================================================
COLORES = {
    "3D legado": "#8d6e63",
    MODELO_2D_NOMBRE: "#1e88e5",
    MODELO_3D_NOMBRE: "#7c3aed",
    MODELO_CASCADE_COARSE_NOMBRE: "#00acc1",
    MODELO_CASCADE_FINE_NOMBRE: "#ff7043",
    "Cascade final": "#f4511e",
    "Base": "#43a047",
    "Fuerte": "#fb8c00",
}

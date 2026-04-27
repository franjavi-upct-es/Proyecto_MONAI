"""
Pipeline 3D patch-based con MONAI idiomatico.

Reemplaza al pipeline 2D/2.5D con priors. La motivacion es que el modelo 2D
no resolvia la localizacion del tumor (caso lung_078: tamano correcto, sitio
incorrecto), mientras que los modelos 3D legados (DynUNet, SegResNet) ya
alcanzaban Dice 0.32-0.33 sin priors. Aqui se aprovecha eso con una
implementacion compacta basada en componentes MONAI.

Componentes usados:
  - Transforms diccionario: LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, RandScaleIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd.
  - CacheDataset para caching en RAM.
  - SegResNet (~4.7M params con init_filters=16).
  - DiceFocalLoss (sigmoid, gamma y lambdas configurables).
  - SlidingWindowInferer (overlap=0.5, mode=gaussian).
  - DiceMetric para validacion.

El balance pos/neg se gestiona via RandCropByPosNegLabeld y elimina ~600 lineas
de manifest custom del pipeline anterior.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)
from scipy import ndimage
from sklearn.model_selection import KFold
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from monai_pipeline.core.config import (
    AMP_ENABLED,
    CACHE_RATE_3D,
    CT_CLIP_RANGE,
    DEVICE,
    EARLY_STOPPING_PATIENCE_3D,
    INFER_OVERLAP_3D,
    INFER_SW_BATCH_3D,
    KFOLD_DEFAULT_INDEX,
    LOSS_FOCAL_GAMMA,
    LOSS_LAMBDA_DICE,
    LOSS_LAMBDA_FOCAL,
    LR_3D,
    LR_WARMUP_EPOCAS,
    MIN_COMPONENT_SIZE,
    MODELO_3D_NOMBRE,
    N_FOLDS_KFOLD,
    N_WORKERS_DATALOADER,
    N_WORKERS_PREPROCESADO,
    PATCH_SIZE_3D,
    POS_NEG_RATIO_3D,
    RUTA_DATASET_JSON,
    SAMPLES_PER_VOLUME_3D,
    SEGRESNET_DROPOUT,
    SEGRESNET_INIT_FILTERS,
    SEMILLA,
    TARGET_SPACING_3D,
    TASK_DIR,
    TRAIN_BATCH_SIZE_3D,
    VAL_INTERVAL_3D,
    WEIGHT_DECAY_3D,
)
from monai_pipeline.core.utils import limpiar_componentes_pequenas


# =============================================================================
# Datos
# =============================================================================
def construir_data_dicts() -> list[dict[str, str]]:
    """Lee el dataset.json y construye la lista [{image, label, case_id}, ...]."""
    if not RUTA_DATASET_JSON.exists():
        raise FileNotFoundError(
            f"No se ha encontrado {RUTA_DATASET_JSON}. "
            "Comprueba que el dataset Task06_Lung este extraido."
        )
    contenido = json.loads(RUTA_DATASET_JSON.read_text(encoding="utf-8"))
    datos: list[dict[str, str]] = []
    for fila in contenido["training"]:
        ruta_img = TASK_DIR / fila["image"].replace("./", "")
        ruta_lbl = TASK_DIR / fila["label"].replace("./", "")
        case_id = Path(fila["image"]).name.replace(".nii.gz", "")
        datos.append(
            {
                "image": str(ruta_img),
                "label": str(ruta_lbl),
                "case_id": case_id,
            }
        )
    return datos


def split_kfold(
    data_dicts: list[dict[str, str]],
    fold_idx: int = KFOLD_DEFAULT_INDEX,
    n_splits: int = N_FOLDS_KFOLD,
    seed: int = SEMILLA,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(data_dicts) < n_splits:
        raise ValueError(f"No hay suficientes casos ({len(data_dicts)}) para {n_splits} folds.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(data_dicts))
    splits = list(kf.split(indices))
    if not 0 <= fold_idx < len(splits):
        raise IndexError(f"fold_idx={fold_idx} fuera de rango (0..{len(splits) - 1}).")
    train_idx, val_idx = splits[fold_idx]
    train = [data_dicts[i] for i in train_idx]
    val = [data_dicts[i] for i in val_idx]
    return train, val


def seleccionar_subconjunto(
    data_dicts: list[dict[str, str]],
    fraccion: float,
    seed: int = SEMILLA,
) -> list[dict[str, str]]:
    if fraccion >= 1.0:
        return list(data_dicts)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(data_dicts))
    rng.shuffle(indices)
    n = max(1, round(len(data_dicts) * fraccion))
    keep = set(indices[:n].tolist())
    return [d for i, d in enumerate(data_dicts) if i in keep]


# =============================================================================
# Transforms
# =============================================================================
KEYS_BOTH = ["image", "label"]


def _base_pre_transforms(con_label: bool = True) -> list:
    """Carga, orientacion, remuestreo, ventaneo y crop de foreground."""
    keys = list(KEYS_BOTH) if con_label else ["image"]
    modos_spacing = ("bilinear", "nearest") if con_label else ("bilinear",)
    return [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=TARGET_SPACING_3D, mode=modos_spacing),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=CT_CLIP_RANGE[0],
            a_max=CT_CLIP_RANGE[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(
            keys=keys,
            source_key="image",
            select_fn=lambda x: x > 0.1,
            allow_smaller=True,
        ),
    ]


def get_train_transforms(augment_level: int) -> Compose:
    """
    Compose de entrenamiento. augment_level:
      0 -> solo crop balanceado, sin augmentacion.
      1 (Base) -> flips, rotaciones 90, shift e scale de intensidad.
      2 (Fuerte) -> base + ruido gaussiano, suavizado y gamma.
    """
    pre = _base_pre_transforms()
    crop = RandCropByPosNegLabeld(
        keys=KEYS_BOTH,
        label_key="label",
        spatial_size=PATCH_SIZE_3D,
        pos=POS_NEG_RATIO_3D,
        neg=1.0,
        num_samples=SAMPLES_PER_VOLUME_3D,
        image_key="image",
        image_threshold=0.0,
        allow_smaller=True,
    )
    aug_base = [
        RandFlipd(keys=KEYS_BOTH, prob=0.5, spatial_axis=0),
        RandFlipd(keys=KEYS_BOTH, prob=0.5, spatial_axis=1),
        RandFlipd(keys=KEYS_BOTH, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=KEYS_BOTH, prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.5),
    ]
    aug_strong = [
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.02),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)
        ),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
    ]
    transforms = list(pre) + [crop]
    if augment_level >= 1:
        transforms.extend(aug_base)
    if augment_level >= 2:
        transforms.extend(aug_strong)
    transforms.append(EnsureTyped(keys=KEYS_BOTH))
    return Compose(transforms)


def get_val_transforms(con_label: bool = True) -> Compose:
    keys = list(KEYS_BOTH) if con_label else ["image"]
    return Compose(_base_pre_transforms(con_label=con_label) + [EnsureTyped(keys=keys)])


# =============================================================================
# Modelo
# =============================================================================
def crear_modelo() -> SegResNet:
    """SegResNet 3D estandar de MONAI."""
    return SegResNet(
        spatial_dims=3,
        init_filters=SEGRESNET_INIT_FILTERS,
        in_channels=1,
        out_channels=1,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=SEGRESNET_DROPOUT,
        norm="instance",
        act="relu",
    )


def contar_parametros(modelo: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in modelo.parameters()))


def cargar_modelo_checkpoint(ruta: Path) -> SegResNet:
    modelo = crear_modelo().to(DEVICE)
    state = torch.load(str(ruta), map_location=DEVICE)
    modelo.load_state_dict(state)
    modelo.eval()
    return modelo


# =============================================================================
# Inferencia
# =============================================================================
def _crear_inferer() -> SlidingWindowInferer:
    return SlidingWindowInferer(
        roi_size=PATCH_SIZE_3D,
        sw_batch_size=INFER_SW_BATCH_3D,
        overlap=INFER_OVERLAP_3D,
        mode="gaussian",
        padding_mode="constant",
    )


def _tensor_a_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _aplicar_modelo_completo(
    modelo: torch.nn.Module,
    imagen: torch.Tensor,
    inferer: SlidingWindowInferer,
) -> torch.Tensor:
    modelo.eval()
    with torch.no_grad():
        with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
            logits = inferer(inputs=imagen, network=modelo)
    return torch.sigmoid(logits.float())


def _inferir_probabilidad_tta(
    modelo: torch.nn.Module,
    imagen: torch.Tensor,
    inferer: SlidingWindowInferer,
) -> torch.Tensor:
    """
    Promedia la probabilidad sobre 4 vistas: original + flip en cada eje
    espacial (x, y, z). Cada flip se deshace antes del promedio.
    """
    probs = [_aplicar_modelo_completo(modelo, imagen, inferer)]
    for axis in (-1, -2, -3):
        flipped = torch.flip(imagen, dims=[axis])
        prob_f = _aplicar_modelo_completo(modelo, flipped, inferer)
        probs.append(torch.flip(prob_f, dims=[axis]))
    return torch.stack(probs, dim=0).mean(dim=0)


def _lung_mask_simple(imagen_np: np.ndarray, hu_threshold: float = 0.32) -> np.ndarray:
    """
    Mascara pulmonar grosera derivada de la propia imagen ya escalada a [0, 1].
    HU=-320 -> ~0.32 en la escala (a_min=-1024, a_max=400). Se queda con la
    region interna de baja densidad mas grande tras llenar agujeros por slice.
    Pensada para postproceso, no para alimentar al modelo.
    """
    cuerpo = imagen_np > 0.05
    aire_interno = (imagen_np < hu_threshold) & cuerpo
    aire_interno = ndimage.binary_opening(aire_interno, structure=np.ones((1, 3, 3)))
    labels, n = ndimage.label(aire_interno)
    if n == 0:
        return np.zeros_like(imagen_np, dtype=np.uint8)
    sizes = ndimage.sum(aire_interno, labels, range(1, n + 1))
    orden = np.argsort(sizes)[::-1]
    keep = np.zeros_like(aire_interno, dtype=bool)
    for idx in orden[:2]:
        keep |= labels == (idx + 1)
    salida = np.zeros_like(keep, dtype=np.uint8)
    for z in range(keep.shape[0]):
        salida[z] = ndimage.binary_fill_holes(keep[z]).astype(np.uint8)
    return salida


def postprocesar_mascara(
    pred: np.ndarray,
    imagen: np.ndarray | None = None,
    cleanup: bool = True,
    use_lung_mask: bool = True,
) -> np.ndarray:
    """
    Postproceso minimo defendible:
      - Restringe la prediccion a una mascara pulmonar gruesa derivada del CT.
      - Filtra componentes pequenos por debajo de MIN_COMPONENT_SIZE.
    """
    pred_proc = pred.astype(np.uint8)
    if use_lung_mask and imagen is not None:
        lung = _lung_mask_simple(imagen)
        pred_proc = (pred_proc & lung).astype(np.uint8)
    if cleanup:
        pred_proc = limpiar_componentes_pequenas(pred_proc, min_size=MIN_COMPONENT_SIZE)
    return pred_proc.astype(np.uint8)


def inferir_caso(
    modelo: torch.nn.Module,
    data_dict: dict[str, str],
    val_transforms: Compose | None = None,
    use_tta: bool = False,
) -> dict[str, np.ndarray | tuple[float, float, float] | str]:
    """
    Aplica las transformaciones de validacion al caso y devuelve probabilidad,
    imagen e idealmente label en el espacio post-Spacingd post-CropForegroundd.
    """
    con_label = "label" in data_dict and data_dict.get("label") is not None
    if val_transforms is None:
        val_transforms = get_val_transforms(con_label=con_label)
    sample_in = {k: v for k, v in data_dict.items() if k in {"image", "label"}}
    sample = val_transforms(sample_in)
    imagen_meta = sample["image"]  # shape (1, X, Y, Z)
    label_meta = sample.get("label") if con_label else None
    imagen = imagen_meta.unsqueeze(0).to(DEVICE)  # (1, 1, X, Y, Z)

    inferer = _crear_inferer()
    if use_tta:
        prob = _inferir_probabilidad_tta(modelo, imagen, inferer)
    else:
        prob = _aplicar_modelo_completo(modelo, imagen, inferer)

    prob_np_xyz = _tensor_a_numpy(prob[0, 0])
    imagen_np_xyz = _tensor_a_numpy(imagen_meta[0])
    prob_np_zyx = np.transpose(prob_np_xyz, (2, 1, 0))
    imagen_np_zyx = np.transpose(imagen_np_xyz, (2, 1, 0))

    salida: dict[str, np.ndarray | tuple[float, float, float] | str] = {
        "case_id": str(data_dict.get("case_id", Path(data_dict["image"]).name)),
        "probability": prob_np_zyx.astype(np.float32),
        "image": imagen_np_zyx.astype(np.float32),
        "spacing": (
            float(TARGET_SPACING_3D[2]),
            float(TARGET_SPACING_3D[1]),
            float(TARGET_SPACING_3D[0]),
        ),
    }
    if label_meta is not None:
        label_np_xyz = _tensor_a_numpy(label_meta[0])
        salida["label"] = (np.transpose(label_np_xyz, (2, 1, 0)) > 0.5).astype(np.uint8)
    return salida


def aplicar_threshold(
    inferencia: dict,
    threshold: float,
    cleanup: bool = True,
    use_lung_mask: bool = True,
) -> np.ndarray:
    pred = (inferencia["probability"] >= threshold).astype(np.uint8)
    return postprocesar_mascara(
        pred,
        imagen=inferencia.get("image"),
        cleanup=cleanup,
        use_lung_mask=use_lung_mask,
    )


# =============================================================================
# Entrenamiento
# =============================================================================
def _build_loaders(
    train_files: list[dict[str, str]],
    val_files: list[dict[str, str]],
    augment_level: int,
) -> tuple[DataLoader, DataLoader, CacheDataset, CacheDataset]:
    train_tf = get_train_transforms(augment_level)
    val_tf = get_val_transforms()
    train_ds = CacheDataset(
        data=train_files,
        transform=train_tf,
        cache_rate=CACHE_RATE_3D,
        num_workers=N_WORKERS_PREPROCESADO,
        copy_cache=False,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_tf,
        cache_rate=CACHE_RATE_3D,
        num_workers=N_WORKERS_PREPROCESADO,
        copy_cache=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE_3D,
        shuffle=True,
        num_workers=N_WORKERS_DATALOADER,
        pin_memory=AMP_ENABLED,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=N_WORKERS_DATALOADER,
        pin_memory=AMP_ENABLED,
    )
    return train_loader, val_loader, train_ds, val_ds


def entrenar_modelo(
    train_files: list[dict[str, str]],
    val_files: list[dict[str, str]],
    augment_level: int,
    num_epocas: int,
    checkpoint_path: Path | None,
    descripcion: str | None = None,
    fold_idx: int | None = None,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE_3D,
    val_interval: int = VAL_INTERVAL_3D,
) -> tuple[SegResNet, dict[str, object]]:
    train_loader, val_loader, _train_ds, _val_ds = _build_loaders(
        train_files, val_files, augment_level
    )

    modelo = crear_modelo().to(DEVICE)
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR_3D, weight_decay=WEIGHT_DECAY_3D)
    warmup = min(LR_WARMUP_EPOCAS, num_epocas)
    cosine = max(num_epocas - warmup, 1)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, end_factor=1.0, total_iters=warmup
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine),
        ],
        milestones=[warmup],
    )
    criterio = DiceFocalLoss(
        sigmoid=True,
        gamma=LOSS_FOCAL_GAMMA,
        lambda_dice=LOSS_LAMBDA_DICE,
        lambda_focal=LOSS_LAMBDA_FOCAL,
    )
    scaler = GradScaler(DEVICE.type, enabled=AMP_ENABLED)

    inferer = _crear_inferer()
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = AsDiscrete(threshold=0.5)

    historia: dict[str, object] = {
        "descripcion": descripcion or MODELO_3D_NOMBRE,
        "modelo": MODELO_3D_NOMBRE,
        "fold_idx": fold_idx,
        "n_folds": N_FOLDS_KFOLD,
        "n_params": contar_parametros(modelo),
        "patch_size": list(PATCH_SIZE_3D),
        "samples_per_volume": SAMPLES_PER_VOLUME_3D,
        "warmup_epocas": warmup,
        "loss_fn": (
            f"DiceFocalLoss(gamma={LOSS_FOCAL_GAMMA},"
            f" lambda_dice={LOSS_LAMBDA_DICE},"
            f" lambda_focal={LOSS_LAMBDA_FOCAL})"
        ),
        "augment_level": augment_level,
        "train_loss": [],
        "val_dice": [],
        "val_epochs": [],
    }

    mejor_dice = -1.0
    mejor_epoca = 0
    paciencia = 0
    mejor_state = None
    t0 = time.time()

    for epoca in range(1, num_epocas + 1):
        modelo.train()
        loss_acc = 0.0
        n_batches = 0
        for batch in train_loader:
            imagenes = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                logits = modelo(imagenes)
                loss = criterio(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            loss_acc += float(loss.item())
            n_batches += 1
        sched.step()
        avg_loss = loss_acc / max(n_batches, 1)
        cast_loss = historia["train_loss"]
        if isinstance(cast_loss, list):
            cast_loss.append(round(avg_loss, 4))

        if epoca % val_interval == 0 or epoca == num_epocas:
            modelo.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    imagen = batch["image"].to(DEVICE)
                    label = batch["label"].to(DEVICE)
                    with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                        logits = inferer(inputs=imagen, network=modelo)
                    preds = [post_pred(p) for p in decollate_batch(logits)]
                    lbls = [post_label(l) for l in decollate_batch(label)]
                    dice_metric(y_pred=preds, y=lbls)
            val_dice = float(dice_metric.aggregate().item())
            cast_val = historia["val_dice"]
            cast_eps = historia["val_epochs"]
            if isinstance(cast_val, list):
                cast_val.append(round(val_dice, 4))
            if isinstance(cast_eps, list):
                cast_eps.append(epoca)

            if val_dice > mejor_dice:
                mejor_dice = val_dice
                mejor_epoca = epoca
                paciencia = 0
                mejor_state = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
                if checkpoint_path is not None:
                    torch.save(mejor_state, checkpoint_path)
            else:
                paciencia += 1

            lr_actual = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else LR_3D
            print(
                f"  ep {epoca:>3d}/{num_epocas} | loss {avg_loss:.4f} | "
                f"val_dice {val_dice:.4f} | mejor {mejor_dice:.4f} | lr {lr_actual:.2e}"
            )

            if paciencia >= early_stopping_patience:
                print(f"  early stopping en epoca {epoca}")
                break

    if mejor_state is not None:
        modelo.load_state_dict(mejor_state)

    historia["mejor_dice"] = round(max(mejor_dice, 0.0), 4)
    historia["mejor_epoca"] = mejor_epoca
    historia["tiempo_total_s"] = round(time.time() - t0, 1)
    return modelo, historia


# =============================================================================
# Wrappers de evaluacion (para apps)
# =============================================================================
def inferir_todos(
    modelo: torch.nn.Module,
    val_files: list[dict[str, str]],
    use_tta: bool = False,
) -> list[dict]:
    """Devuelve una lista de inferencias (probability + image + label) por caso."""
    val_tf = get_val_transforms()
    salidas = []
    etiqueta = "TTA" if use_tta else "single"
    for i, df in enumerate(val_files, start=1):
        case_id = df.get("case_id", Path(df["image"]).name)
        print(f"  inferencia {etiqueta}: {i}/{len(val_files)} {case_id}")
        salidas.append(inferir_caso(modelo, df, val_transforms=val_tf, use_tta=use_tta))
    return salidas

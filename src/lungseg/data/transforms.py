"""Transformaciones MONAI específicas para TC de tórax.

Pipeline (refactor 2.5D):
- Carga + reorientación RAS + Spacing al `target_spacing` configurado.
- Máscara pulmonar (lungmask R231 cacheada) aplicada como prior duro:
  voxels no-pulmón se sustituyen por `lung_mask.fill_value` HU.
- CropForegroundd por umbral HU absoluto, antes de expandir canales.
- Multi-ventana HU: el canal único recortado se sustituye por 3 canales con
  ventanas pulmonar / mediastínica / completa, normalizadas a [0, 1].
- Sampler RandCropByPosNegLabeld (pos/neg/num_samples desde cfg.data.sampler).
- Aumentos controlados por cfg.training.augment_regime; RandFlipd solo en Z.

Reglas defensivas: `_check_no_lr_flip` aborta si una config introduce
RandFlipd con `spatial_axis=0` (LR), prohibido por CLAUDE.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandomizableTrait,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResampleToMatch,
    Spacingd,
)
from omegaconf import DictConfig

KEYS = ["image", "label"]
_CROP_METADATA_KEYS = ["foreground_start_coord", "foreground_end_coord"]
_AUG_PROB = {"none": 0.0, "standard": 0.15, "aggressive": 0.30}

_DEFAULT_HU_WINDOWS: tuple[tuple[float, float], ...] = (
    (-1000.0, 0.0),     # pulmón
    (-150.0, 250.0),    # mediastino
    (-1024.0, 400.0),   # completa
)

LUNG_MASK_PATH_KEY = "lung_mask_path"


class MaskNonLungVoxelsd(MapTransform):
    """Aplica la máscara pulmonar cacheada como prior duro al canal `image`.

    Carga el NIfTI desde `data[LUNG_MASK_PATH_KEY]`, lo remuestrea al grid
    actual de la imagen (post-Spacingd) usando `nearest`, y reemplaza los
    voxels marcados como no-pulmón por `fill_value` (HU). Debe aplicarse
    después de LoadImaged + Spacingd y antes de cualquier ventaneo.

    Lanza FileNotFoundError con mensaje claro si la máscara no está
    cacheada — el usuario debe haber corrido `lungseg precompute-lung-masks`
    antes.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image",),
        fill_value: float = -1024.0,
        path_key: str = LUNG_MASK_PATH_KEY,
    ) -> None:
        super().__init__(keys=tuple(keys), allow_missing_keys=False)
        self.fill_value = float(fill_value)
        self.path_key = path_key
        self._resampler = ResampleToMatch(mode="nearest", padding_mode="zeros")

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        mask_path = d.get(self.path_key)
        if mask_path is None:
            raise KeyError(
                f"MaskNonLungVoxelsd: el data dict no tiene la clave "
                f"{self.path_key!r}; el dataloader debe inyectarla."
            )
        path = Path(str(mask_path))
        if not path.exists():
            raise FileNotFoundError(
                f"Máscara pulmonar no encontrada en {path}. "
                "Corre `lungseg precompute-lung-masks` antes de entrenar."
            )

        nii = nib.load(str(path))
        mask_array = np.asarray(nii.get_fdata(), dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_array)[None]  # (1, H, W, D)
        mask_meta = MetaTensor(mask_tensor, affine=torch.as_tensor(nii.affine))

        for key in self.keys:
            image = d[key]
            if not isinstance(image, MetaTensor):
                raise TypeError(
                    f"MaskNonLungVoxelsd espera un MetaTensor en {key!r}, "
                    f"recibió {type(image).__name__}; falta Spacingd previo."
                )
            resampled = self._resampler(mask_meta, image)
            mask_bool = resampled.as_tensor().to(torch.bool)
            image_t = image.as_tensor().clone()
            image_t[~mask_bool] = self.fill_value
            # Construye un MetaTensor NUEVO; no mutar el de entrada por si vive
            # dentro del cache de un CacheDataset(copy_cache=False).
            new_image = MetaTensor(image_t)
            new_image.copy_meta_from(image)
            d[key] = new_image
        return d


class MultiWindowHUd(MapTransform, RandomizableTrait):
    """Sustituye el canal único de HU por 3 canales con ventanas distintas.

    Para cada ventana `(a_min, a_max)` el canal de salida es
    `(clip(x, a_min, a_max) - a_min) / (a_max - a_min)`, saturado en [0, 1].
    Output: `(C_windows, H, W, D)` reemplazando `(1, H, W, D)`.

    Hereda de `RandomizableTrait` (sin RNG real) para que `CacheDataset` lo
    trate como per-sample y no expanda los volúmenes a 3 canales en cache.
    Aplicar siempre **después** del sampler en train, o como último paso en
    val — nunca antes del crop, porque triplicaría la huella de memoria del
    cache (≈120 MB x 3 x N volúmenes).
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image",),
        windows: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(keys=tuple(keys), allow_missing_keys=False)
        if windows is None:
            windows = _DEFAULT_HU_WINDOWS
        parsed: list[tuple[float, float]] = []
        for w in windows:
            if len(w) != 2:
                raise ValueError(f"hu_windows debe contener pares (a_min, a_max); recibido {w}")
            a_min, a_max = float(w[0]), float(w[1])
            if a_max <= a_min:
                raise ValueError(f"hu window inválida: a_max ({a_max}) <= a_min ({a_min})")
            parsed.append((a_min, a_max))
        if not parsed:
            raise ValueError("hu_windows está vacío; provee al menos una ventana")
        self.windows = tuple(parsed)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            image = d[key]
            tensor = image.as_tensor() if hasattr(image, "as_tensor") else image
            if tensor.ndim != 4 or tensor.shape[0] != 1:
                raise ValueError(
                    f"MultiWindowHUd espera (1, H, W, D) en {key!r}, "
                    f"recibió {tuple(tensor.shape)}"
                )
            base = tensor[0].to(torch.float32)
            channels = []
            for a_min, a_max in self.windows:
                normalized = (base.clamp(min=a_min, max=a_max) - a_min) / (a_max - a_min)
                channels.append(normalized)
            stacked = torch.stack(channels, dim=0)
            if isinstance(image, MetaTensor):
                # Construye un MetaTensor NUEVO; no mutar el de entrada por si
                # vive dentro del cache de un CacheDataset(copy_cache=False) —
                # mutarlo corrompería el cache y dispararía COW masivo en cada
                # worker (típico OOM en mid-training).
                new_image = MetaTensor(stacked)
                new_image.copy_meta_from(image)
                d[key] = new_image
            else:
                d[key] = stacked
        return d


def _check_no_lr_flip(transforms: list) -> None:
    """Guardia defensiva contra la reintroducción del error (c) de REPORT_DIAGNOSIS."""
    for tr in transforms:
        if isinstance(tr, RandFlipd):
            axis = tr.flipper.spatial_axis
            if axis is None:
                raise ValueError(
                    "RandFlipd without spatial_axis flips all axes, including LR. "
                    "Forbidden on chest CT (CLAUDE.md hard rule)."
                )
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            if 0 in axes:
                raise ValueError(
                    "RandFlipd spatial_axis=0 (LR) is forbidden on chest CT "
                    "(CLAUDE.md hard rule). Use spatial_axis=2 only."
                )


def _hu_windows_from_cfg(cfg: DictConfig) -> tuple[tuple[float, float], ...]:
    raw = cfg.data.get("hu_windows", None) if "data" in cfg else None
    if raw is None:
        return _DEFAULT_HU_WINDOWS
    return tuple((float(a), float(b)) for a, b in raw)


def _lung_fill_value(cfg: DictConfig) -> float:
    if "data" in cfg and "lung_mask" in cfg.data:
        return float(cfg.data.lung_mask.get("fill_value", -1024.0))
    return -1024.0


def _pre_transforms(cfg: DictConfig, with_label: bool = True) -> list:
    """Pipeline cacheable: 1 canal HU recortado al pulmón.

    `MultiWindowHUd` queda **fuera** intencionalmente: aplicarlo aquí
    dispararía el cache a 3x la memoria. Se inserta más tarde en train/val
    (después del sampler / como per-sample en val).
    """
    keys = list(KEYS) if with_label else ["image"]
    spacing_modes = ("bilinear", "nearest") if with_label else ("bilinear",)
    return [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=tuple(cfg.data.target_spacing), mode=spacing_modes),
        MaskNonLungVoxelsd(keys=["image"], fill_value=_lung_fill_value(cfg)),
        CropForegroundd(
            keys=keys,
            source_key="image",
            select_fn=lambda x, t=float(cfg.data.crop_foreground.threshold): x > t,
            allow_smaller=True,
        ),
    ]


def _rotate90_augmentation(prob: float, patch_size: Sequence[int]) -> list:
    if len(patch_size) < 2 or int(patch_size[0]) == int(patch_size[1]):
        return [RandRotate90d(keys=KEYS, prob=prob, max_k=3, spatial_axes=(0, 1))]
    return []


def _augmentations(prob: float, patch_size: Sequence[int]) -> list:
    if prob <= 0.0:
        return []
    return [
        RandFlipd(keys=KEYS, prob=prob, spatial_axis=2),
        *_rotate90_augmentation(prob, patch_size),
        RandGaussianNoised(keys=["image"], prob=prob, mean=0.0, std=0.02),
        RandGaussianSmoothd(
            keys=["image"],
            prob=prob,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),
        RandScaleIntensityd(keys=["image"], factors=0.10, prob=prob),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=prob),
    ]


def build_train_transforms(cfg: DictConfig) -> Compose:
    regime = str(cfg.training.augment_regime)
    if regime not in _AUG_PROB:
        raise ValueError(f"unknown augment_regime: {regime!r} (expected one of {list(_AUG_PROB)})")
    prob = _AUG_PROB[regime]
    pre = _pre_transforms(cfg)
    crop = RandCropByPosNegLabeld(
        keys=KEYS,
        label_key="label",
        spatial_size=tuple(cfg.training.patch_size),
        pos=float(cfg.data.sampler.pos),
        neg=float(cfg.data.sampler.neg),
        num_samples=int(cfg.data.sampler.num_samples),
        image_key="image",
        image_threshold=0.0,
        allow_smaller=True,
    )
    multi_window = MultiWindowHUd(keys=["image"], windows=_hu_windows_from_cfg(cfg))
    aug = _augmentations(prob, cfg.training.patch_size)
    # MultiWindow corre sobre el patch (96x96x16), no sobre el volumen completo:
    # 1 canal cacheado en RAM, 3 canales solo durante el forward.
    transforms = [
        *pre,
        crop,
        multi_window,
        *aug,
        EnsureTyped(keys=[*KEYS, *_CROP_METADATA_KEYS], allow_missing_keys=True),
    ]
    _check_no_lr_flip(transforms)
    return Compose(transforms)


def build_val_transforms(cfg: DictConfig, with_label: bool = True) -> Compose:
    keys = list(KEYS) if with_label else ["image"]
    # MultiWindowHUd hereda de RandomizableTrait, así que CacheDataset NO lo
    # cachea: el volumen guardado en RAM mantiene 1 canal.
    return Compose(
        [
            *_pre_transforms(cfg, with_label=with_label),
            MultiWindowHUd(keys=["image"], windows=_hu_windows_from_cfg(cfg)),
            EnsureTyped(keys=[*keys, *_CROP_METADATA_KEYS], allow_missing_keys=True),
        ]
    )

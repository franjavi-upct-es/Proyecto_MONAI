"""
Pipeline de segmentacion coarse-to-fine con ROI anatomica.

Objetivo:
  - Mejorar la deteccion caso a caso en tumores no diminutos.
  - Reducir el desbalance espacial: primero localizar con alta sensibilidad,
    despues refinar dentro de candidatos volumetricos plausibles.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.inferers.utils import sliding_window_inference
from monai.losses import DiceCELoss, TverskyLoss
from scipy import ndimage
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.functional import binary_cross_entropy_with_logits, interpolate
from torch.utils.data import DataLoader, Dataset

from monai_pipeline.core.config import (
    AMP_ENABLED,
    CASCADE_CANDIDATE_MARGIN_ZYX,
    CASCADE_COARSE_EVAL_USE_ALARM,
    CASCADE_DILATION_SHAPE,
    CASCADE_ENABLE_TTA_FINAL,
    CASCADE_FALLBACK_MARGIN_ZYX,
    CASCADE_FINE_AUGMENT_LEVEL,
    CASCADE_FINE_COARSE_PRIOR_FLOOR,
    CASCADE_FINE_POSITIVE_RATIO,
    CASCADE_FINE_USE_COARSE_PRIOR,
    CASCADE_MAX_CANDIDATES,
    CASCADE_POSTPROCESS_CLOSING,
    CASCADE_THRESHOLD_MAX,
    CASCADE_THRESHOLD_MIN,
    CASCADE_THRESHOLD_STEP,
    COARSE_SUPPORT_THRESHOLD,
    CONTEXT_SLICES,
    DEVICE,
    EARLY_STOPPING_PATIENCE_CASCADE_COARSE,
    EARLY_STOPPING_PATIENCE_CASCADE_FINE,
    FAST_DEV_RUN,
    FINE_ROI_SIZE_2D,
    INFER_BATCH_SIZE,
    INFER_BATCH_SIZE_FINE,
    LR_CASCADE_COARSE,
    LR_CASCADE_FINE,
    LR_WARMUP_EPOCAS,
    MIN_COMPONENT_SIZE,
    MODELO_CASCADE_COARSE_NOMBRE,
    MODELO_CASCADE_FINE_NOMBRE,
    N_WORKERS_DATALOADER,
    NUM_EPOCAS_CASCADE_COARSE,
    NUM_EPOCAS_CASCADE_FINE,
    PATCH_SIZE_2D,
    PRIOR_CHANNELS_2D,
    RUTA_MODELO_CASCADE_COARSE,
    RUTA_MODELO_CASCADE_FINE,
    RUTA_MODELO_CASCADE_FINE_RANK2,
    SEMILLA,
    TARGET_CASE_DICE_THRESHOLD,
    TARGET_CASE_MIN_VOXELS,
    TARGET_CASE_NEAR_SUCCESS_MARGIN,
    TARGET_CASE_SUCCESS_RATE,
    TRAIN_BATCH_SIZE,
    TRAIN_BATCH_SIZE_FINE,
    VAL_INTERVAL,
    WEIGHT_DECAY,
)
from monai_pipeline.core.utils import evaluar_segmentacion, limpiar_componentes_pequenas
from monai_pipeline.pipelines.baseline2d import (
    Case2D,
    SliceDataset,
    construir_entrada_modelo,
    contar_parametros,
    crear_manifest_entrenamiento,
    crear_modelo_2d,
    split_por_fold,
)

BBoxZYX = tuple[int, int, int, int, int, int]


@dataclass(slots=True)
class ROIProposal:
    z0: int
    z1: int
    y0: int
    y1: int
    x0: int
    x1: int
    source: str
    score: float
    size: int
    mean_prob: float
    overlap_gt_voxels: int

    @property
    def bbox(self) -> BBoxZYX:
        return (self.z0, self.z1, self.y0, self.y1, self.x0, self.x1)


@dataclass(slots=True)
class CaseProposals:
    candidates: list[ROIProposal]
    fallback: ROIProposal

    def rois_for_inference(self) -> list[ROIProposal]:
        return self.candidates if self.candidates else [self.fallback]

    def rois_for_audit(self) -> list[ROIProposal]:
        return [*self.candidates, self.fallback]


@dataclass(slots=True)
class FineSliceItem:
    case_idx: int
    roi: ROIProposal
    z: int
    is_positive: bool


def es_caso_objetivo(voxeles_gt: int) -> bool:
    return int(voxeles_gt) >= TARGET_CASE_MIN_VOXELS


def _score_tuple(summary: dict[str, float | int]) -> tuple[float, float, float]:
    return (
        float(summary["target_pass_rate"]),
        float(summary["target_mean_dice"]),
        float(summary["all_case_mean_dice"]),
    )


def _build_thresholds() -> list[float]:
    thresholds = np.arange(
        CASCADE_THRESHOLD_MIN,
        CASCADE_THRESHOLD_MAX + (CASCADE_THRESHOLD_STEP * 0.5),
        CASCADE_THRESHOLD_STEP,
    )
    return [round(float(x), 2) for x in thresholds]


def _bbox_from_mask(mask: np.ndarray) -> BBoxZYX | None:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return (
        int(mins[0]),
        int(maxs[0]),
        int(mins[1]),
        int(maxs[1]),
        int(mins[2]),
        int(maxs[2]),
    )


def _expand_bbox(
    bbox: BBoxZYX, shape: tuple[int, int, int], margins: tuple[int, int, int]
) -> BBoxZYX:
    z0, z1, y0, y1, x0, x1 = bbox
    mz, my, mx = margins
    return (
        max(0, z0 - mz),
        min(shape[0], z1 + mz),
        max(0, y0 - my),
        min(shape[1], y1 + my),
        max(0, x0 - mx),
        min(shape[2], x1 + mx),
    )


def _bbox_contains(outer: BBoxZYX, inner: BBoxZYX) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] >= inner[1]
        and outer[2] <= inner[2]
        and outer[3] >= inner[3]
        and outer[4] <= inner[4]
        and outer[5] >= inner[5]
    )


def _crop_zyx(arr: np.ndarray, bbox: BBoxZYX) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bbox
    return arr[z0:z1, y0:y1, x0:x1]


def _crop_hw(arr: np.ndarray, bbox: BBoxZYX) -> np.ndarray:
    _, _, y0, y1, x0, x1 = bbox
    return arr[..., y0:y1, x0:x1]


def _resize_chw(image: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(image[None].astype(np.float32))
    kwargs: dict[str, object] = {"size": size, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    resized = interpolate(tensor, **kwargs)
    return resized[0].cpu().numpy()


def _resize_bchw(image: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(image.astype(np.float32))
    kwargs: dict[str, object] = {"size": size, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    resized = interpolate(tensor, **kwargs)
    return resized.cpu().numpy()


def _postprocess_mask(mask: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    pred = (mask > 0) & (lung_mask > 0)
    pred = ndimage.binary_closing(pred, structure=np.ones(CASCADE_POSTPROCESS_CLOSING, dtype=bool))
    pred = limpiar_componentes_pequenas(
        pred.astype(np.uint8), min_size=MIN_COMPONENT_SIZE, max_componentes=128
    )
    return pred.astype(np.uint8)


def _aplicar_prior_coarse(
    prob_fine: np.ndarray,
    coarse_prob: np.ndarray | None,
) -> np.ndarray:
    if coarse_prob is None or not CASCADE_FINE_USE_COARSE_PRIOR:
        return prob_fine
    floor = float(np.clip(CASCADE_FINE_COARSE_PRIOR_FLOOR, 0.0, 1.0))
    prior = floor + ((1.0 - floor) * np.clip(coarse_prob.astype(np.float32), 0.0, 1.0))
    return (prob_fine.astype(np.float32) * prior).astype(np.float32)


def anotar_target_cases(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["is_target_case"] = out["voxeles_gt"] >= TARGET_CASE_MIN_VOXELS
    out["passes_target_dice"] = out["dice"] >= TARGET_CASE_DICE_THRESHOLD
    return out


def resumir_metricas_cascade(df: pd.DataFrame) -> dict[str, float | int]:
    df_ann = anotar_target_cases(df)
    target_df = df_ann[df_ann["is_target_case"]]
    n_target = len(target_df)
    n_target_passed = int(target_df["passes_target_dice"].sum()) if n_target > 0 else 0
    return {
        "n_cases": len(df_ann),
        "n_target_cases": n_target,
        "n_target_passed": n_target_passed,
        "all_case_mean_dice": round(float(df_ann["dice"].mean()) if not df_ann.empty else 0.0, 4),
        "all_case_median_dice": round(
            float(df_ann["dice"].median()) if not df_ann.empty else 0.0, 4
        ),
        "target_mean_dice": round(float(target_df["dice"].mean()) if n_target > 0 else 0.0, 4),
        "target_pass_rate": round(float(n_target_passed / max(n_target, 1)), 4),
    }


class RecallDrivenLoss(nn.Module):
    def __init__(self, pos_weight: float = 8.0) -> None:
        super().__init__()
        self.tversky = TverskyLoss(sigmoid=True, alpha=0.2, beta=0.8)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_tversky = self.tversky(logits, target)
        loss_bce = binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        return (0.7 * loss_tversky) + (0.3 * loss_bce)


def _augment_2d(
    image: np.ndarray, label: np.ndarray, augment_level: int, train: bool
) -> tuple[np.ndarray, np.ndarray]:
    if augment_level <= 0 or not train:
        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    if np.random.rand() < 0.5:
        image = np.flip(image, axis=1)
        label = np.flip(label, axis=1)
    if np.random.rand() < 0.5:
        image = np.flip(image, axis=2)
        label = np.flip(label, axis=2)
    if np.random.rand() < 0.5:
        k = int(np.random.randint(0, 4))
        image = np.rot90(image, k=k, axes=(1, 2))
        label = np.rot90(label, k=k, axes=(1, 2))

    if augment_level >= 2:
        if np.random.rand() < 0.3:
            image = np.clip(image + np.random.normal(0.0, 0.02, size=image.shape), 0.0, 1.0)
        if np.random.rand() < 0.3:
            shift = float(np.random.uniform(-0.08, 0.08))
            image = np.clip(image + shift, 0.0, 1.0)
        if np.random.rand() < 0.25:
            gain = float(np.random.uniform(0.9, 1.15))
            image = np.clip((image - 0.5) * gain + 0.5, 0.0, 1.0)
        if np.random.rand() < 0.3:
            gamma = float(np.random.uniform(0.7, 1.5))
            n_img_channels = (2 * CONTEXT_SLICES) + 1
            image[:n_img_channels] = np.clip(image[:n_img_channels] ** gamma, 0.0, 1.0)

    return np.ascontiguousarray(image), np.ascontiguousarray(label)


class FineSliceDataset(Dataset):
    def __init__(
        self,
        casos: list[Case2D],
        manifest: list[FineSliceItem],
        augment_level: int = 0,
        train: bool = True,
    ) -> None:
        self.casos = casos
        self.manifest = manifest
        self.augment_level = augment_level
        self.train = train

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        item = self.manifest[idx]
        caso = self.casos[item.case_idx]
        if caso.label is None:
            raise ValueError(f"El caso {caso.case_id} no tiene mascara.")

        full_input = construir_entrada_modelo(caso, item.z)
        cropped_input = _crop_hw(full_input, item.roi.bbox)
        cropped_label = _crop_hw(caso.label[item.z][None].astype(np.float32), item.roi.bbox)

        image = _resize_chw(cropped_input, FINE_ROI_SIZE_2D, mode="bilinear")
        label = _resize_chw(cropped_label, FINE_ROI_SIZE_2D, mode="nearest")
        image, label = _augment_2d(image, label, augment_level=self.augment_level, train=self.train)

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "label": torch.from_numpy(label.astype(np.float32)),
            "case_id": caso.case_id,
            "slice_idx": int(item.z),
            "is_positive": int(item.is_positive),
        }


def _inferir_probabilidades_coarse_batch(
    modelo: torch.nn.Module,
    batch_np: np.ndarray,
    flip_h: bool = False,
    flip_v: bool = False,
) -> np.ndarray:
    if flip_h:
        batch_np = np.flip(batch_np, axis=-1).copy()
    if flip_v:
        batch_np = np.flip(batch_np, axis=-2).copy()

    batch = torch.from_numpy(batch_np.astype(np.float32)).to(DEVICE)
    with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
        logits = sliding_window_inference(
            inputs=batch,
            roi_size=PATCH_SIZE_2D,
            sw_batch_size=max(1, batch.shape[0]),
            predictor=modelo,
            overlap=0.5,
            mode="gaussian",
            padding_mode="constant",
        )
    prob = torch.sigmoid(logits).cpu().numpy()[:, 0]
    if flip_h:
        prob = np.flip(prob, axis=-1).copy()
    if flip_v:
        prob = np.flip(prob, axis=-2).copy()
    return prob


def inferir_probabilidades_coarse_caso(
    modelo: torch.nn.Module,
    caso: Case2D,
    batch_size: int = INFER_BATCH_SIZE,
    use_tta: bool = False,
) -> np.ndarray:
    modelo.eval()
    probs_orig: list[np.ndarray] = []
    probs_fh: list[np.ndarray] = []
    probs_fv: list[np.ndarray] = []

    with torch.no_grad():
        for inicio in range(0, caso.image.shape[0], batch_size):
            fin = min(caso.image.shape[0], inicio + batch_size)
            batch_np = np.stack(
                [construir_entrada_modelo(caso, z) for z in range(inicio, fin)],
                axis=0,
            ).astype(np.float32)
            probs_orig.append(_inferir_probabilidades_coarse_batch(modelo, batch_np))
            if use_tta:
                probs_fh.append(_inferir_probabilidades_coarse_batch(modelo, batch_np, flip_h=True))
                probs_fv.append(_inferir_probabilidades_coarse_batch(modelo, batch_np, flip_v=True))

    prob = np.concatenate(probs_orig, axis=0)
    if use_tta:
        prob = (prob + np.concatenate(probs_fh, axis=0) + np.concatenate(probs_fv, axis=0)) / 3.0
    return prob.astype(np.float32)


def inferir_probabilidades_coarse_casos(
    modelo: torch.nn.Module,
    casos: list[Case2D],
    use_tta: bool = False,
    prefix: str = "",
) -> dict[str, np.ndarray]:
    prob_maps: dict[str, np.ndarray] = {}
    for idx, caso in enumerate(casos, start=1):
        if prefix:
            print(f"  {prefix} {idx:>2d}/{len(casos)} -> {caso.case_id}")
        prob_maps[caso.case_id] = inferir_probabilidades_coarse_caso(modelo, caso, use_tta=use_tta)
    return prob_maps


def _crear_fallback_proposal(caso: Case2D) -> ROIProposal:
    bbox = _bbox_from_mask(caso.lung_mask)
    if bbox is None:
        bbox = _bbox_from_mask(caso.soft_tissue_alarm_mask)
    if bbox is None:
        bbox = (0, caso.image.shape[0], 0, caso.image.shape[1], 0, caso.image.shape[2])
    bbox = _expand_bbox(bbox, caso.image.shape, CASCADE_FALLBACK_MARGIN_ZYX)
    z0, z1, y0, y1, x0, x1 = bbox
    return ROIProposal(
        z0=z0,
        z1=z1,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        source="fallback_lung",
        score=0.0,
        size=int(max((z1 - z0) * (y1 - y0) * (x1 - x0), 1)),
        mean_prob=0.0,
        overlap_gt_voxels=0,
    )


def generar_propuestas_caso(caso: Case2D, coarse_prob: np.ndarray) -> CaseProposals:
    support = ((coarse_prob >= COARSE_SUPPORT_THRESHOLD) | (caso.soft_tissue_alarm_mask > 0)) & (
        caso.lung_mask > 0
    )
    support = ndimage.binary_dilation(
        support, structure=np.ones(CASCADE_DILATION_SHAPE, dtype=bool)
    )
    labels, n = ndimage.label(support)

    gt = caso.label > 0 if caso.label is not None else np.zeros_like(support, dtype=bool)
    candidates: list[ROIProposal] = []

    for etiqueta in range(1, n + 1):
        comp = labels == etiqueta
        size = int(np.sum(comp))
        if size == 0:
            continue
        bbox = _bbox_from_mask(comp)
        if bbox is None:
            continue
        bbox = _expand_bbox(bbox, caso.image.shape, CASCADE_CANDIDATE_MARGIN_ZYX)
        z0, z1, y0, y1, x0, x1 = bbox
        mean_prob = float(coarse_prob[comp].mean()) if np.any(comp) else 0.0
        overlap_gt = int(np.sum(comp & gt))
        score = float(mean_prob * (size**0.25))
        candidates.append(
            ROIProposal(
                z0=z0,
                z1=z1,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                source="coarse_component",
                score=score,
                size=size,
                mean_prob=mean_prob,
                overlap_gt_voxels=overlap_gt,
            )
        )

    candidates = sorted(candidates, key=lambda roi: roi.score, reverse=True)[
        :CASCADE_MAX_CANDIDATES
    ]
    fallback = _crear_fallback_proposal(caso)
    return CaseProposals(candidates=candidates, fallback=fallback)


def generar_propuestas_casos(
    casos: list[Case2D],
    coarse_probs: dict[str, np.ndarray],
) -> dict[str, CaseProposals]:
    propuestas: dict[str, CaseProposals] = {}
    for caso in casos:
        propuestas[caso.case_id] = generar_propuestas_caso(caso, coarse_probs[caso.case_id])
    return propuestas


def auditar_propuestas(
    casos: list[Case2D],
    propuestas: dict[str, CaseProposals],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    filas: list[dict[str, object]] = []
    for caso in casos:
        if caso.label is None:
            continue
        voxels_gt = int(np.sum(caso.label > 0))
        bbox_gt = _bbox_from_mask(caso.label)
        if bbox_gt is None:
            continue
        bundle = propuestas[caso.case_id]
        covered = any(_bbox_contains(roi.bbox, bbox_gt) for roi in bundle.rois_for_audit())
        best_roi = bundle.candidates[0] if bundle.candidates else bundle.fallback
        best_roi_volume = int(
            max(
                (best_roi.z1 - best_roi.z0)
                * (best_roi.y1 - best_roi.y0)
                * (best_roi.x1 - best_roi.x0),
                1,
            )
        )
        filas.append(
            {
                "case_id": caso.case_id,
                "voxeles_gt": voxels_gt,
                "is_target_case": es_caso_objetivo(voxels_gt),
                "n_candidates": len(bundle.candidates),
                "uses_fallback_only": int(len(bundle.candidates) == 0),
                "bbox_gt_covered": int(covered),
                "best_candidate_score": round(
                    float(bundle.candidates[0].score) if bundle.candidates else 0.0,
                    4,
                ),
                "best_candidate_volume_voxels": best_roi_volume,
                "best_candidate_gt_volume_ratio": round(
                    float(best_roi_volume / max(voxels_gt, 1)),
                    2,
                ),
            }
        )

    df = pd.DataFrame(filas)
    target_df = df[df["is_target_case"]] if not df.empty else df
    summary = {
        "n_cases": len(df),
        "n_target_cases": len(target_df),
        "coverage_all_cases": round(
            float(df["bbox_gt_covered"].mean()) if not df.empty else 0.0, 4
        ),
        "coverage_target_cases": round(
            float(target_df["bbox_gt_covered"].mean()) if not target_df.empty else 0.0,
            4,
        ),
        "mean_best_candidate_gt_volume_ratio": round(
            float(df["best_candidate_gt_volume_ratio"].mean()) if not df.empty else 0.0,
            2,
        ),
        "median_best_candidate_gt_volume_ratio": round(
            float(df["best_candidate_gt_volume_ratio"].median()) if not df.empty else 0.0,
            2,
        ),
        "fallback_only_cases": int(df["uses_fallback_only"].sum()) if not df.empty else 0,
    }
    return df, summary


def _crear_gt_roi(caso: Case2D) -> ROIProposal | None:
    if caso.label is None:
        return None
    bbox = _bbox_from_mask(caso.label)
    if bbox is None:
        return None
    bbox = _expand_bbox(bbox, caso.image.shape, CASCADE_CANDIDATE_MARGIN_ZYX)
    z0, z1, y0, y1, x0, x1 = bbox
    voxels = int(np.sum(caso.label > 0))
    return ROIProposal(
        z0=z0,
        z1=z1,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        source="gt_bbox",
        score=1.0,
        size=max((z1 - z0) * (y1 - y0) * (x1 - x0), 1),
        mean_prob=1.0,
        overlap_gt_voxels=voxels,
    )


def construir_items_fine(
    casos: list[Case2D],
    propuestas_train: dict[str, CaseProposals],
) -> tuple[list[FineSliceItem], list[FineSliceItem]]:
    positivos: list[FineSliceItem] = []
    negativos: list[FineSliceItem] = []

    for case_idx, caso in enumerate(casos):
        if caso.label is None or caso.positive_slices is None:
            continue

        used_items: set[tuple[int, BBoxZYX, int, bool]] = set()

        def add_item(
            roi: ROIProposal,
            z: int,
            is_positive: bool,
            *,
            case_idx_actual: int = case_idx,
            used_items_actual: set[tuple[int, BBoxZYX, int, bool]] = used_items,
        ) -> None:
            key = (case_idx_actual, roi.bbox, int(z), bool(is_positive))
            if key in used_items_actual:
                return
            used_items_actual.add(key)
            destino = positivos if is_positive else negativos
            destino.append(
                FineSliceItem(
                    case_idx=case_idx_actual,
                    roi=roi,
                    z=int(z),
                    is_positive=is_positive,
                )
            )

        roi_gt = _crear_gt_roi(caso)
        if roi_gt is not None:
            for z in caso.positive_slices.astype(int).tolist():
                if roi_gt.z0 <= z < roi_gt.z1:
                    add_item(roi_gt, z, True)

        bundle = propuestas_train[caso.case_id]
        for roi in bundle.candidates:
            if roi.overlap_gt_voxels > 0:
                for z in caso.positive_slices.astype(int).tolist():
                    if roi.z0 <= z < roi.z1:
                        patch_gt = caso.label[z, roi.y0 : roi.y1, roi.x0 : roi.x1]
                        if np.sum(patch_gt > 0) > 0:
                            add_item(roi, z, True)
                continue
            for z in range(roi.z0, roi.z1):
                patch_gt = caso.label[z, roi.y0 : roi.y1, roi.x0 : roi.x1]
                patch_lung = caso.lung_mask[z, roi.y0 : roi.y1, roi.x0 : roi.x1]
                if np.sum(patch_gt > 0) == 0 and float(np.mean(patch_lung > 0)) > 0.01:
                    add_item(roi, z, False)

    return positivos, negativos


def crear_manifest_fine_training(
    positivos: list[FineSliceItem],
    negativos: list[FineSliceItem],
    seed: int = SEMILLA,
) -> list[FineSliceItem]:
    rng = np.random.default_rng(seed)
    manifest = list(positivos)

    if negativos:
        ratio_neg = max(1.0 - CASCADE_FINE_POSITIVE_RATIO, 0.0)
        n_neg = math.ceil(len(positivos) * ratio_neg / max(CASCADE_FINE_POSITIVE_RATIO, 1e-8))
        indices = rng.choice(
            len(negativos),
            size=max(n_neg, 1),
            replace=len(negativos) < max(n_neg, 1),
        )
        manifest.extend(negativos[int(i)] for i in indices)

    rng.shuffle(manifest)
    return manifest


def _evaluar_prob_maps(
    casos: list[Case2D],
    prob_maps: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    filas = []
    for caso in casos:
        if caso.label is None:
            continue
        prob = prob_maps[caso.case_id]
        pred = _postprocess_mask((prob >= threshold).astype(np.uint8), caso.lung_mask)
        met = evaluar_segmentacion(pred, caso.label, caso.spacing)
        met["case_id"] = caso.case_id
        filas.append(met)
    return anotar_target_cases(pd.DataFrame(filas))


def _evaluate_coarse_segmentation(
    casos: list[Case2D],
    prob_maps: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    filas = []
    for caso in casos:
        if caso.label is None:
            continue
        support = prob_maps[caso.case_id] >= COARSE_SUPPORT_THRESHOLD
        if CASCADE_COARSE_EVAL_USE_ALARM:
            support = support | (caso.soft_tissue_alarm_mask > 0)
        support = support & (caso.lung_mask > 0)
        support = ndimage.binary_closing(
            support, structure=np.ones(CASCADE_POSTPROCESS_CLOSING, dtype=bool)
        )
        support = limpiar_componentes_pequenas(
            support.astype(np.uint8), min_size=MIN_COMPONENT_SIZE, max_componentes=128
        )
        met = evaluar_segmentacion(support, caso.label, caso.spacing)
        met["case_id"] = caso.case_id
        filas.append(met)
    df = anotar_target_cases(pd.DataFrame(filas))
    return df, resumir_metricas_cascade(df)


def buscar_threshold_optimo(
    casos: list[Case2D],
    prob_maps: dict[str, np.ndarray],
    mode: str = "single",
) -> tuple[pd.DataFrame, float, pd.DataFrame, dict[str, float | int]]:
    filas: list[dict[str, object]] = []
    best_threshold = 0.5
    best_df = pd.DataFrame()
    best_summary: dict[str, float | int] = {
        "target_pass_rate": -1.0,
        "target_mean_dice": -1.0,
        "all_case_mean_dice": -1.0,
    }

    for threshold in _build_thresholds():
        df_eval = _evaluar_prob_maps(casos, prob_maps, threshold=threshold)
        summary = resumir_metricas_cascade(df_eval)
        filas.append(
            {
                "mode": mode,
                "threshold": threshold,
                "target_pass_rate": summary["target_pass_rate"],
                "target_mean_dice": summary["target_mean_dice"],
                "all_case_mean_dice": summary["all_case_mean_dice"],
                "n_target_passed": summary["n_target_passed"],
                "n_target_cases": summary["n_target_cases"],
            }
        )
        if _score_tuple(summary) > _score_tuple(best_summary):
            best_threshold = threshold
            best_df = df_eval
            best_summary = summary

    return pd.DataFrame(filas), best_threshold, best_df, best_summary


def _insert_snapshot(
    snapshots: list[dict[str, object]],
    snapshot: dict[str, object],
    top_k: int,
) -> list[dict[str, object]]:
    updated = [*snapshots, snapshot]
    updated.sort(key=lambda item: _score_tuple(item["summary"]), reverse=True)  # type: ignore[index]
    return updated[:top_k]


def _crear_scheduler(
    opt: torch.optim.Optimizer,
    num_epocas: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    warmup_epocas = min(LR_WARMUP_EPOCAS, num_epocas)
    cosine_epocas = max(num_epocas - warmup_epocas, 1)
    sched_warmup = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epocas,
    )
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_epocas)
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[sched_warmup, sched_cosine],
        milestones=[warmup_epocas],
    )


def _entrenar_modelo_generico(
    *,
    descripcion: str,
    build_loader,
    evaluate_model,
    criterion: nn.Module,
    num_epocas: int,
    patience: int,
    lr: float,
    checkpoint_paths: list[Path],
    history_extra: dict[str, object],
) -> tuple[torch.nn.Module, dict[str, object], list[dict[str, object]]]:
    modelo = crear_modelo_2d().to(DEVICE)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = _crear_scheduler(opt, num_epocas=num_epocas)
    scaler = GradScaler(DEVICE.type, enabled=AMP_ENABLED)

    history: dict[str, object] = {
        "descripcion": descripcion,
        "train_loss": [],
        "val_target_pass_rate": [],
        "val_target_mean_dice": [],
        "val_all_case_mean_dice": [],
        "n_params": contar_parametros(modelo),
        "prior_channels": PRIOR_CHANNELS_2D,
        "context_slices": CONTEXT_SLICES,
        "fast_dev_run": FAST_DEV_RUN,
        **history_extra,
    }

    top_k = max(len(checkpoint_paths), 1)
    snapshots: list[dict[str, object]] = []
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    patience_ctr = 0
    t0 = time.time()

    for epoca in range(1, num_epocas + 1):
        modelo.train()
        loader = build_loader(SEMILLA + epoca)
        train_loss = 0.0
        n_batches = 0

        for batch in loader:
            image = batch["image"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
                logits = modelo(image)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            train_loss += float(loss.item())
            n_batches += 1

        sched.step()
        avg_loss = train_loss / max(n_batches, 1)
        cast_loss = history["train_loss"]
        if isinstance(cast_loss, list):
            cast_loss.append(round(avg_loss, 4))

        if epoca % VAL_INTERVAL == 0 or epoca == num_epocas:
            df_val, summary = evaluate_model(modelo)
            cast_pass = history["val_target_pass_rate"]
            cast_target = history["val_target_mean_dice"]
            cast_all = history["val_all_case_mean_dice"]
            if isinstance(cast_pass, list):
                cast_pass.append(float(summary["target_pass_rate"]))
            if isinstance(cast_target, list):
                cast_target.append(float(summary["target_mean_dice"]))
            if isinstance(cast_all, list):
                cast_all.append(float(summary["all_case_mean_dice"]))

            score = _score_tuple(summary)
            snapshot = {
                "epoch": epoca,
                "summary": summary,
                "state": {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()},
                "df_val": df_val,
            }
            snapshots = _insert_snapshot(snapshots, snapshot, top_k=top_k)

            if score > best_score:
                best_score = score
                best_epoch = epoca
                patience_ctr = 0
            else:
                patience_ctr += 1

            lr_actual = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else lr
            print(
                f"  ep {epoca:>3d}/{num_epocas} | loss {avg_loss:.4f} | "
                f"target_pass {summary['target_pass_rate']:.4f} | "
                f"target_mean {summary['target_mean_dice']:.4f} | "
                f"all_mean {summary['all_case_mean_dice']:.4f} | "
                f"lr {lr_actual:.2e}"
            )

            if patience_ctr >= patience:
                print(f"  early stopping en epoca {epoca}")
                break

    if snapshots:
        modelo.load_state_dict(snapshots[0]["state"])  # type: ignore[arg-type]
        for idx, path in enumerate(checkpoint_paths):
            if idx < len(snapshots):
                torch.save(snapshots[idx]["state"], path)  # type: ignore[arg-type]

    history["best_epoch"] = best_epoch
    history["best_summary"] = snapshots[0]["summary"] if snapshots else {}
    history["tiempo_total_s"] = round(time.time() - t0, 1)
    return modelo, history, snapshots


def entrenar_modelo_coarse(
    train_cases: list[Case2D],
    val_cases: list[Case2D],
) -> tuple[torch.nn.Module, dict[str, object], list[dict[str, object]]]:
    criterio = RecallDrivenLoss(pos_weight=8.0).to(DEVICE)

    def build_loader(seed: int) -> DataLoader:
        manifest = crear_manifest_entrenamiento(train_cases, seed=seed)
        dataset = SliceDataset(train_cases, manifest, augment_level=2, train=True)
        return DataLoader(
            dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=N_WORKERS_DATALOADER,
            pin_memory=AMP_ENABLED,
        )

    def evaluate_model(modelo: torch.nn.Module) -> tuple[pd.DataFrame, dict[str, float | int]]:
        probs_val = inferir_probabilidades_coarse_casos(modelo, val_cases)
        return _evaluate_coarse_segmentation(val_cases, probs_val)

    return _entrenar_modelo_generico(
        descripcion=MODELO_CASCADE_COARSE_NOMBRE,
        build_loader=build_loader,
        evaluate_model=evaluate_model,
        criterion=criterio,
        num_epocas=NUM_EPOCAS_CASCADE_COARSE,
        patience=EARLY_STOPPING_PATIENCE_CASCADE_COARSE,
        lr=LR_CASCADE_COARSE,
        checkpoint_paths=[RUTA_MODELO_CASCADE_COARSE],
        history_extra={
            "loss_fn": "0.7*TverskyLoss(alpha=0.2,beta=0.8)+0.3*BCEWithLogitsLoss(pos_weight=8.0)",
            "coarse_support_threshold": COARSE_SUPPORT_THRESHOLD,
        },
    )


def entrenar_modelo_fine(
    train_cases: list[Case2D],
    val_cases: list[Case2D],
    propuestas_train: dict[str, CaseProposals],
    propuestas_val: dict[str, CaseProposals],
    coarse_probs_val: dict[str, np.ndarray] | None = None,
) -> tuple[
    torch.nn.Module,
    dict[str, object],
    list[dict[str, object]],
    list[FineSliceItem],
    list[FineSliceItem],
]:
    positivos, negativos = construir_items_fine(train_cases, propuestas_train)
    criterio = DiceCELoss(sigmoid=True, lambda_dice=0.7, lambda_ce=0.3, squared_pred=True).to(
        DEVICE
    )

    def build_loader(seed: int) -> DataLoader:
        manifest = crear_manifest_fine_training(positivos, negativos, seed=seed)
        dataset = FineSliceDataset(
            train_cases,
            manifest,
            augment_level=CASCADE_FINE_AUGMENT_LEVEL,
            train=True,
        )
        return DataLoader(
            dataset,
            batch_size=TRAIN_BATCH_SIZE_FINE,
            shuffle=True,
            num_workers=N_WORKERS_DATALOADER,
            pin_memory=AMP_ENABLED,
        )

    def evaluate_model(modelo: torch.nn.Module) -> tuple[pd.DataFrame, dict[str, float | int]]:
        probs_val = inferir_probabilidades_fine_casos(
            [modelo],
            val_cases,
            propuestas_val,
            use_tta=False,
            coarse_probs=coarse_probs_val,
        )
        df_val = _evaluar_prob_maps(val_cases, probs_val, threshold=0.5)
        return df_val, resumir_metricas_cascade(df_val)

    modelo, history, snapshots = _entrenar_modelo_generico(
        descripcion=MODELO_CASCADE_FINE_NOMBRE,
        build_loader=build_loader,
        evaluate_model=evaluate_model,
        criterion=criterio,
        num_epocas=NUM_EPOCAS_CASCADE_FINE,
        patience=EARLY_STOPPING_PATIENCE_CASCADE_FINE,
        lr=LR_CASCADE_FINE,
        checkpoint_paths=[RUTA_MODELO_CASCADE_FINE, RUTA_MODELO_CASCADE_FINE_RANK2],
        history_extra={
            "loss_fn": "DiceCELoss(lambda_dice=0.7,lambda_ce=0.3,squared_pred=True)",
            "fine_roi_size_2d": FINE_ROI_SIZE_2D,
            "positive_items": len(positivos),
            "positive_gt_roi_items": sum(1 for item in positivos if item.roi.source == "gt_bbox"),
            "positive_candidate_roi_items": sum(
                1 for item in positivos if item.roi.source != "gt_bbox"
            ),
            "negative_items": len(negativos),
            "positive_ratio_target": CASCADE_FINE_POSITIVE_RATIO,
            "augment_level": CASCADE_FINE_AUGMENT_LEVEL,
            "use_coarse_prior": CASCADE_FINE_USE_COARSE_PRIOR,
            "coarse_prior_floor": CASCADE_FINE_COARSE_PRIOR_FLOOR,
        },
    )
    return modelo, history, snapshots, positivos, negativos


def _inferir_probabilidades_fine_batch(
    modelo: torch.nn.Module,
    batch_np: np.ndarray,
    flip_h: bool = False,
    flip_v: bool = False,
) -> np.ndarray:
    if flip_h:
        batch_np = np.flip(batch_np, axis=-1).copy()
    if flip_v:
        batch_np = np.flip(batch_np, axis=-2).copy()

    batch = torch.from_numpy(batch_np.astype(np.float32)).to(DEVICE)
    with autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
        logits = modelo(batch)
    prob = torch.sigmoid(logits).cpu().numpy()[:, 0]
    if flip_h:
        prob = np.flip(prob, axis=-1).copy()
    if flip_v:
        prob = np.flip(prob, axis=-2).copy()
    return prob.astype(np.float32)


def inferir_probabilidades_fine_roi(
    modelo: torch.nn.Module,
    caso: Case2D,
    roi: ROIProposal,
    batch_size: int = INFER_BATCH_SIZE_FINE,
    use_tta: bool = False,
) -> np.ndarray:
    modelo.eval()
    roi_h = roi.y1 - roi.y0
    roi_w = roi.x1 - roi.x0
    roi_z = roi.z1 - roi.z0
    prob_roi = np.zeros((roi_z, roi_h, roi_w), dtype=np.float32)

    z_values = list(range(roi.z0, roi.z1))
    with torch.no_grad():
        for inicio in range(0, len(z_values), batch_size):
            z_batch = z_values[inicio : inicio + batch_size]
            batch_np = []
            for z in z_batch:
                full_input = construir_entrada_modelo(caso, z)
                cropped_input = _crop_hw(full_input, roi.bbox)
                resized_input = _resize_chw(cropped_input, FINE_ROI_SIZE_2D, mode="bilinear")
                batch_np.append(resized_input)

            batch_arr = np.stack(batch_np, axis=0).astype(np.float32)
            prob_small = _inferir_probabilidades_fine_batch(modelo, batch_arr)
            if use_tta:
                prob_fh = _inferir_probabilidades_fine_batch(modelo, batch_arr, flip_h=True)
                prob_fv = _inferir_probabilidades_fine_batch(modelo, batch_arr, flip_v=True)
                prob_small = (prob_small + prob_fh + prob_fv) / 3.0

            prob_small_t = prob_small[:, None].astype(np.float32)
            prob_resized = _resize_bchw(prob_small_t, (roi_h, roi_w), mode="bilinear")[:, 0]
            prob_roi[inicio : inicio + len(z_batch)] = prob_resized.astype(np.float32)

    return prob_roi


def inferir_probabilidades_fine_casos(
    modelos: list[torch.nn.Module],
    casos: list[Case2D],
    propuestas: dict[str, CaseProposals],
    use_tta: bool = False,
    coarse_probs: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    prob_maps: dict[str, np.ndarray] = {}
    for idx, caso in enumerate(casos, start=1):
        print(f"  fine inferencia {idx:>2d}/{len(casos)} -> {caso.case_id}")
        bundle = propuestas[caso.case_id]
        rois = bundle.rois_for_inference()
        full_prob = np.zeros(caso.image.shape, dtype=np.float32)

        for roi in rois:
            model_probs = []
            for modelo in modelos:
                model_probs.append(
                    inferir_probabilidades_fine_roi(modelo, caso, roi, use_tta=use_tta)
                )
            prob_roi = np.mean(model_probs, axis=0).astype(np.float32)
            full_prob[roi.z0 : roi.z1, roi.y0 : roi.y1, roi.x0 : roi.x1] = np.maximum(
                full_prob[roi.z0 : roi.z1, roi.y0 : roi.y1, roi.x0 : roi.x1],
                prob_roi,
            )

        full_prob *= caso.lung_mask.astype(np.float32)
        coarse_prob = coarse_probs.get(caso.case_id) if coarse_probs is not None else None
        full_prob = _aplicar_prior_coarse(full_prob, coarse_prob)
        prob_maps[caso.case_id] = full_prob
    return prob_maps


def _cargar_modelo_desde_state(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    modelo = crear_modelo_2d().to(DEVICE)
    modelo.load_state_dict(state)
    modelo.eval()
    return modelo


def ejecutar_pipeline_cascada(
    casos: list[Case2D],
    fold_idx: int,
) -> dict[str, object]:
    train_cases, val_cases = split_por_fold(casos, fold_idx=fold_idx)
    print(
        f"Entrenando cascada en fold={fold_idx} | train={len(train_cases)} | val={len(val_cases)}"
    )

    modelo_coarse, historia_coarse, snapshots_coarse = entrenar_modelo_coarse(
        train_cases, val_cases
    )

    print("\nGenerando mapas coarse para train...")
    coarse_probs_train = inferir_probabilidades_coarse_casos(
        modelo_coarse,
        train_cases,
        prefix="coarse train",
    )
    print("\nGenerando mapas coarse para val...")
    coarse_probs_val = inferir_probabilidades_coarse_casos(
        modelo_coarse,
        val_cases,
        prefix="coarse val",
    )

    propuestas_train = generar_propuestas_casos(train_cases, coarse_probs_train)
    propuestas_val = generar_propuestas_casos(val_cases, coarse_probs_val)

    df_proposals_train, resumen_proposals_train = auditar_propuestas(train_cases, propuestas_train)
    df_proposals_val, resumen_proposals_val = auditar_propuestas(val_cases, propuestas_val)

    historia_coarse["proposal_audit_train"] = resumen_proposals_train
    historia_coarse["proposal_audit_val"] = resumen_proposals_val

    print("\nEntrenando etapa fine ROI...")
    modelo_fine, historia_fine, snapshots_fine, fine_pos_items, fine_neg_items = (
        entrenar_modelo_fine(
            train_cases,
            val_cases,
            propuestas_train,
            propuestas_val,
            coarse_probs_val=coarse_probs_val,
        )
    )
    historia_fine["positive_items"] = len(fine_pos_items)
    historia_fine["negative_items"] = len(fine_neg_items)

    print("\nInferencia fine final (single)...")
    prob_maps_single = inferir_probabilidades_fine_casos(
        [modelo_fine],
        val_cases,
        propuestas_val,
        use_tta=False,
        coarse_probs=coarse_probs_val,
    )
    df_thresh_single, threshold_single, df_final_single, resumen_single = buscar_threshold_optimo(
        val_cases,
        prob_maps_single,
        mode="single",
    )

    threshold_df = df_thresh_single
    final_prob_maps = prob_maps_single
    final_df = df_final_single
    final_summary = resumen_single
    final_threshold = threshold_single
    selected_mode = "single"

    near_success = (
        float(resumen_single["target_pass_rate"])
        >= (TARGET_CASE_SUCCESS_RATE - TARGET_CASE_NEAR_SUCCESS_MARGIN)
        and float(resumen_single["target_pass_rate"]) < TARGET_CASE_SUCCESS_RATE
        and len(snapshots_fine) >= 2
    )

    if near_success:
        print(
            "\nObjetivo cerca de cumplirse: probando ensemble con los 2 mejores checkpoints fine..."
        )
        modelo_rank2 = _cargar_modelo_desde_state(snapshots_fine[1]["state"])  # type: ignore[arg-type]
        prob_maps_ensemble = inferir_probabilidades_fine_casos(
            [modelo_fine, modelo_rank2],
            val_cases,
            propuestas_val,
            use_tta=False,
            coarse_probs=coarse_probs_val,
        )
        df_thresh_ensemble, threshold_ensemble, df_final_ensemble, resumen_ensemble = (
            buscar_threshold_optimo(
                val_cases,
                prob_maps_ensemble,
                mode="ensemble",
            )
        )
        threshold_df = pd.concat([df_thresh_single, df_thresh_ensemble], ignore_index=True)
        if _score_tuple(resumen_ensemble) > _score_tuple(resumen_single):
            final_prob_maps = prob_maps_ensemble
            final_df = df_final_ensemble
            final_summary = resumen_ensemble
            final_threshold = threshold_ensemble
            selected_mode = "ensemble"

    final_prob_maps_tta: dict[str, np.ndarray] | None = None
    final_df_tta: pd.DataFrame | None = None
    final_summary_tta: dict[str, float | int] | None = None
    final_threshold_tta: float | None = None

    if CASCADE_ENABLE_TTA_FINAL:
        print("\nEjecutando TTA final opcional...")
        modelos_tta = [modelo_fine]
        if selected_mode == "ensemble" and len(snapshots_fine) >= 2:
            modelos_tta = [
                modelo_fine,
                _cargar_modelo_desde_state(snapshots_fine[1]["state"]),  # type: ignore[arg-type]
            ]
        final_prob_maps_tta = inferir_probabilidades_fine_casos(
            modelos_tta,
            val_cases,
            propuestas_val,
            use_tta=True,
            coarse_probs=coarse_probs_val,
        )
        _, final_threshold_tta, final_df_tta, final_summary_tta = buscar_threshold_optimo(
            val_cases,
            final_prob_maps_tta,
            mode=f"{selected_mode}_tta",
        )

    combined_history = {
        "coarse": historia_coarse,
        "fine": historia_fine,
        "selected_mode": selected_mode,
        "target_case_min_voxels": TARGET_CASE_MIN_VOXELS,
        "target_case_dice_threshold": TARGET_CASE_DICE_THRESHOLD,
        "target_case_success_rate": TARGET_CASE_SUCCESS_RATE,
        "threshold_range": {
            "min": CASCADE_THRESHOLD_MIN,
            "max": CASCADE_THRESHOLD_MAX,
            "step": CASCADE_THRESHOLD_STEP,
        },
        "fine_use_coarse_prior": CASCADE_FINE_USE_COARSE_PRIOR,
        "fine_coarse_prior_floor": CASCADE_FINE_COARSE_PRIOR_FLOOR,
        "coarse_eval_use_alarm": CASCADE_COARSE_EVAL_USE_ALARM,
        "train_case_ids": [caso.case_id for caso in train_cases],
        "val_case_ids": [caso.case_id for caso in val_cases],
    }

    return {
        "train_cases": train_cases,
        "val_cases": val_cases,
        "coarse_history": historia_coarse,
        "fine_history": historia_fine,
        "combined_history": combined_history,
        "coarse_prob_maps_train": coarse_probs_train,
        "coarse_prob_maps_val": coarse_probs_val,
        "proposals_train": propuestas_train,
        "proposals_val": propuestas_val,
        "proposal_df_train": df_proposals_train,
        "proposal_df_val": df_proposals_val,
        "proposal_summary_train": resumen_proposals_train,
        "proposal_summary_val": resumen_proposals_val,
        "threshold_df": threshold_df,
        "final_threshold": final_threshold,
        "final_df": final_df.sort_values("dice", ascending=False).reset_index(drop=True),
        "final_summary": final_summary,
        "final_prob_maps": final_prob_maps,
        "final_df_target": final_df[final_df["is_target_case"]]
        .sort_values("dice", ascending=False)
        .reset_index(drop=True),
        "selected_mode": selected_mode,
        "snapshots_coarse": snapshots_coarse,
        "snapshots_fine": snapshots_fine,
        "final_prob_maps_tta": final_prob_maps_tta,
        "final_df_tta": final_df_tta.sort_values("dice", ascending=False).reset_index(drop=True)
        if final_df_tta is not None
        else None,
        "final_summary_tta": final_summary_tta,
        "final_threshold_tta": final_threshold_tta,
    }

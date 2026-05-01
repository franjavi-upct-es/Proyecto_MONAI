"""Envoltorio para inferencia por ventana deslizante optimizado para modelos SSL/Hybrid."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from monai.inferers import sliding_window_inference
from omegaconf import DictConfig


def _main_output(output: torch.Tensor | tuple | list | dict) -> torch.Tensor:
    """Extrae la salida principal para modelos con varias cabezas."""
    if isinstance(output, (tuple, list)):
        return output[0]
    if isinstance(output, dict):
        return output.get("logits", next(iter(output.values())))
    return output


def _is_slice_wise_25d(model: torch.nn.Module, cfg: DictConfig) -> bool:
    model_cfg = cfg.get("model", {})
    configured_name = str(model_cfg.get("name", "")).lower() if model_cfg else ""
    return configured_name == "vit_25d" or model.__class__.__name__ == "ViT25D"


def _xy_overlap(value: object) -> tuple[float, float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        overlap_h = float(value[0])
        overlap_w = float(value[1] if len(value) > 1 else value[0])
    else:
        overlap_h = overlap_w = float(value)
    return (overlap_h, overlap_w, 0.0)


def _encoder_patch_size_xy(model: torch.nn.Module) -> tuple[int, int]:
    encoder = getattr(model, "encoder", None)
    encoder_model = getattr(encoder, "model", None)
    patch_embed = getattr(encoder_model, "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", 1)
    if isinstance(patch_size, int):
        return (int(patch_size), int(patch_size))
    return (int(patch_size[0]), int(patch_size[1]))


def _predict_slice_wise_25d(
    model: torch.nn.Module,
    image: torch.Tensor,
    cfg: DictConfig,
    inference_cfg: DictConfig,
) -> torch.Tensor:
    roi_size = inference_cfg.get("roi_size", cfg.training.patch_size)
    patch_h, patch_w, *_ = (int(v) for v in roi_size)
    spatial_h, spatial_w, spatial_d = (int(v) for v in image.shape[-3:])
    encoder_patch_h, encoder_patch_w = _encoder_patch_size_xy(model)
    if patch_h % encoder_patch_h != 0 or patch_w % encoder_patch_w != 0:
        raise ValueError(
            "training.inference.roi_size debe ser divisible por el patch size del encoder"
        )

    can_run_full = (
        spatial_h <= patch_h
        and spatial_w <= patch_w
        and spatial_h % encoder_patch_h == 0
        and spatial_w % encoder_patch_w == 0
    )
    if can_run_full:
        return _main_output(model(image))

    def _predictor(window: torch.Tensor) -> torch.Tensor:
        return _main_output(model(window))

    return sliding_window_inference(
        inputs=image,
        roi_size=(patch_h, patch_w, spatial_d),
        sw_batch_size=int(inference_cfg.get("sw_batch_size", 1)),
        predictor=_predictor,
        overlap=_xy_overlap(inference_cfg.get("overlap", 0.25)),
        mode=str(inference_cfg.get("mode", "gaussian")),
        padding_mode=str(inference_cfg.get("padding_mode", "constant")),
    )


def predict_volume(model: torch.nn.Module, image: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """Ejecuta la inferencia por ventana deslizante de MONAI.

    Se asegura de extraer la salida principal en caso de modelos que devuelvan tuplas,
    manteniendo la compatibilidad con la arquitectura de segmentación de producción.
    """

    def _predictor(window: torch.Tensor) -> torch.Tensor:
        return _main_output(model(window))

    inference_cfg = cfg.training.get("inference", {})
    strategy = str(inference_cfg.get("strategy", "auto")).lower()
    if strategy == "auto" and _is_slice_wise_25d(model, cfg):
        # ViT25D procesa cada corte axial como una imagen 2D independiente. Deslizamos
        # solo en XY para mantener entradas compatibles con patch16 sin repetir Z.
        return _predict_slice_wise_25d(model, image, cfg, inference_cfg)
    if strategy in {"slice_full", "full_slice", "slicewise_full"}:
        return _main_output(model(image))
    if strategy not in {"auto", "sliding_window", "sliding"}:
        raise ValueError(
            "training.inference.strategy debe ser 'auto', 'slice_full' o 'sliding_window'"
        )

    return sliding_window_inference(
        inputs=image,
        roi_size=tuple(int(v) for v in cfg.training.patch_size),
        sw_batch_size=int(inference_cfg.get("sw_batch_size", 1)),
        predictor=_predictor,
        overlap=float(inference_cfg.get("overlap", 0.25)),
        mode=str(inference_cfg.get("mode", "gaussian")),
        padding_mode=str(inference_cfg.get("padding_mode", "constant")),
    )

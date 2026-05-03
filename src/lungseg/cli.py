"""Punto de entrada de la CLI de Typer para entrenamiento, inferencia, ablación y clasificación."""

from __future__ import annotations

import os

# Optimizar gestión de memoria CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import nibabel as nib
import numpy as np
import torch
import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from lungseg.ablation import analyze, run_cell
from lungseg.classification import evaluate_pipeline, evaluate_size_only
from lungseg.data import build_loaders, build_val_transforms, compute_lung_mask
from lungseg.inference import predict_volume
from lungseg.models import build_model
from lungseg.radiomics import build_radiomic_dataset
from lungseg.training import train_iters
from lungseg.utils.logging import get_logger

LOGGER = get_logger(__name__)
app = typer.Typer(
    name="lungseg",
    help="Segmentación y clasificación de tumores de pulmón en MSD Task06_Lung.",
    no_args_is_help=True,
    add_completion=False,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_overrides(overrides: list[str] | None) -> list[str]:
    normalized = []
    append_keys = {"data_fraction", "aug_regime"}
    for item in overrides or []:
        key = item.split("=", 1)[0]
        if key in append_keys:
            normalized.append("+" + item)
        else:
            normalized.append(item)
    return normalized


def _compose_config(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    config_dir = str((_repo_root() / "configs").resolve())
    normalized = _normalize_overrides(overrides)
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=normalized)
    return _prepare_output_dir(cfg, config_name=config_name, overrides=normalized)


def _raw_select(cfg: DictConfig, key: str, default=None):
    value = OmegaConf.to_container(cfg, resolve=False)
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _prepare_output_dir(cfg: DictConfig, config_name: str, overrides: list[str]) -> DictConfig:
    output_value = _raw_select(cfg, "paths.outputs", default=None)
    if output_value is None or "${hydra:" in str(output_value):
        now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        output_dir = _repo_root() / "outputs" / now
        OmegaConf.update(cfg, "paths.outputs", str(output_dir), force_add=True)
    else:
        output_dir = Path(str(output_value))
        if not output_dir.is_absolute():
            output_dir = _repo_root() / output_dir
        OmegaConf.update(cfg, "paths.outputs", str(output_dir), force_add=True)

    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml", resolve=False)
    (hydra_dir / "overrides.yaml").write_text(
        "\n".join([f"- {item}" for item in overrides]) + ("\n" if overrides else ""),
        encoding="utf-8",
    )
    (hydra_dir / "config_name.txt").write_text(config_name + "\n", encoding="utf-8")
    return cfg


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@app.command()
def train(
    config_name: Annotated[str, typer.Option("--config-name")] = "config",
    overrides: Annotated[list[str] | None, typer.Argument()] = None,
) -> None:
    """Entrena un modelo de segmentación con el entrenador por épocas SSL."""
    cfg = _compose_config(config_name, overrides)
    model = build_model(cfg)
    loaders = build_loaders(cfg, fold=int(cfg.fold), repo_root=_repo_root())
    summary = train_iters(cfg, model, loaders)
    LOGGER.info(
        "training complete: best Dice=%s checkpoint=%s",
        summary["best_val_dice"],
        summary["checkpoint_path"],
    )


@app.command()
def predict(
    checkpoint: Annotated[str, typer.Option("--checkpoint")],
    image: Annotated[str, typer.Option("--image")],
    label: Annotated[str | None, typer.Option("--label")] = None,
    output: Annotated[str, typer.Option("--output")] = "prediction.nii.gz",
    visualize: Annotated[bool, typer.Option("--visualize")] = False,
    config_name: Annotated[str, typer.Option("--config-name")] = "config",
    overrides: Annotated[list[str] | None, typer.Argument()] = None,
) -> None:
    """Ejecuta inferencia de ventana deslizante en un volumen NIfTI."""
    cfg = _compose_config(config_name, overrides)
    payload = torch.load(checkpoint, map_location="cpu")
    if "cfg" in payload:
        cfg = _prepare_output_dir(
            OmegaConf.create(payload["cfg"]), config_name=config_name, overrides=overrides or []
        )
    model = build_model(cfg)
    # Checkpoints parciales (sólo entrenables) requieren strict=False; los
    # pesos restantes vienen ya inicializados desde timm en build_model.
    is_partial = bool(payload.get("model_state_is_partial", False))
    model.load_state_dict(payload["model_state_dict"], strict=not is_partial)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    transform = build_val_transforms(cfg, with_label=False)
    sample = transform({"image": image})
    tensor = sample["image"].unsqueeze(0).to(device)

    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        logits = predict_volume(model, tensor, cfg)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype("uint8")

    affine = nib.load(image).affine
    nib.save(nib.Nifti1Image(pred, affine), output)
    LOGGER.info("prediction saved to %s", output)

    if visualize:
        from lungseg.utils.visualization import save_segmentation_mosaic, save_triplanar_prediction

        img_np = tensor.squeeze().cpu().numpy()
        lab_np = nib.load(label).get_fdata() if label else np.zeros_like(pred)

        vis_dir = Path(output).parent / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        save_triplanar_prediction(img_np, lab_np, pred, vis_dir / "triplanar.png")
        save_segmentation_mosaic(img_np, pred, vis_dir / "mosaic.png")
        LOGGER.info("Visualizations saved to %s", vis_dir)


@app.command()
def ablate(
    config_name: Annotated[str, typer.Option("--config-name")] = "config",
    overrides: Annotated[list[str] | None, typer.Argument()] = None,
) -> None:
    """Ejecuta una celda de ablación y actualiza el informe agregado."""
    effective_overrides = ["experiment=phase6_ablation", *(overrides or [])]
    cfg = _compose_config(config_name, effective_overrides)
    result = run_cell(cfg)
    report = analyze(Path(str(cfg.paths.outputs)))
    if result.get("skipped"):
        LOGGER.info("ablation cell SKIPPED (already complete): %s", result["result_path"])
    else:
        LOGGER.info("ablation cell complete: %s", result["result_path"])
    LOGGER.info("ablation report: %s", report)


@app.command("precompute-lung-masks")
def precompute_lung_masks(
    config_name: Annotated[str, typer.Option("--config-name")] = "config",
    overrides: Annotated[list[str] | None, typer.Argument()] = None,
) -> None:
    """Pre-computa y cachea máscaras pulmonares (lungmask R231) para todos los splits.

    Idempotente: si la máscara cacheada existe, no la recomputa. Recorre los
    `n_folds` splits y procesa la unión de imágenes train+val.
    """
    cfg = _compose_config(config_name, overrides)
    repo_root = _repo_root()

    cache_dir_raw = "data/cache/lung_masks"
    if "data" in cfg and "lung_mask" in cfg.data:
        cache_dir_raw = str(cfg.data.lung_mask.get("cache_dir", cache_dir_raw))
    cache_dir = Path(cache_dir_raw)
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = repo_root / str(cfg.paths.splits)
    n_folds = int(cfg.data.get("n_folds", 5))

    seen: dict[str, Path] = {}
    for fold in range(n_folds):
        fold_path = splits_dir / f"fold_{fold}.json"
        if not fold_path.exists():
            LOGGER.warning("split missing, skipping: %s", fold_path)
            continue
        payload = json.loads(fold_path.read_text(encoding="utf-8"))
        for record in [*payload.get("train", []), *payload.get("val", [])]:
            image_rel = record["image"]
            if image_rel in seen:
                continue
            image_abs = repo_root / image_rel
            seen[image_rel] = image_abs

    LOGGER.info("computing lung masks for %d unique volumes -> %s", len(seen), cache_dir)
    for idx, (rel, image_abs) in enumerate(sorted(seen.items()), start=1):
        cache_path = compute_lung_mask(image_abs, cache_dir)
        LOGGER.info("[%d/%d] %s -> %s", idx, len(seen), rel, cache_path)


@app.command()
def classify(
    config_name: Annotated[str, typer.Option("--config-name")] = "config",
    e2e: Annotated[bool, typer.Option("--e2e")] = False,
    overrides: Annotated[list[str] | None, typer.Argument()] = None,
) -> None:
    """Ejecuta la clasificación de radiómica LIDC de la Fase 5."""
    effective_overrides = ["data=lidc", *(overrides or [])]
    cfg = _compose_config(config_name, effective_overrides)
    dataset = build_radiomic_dataset(cfg, e2e=e2e)
    full = evaluate_pipeline(dataset["X"], dataset["y"], dataset["groups"], cfg)
    volumes = dataset["table"].get("original_shape_VoxelVolume")
    if volumes is None:
        volume_candidates = dataset["table"].filter(like="VoxelVolume")
        if volume_candidates.empty:
            raise ValueError(
                "la línea base de solo tamaño requiere una característica VoxelVolume de PyRadiomics"
            )
        volumes = volume_candidates.iloc[:, 0]
    size_only = evaluate_size_only(volumes.to_numpy(), dataset["y"], dataset["groups"])
    result = {"full": full, "size_only": size_only, "n_nodules": len(dataset["y"])}
    out_path = Path(str(cfg.paths.outputs)) / "classification_results.json"
    _save_json(out_path, result)
    LOGGER.info("resultados de clasificación guardados en %s", out_path)


if __name__ == "__main__":
    app()

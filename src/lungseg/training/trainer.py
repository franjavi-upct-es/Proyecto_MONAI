"""Entrenador basado en iteraciones (sin épocas)."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler

from lungseg.inference import predict_volume
from lungseg.training.losses import build_loss, deep_supervision_loss
from lungseg.training.schedulers import build_poly_scheduler
from lungseg.utils.logging import get_logger, wandb_enabled
from lungseg.utils.metrics import compute_segmentation_metrics
from lungseg.utils.seeds import set_global_determinism

LOGGER = get_logger(__name__)


def _select(cfg: DictConfig, key: str, default: Any = None) -> Any:
    try:
        value = OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default
    return value


def _int_knob(cfg: DictConfig, key: str, default: int) -> int:
    return int(_select(cfg, key, default))


def _outputs_dir(cfg: DictConfig) -> Path:
    value = _select(cfg, "paths.outputs", None)
    if value is None or "${hydra:" in str(value):
        value = "outputs/manual"
    path = Path(str(value))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _model_output_for_loss(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    return image, label


def _make_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.training.optimizer
    name = str(opt_cfg.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"unknown optimizer.name={name!r}; only 'adamw' is supported")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1.0e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1.0e-5)),
    )


def _make_train_iter(train_loader: Iterable, cfg: DictConfig) -> Iterable:
    if bool(_select(cfg, "training.sanity.overfit_one_batch", False)):
        first_batch = next(iter(train_loader))
        return itertools.repeat(first_batch)
    return itertools.cycle(train_loader)


@torch.no_grad()
def _validate(
    cfg: DictConfig,
    model: torch.nn.Module,
    val_loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    dice_values: list[float] = []
    hd95_values: list[float] = []
    spacing = _select(cfg, "data.target_spacing", None)

    for batch in val_loader:
        image, label = _batch_to_device(batch, device)
        logits = predict_volume(model, image, cfg)
        metrics = compute_segmentation_metrics(logits, label, spacing=spacing)
        dice_values.append(metrics["dice"])
        if math.isfinite(metrics["hd95"]):
            hd95_values.append(metrics["hd95"])

    model.train()
    return {
        "val_dice": float(sum(dice_values) / max(len(dice_values), 1)),
        "val_hd95": float(sum(hd95_values) / len(hd95_values)) if hd95_values else float("nan"),
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _save_checkpoint(
    path: Path,
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "step": step,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=False),
        },
        path,
    )


def train_iters(cfg: DictConfig, model, loaders) -> dict:
    """Entrena durante un número fijo de iteraciones del optimizador.

    ``global_step`` cuenta los pasos del optimizador después de la acumulación de gradientes, no
    las épocas del cargador de datos. Por lo tanto, la validación y la detención temprana son independientes
    del tamaño del conjunto de datos.
    """
    set_global_determinism(int(_select(cfg, "seed", 42)))

    train_loader, val_loader = loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    max_iterations = _int_knob(
        cfg,
        "training.sanity.max_iterations",
        _int_knob(cfg, "experiment.max_iterations", 50_000),
    )
    val_every = _int_knob(
        cfg,
        "training.sanity.val_every",
        _int_knob(cfg, "experiment.val_every", 500),
    )
    log_every = _int_knob(cfg, "experiment.log_every", 50)
    grad_accum_steps = max(_int_knob(cfg, "experiment.grad_accum_steps", 1), 1)
    patience = _int_knob(cfg, "experiment.patience", 20)
    fixed_iterations = bool(_select(cfg, "experiment.fixed_iterations", False))

    optimizer = _make_optimizer(cfg, model)
    scheduler = build_poly_scheduler(
        optimizer,
        max_steps=max_iterations,
        exp=float(cfg.training.scheduler.get("exp", 0.9)),
    )
    base_loss = build_loss(cfg)
    amp_enabled = bool(cfg.training.get("amp", False)) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    out_dir = _outputs_dir(cfg)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    summary_path = out_dir / "summary.json"
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"

    wandb_run = None
    if wandb_enabled():
        try:
            import wandb

            wandb_run = wandb.init(
                project="lungseg",
                config=OmegaConf.to_container(cfg, resolve=False),
                dir=str(out_dir),
            )
        except Exception as exc:
            LOGGER.warning("W&B requested but could not be initialized: %s", exc)

    train_iter = iter(_make_train_iter(train_loader, cfg))
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    best_dice = -1.0
    best_hd95 = float("nan")
    best_step = 0
    validations_without_improvement = 0
    rows: list[dict[str, float | int]] = []
    last_metrics = {"val_dice": float("nan"), "val_hd95": float("nan")}

    while global_step < max_iterations:
        batch = next(train_iter)
        image, label = _batch_to_device(batch, device)
        with _autocast_context(device, amp_enabled):
            output = _model_output_for_loss(model(image))
            loss = deep_supervision_loss(output, label, base_loss) / grad_accum_steps

        scaler.scale(loss).backward()
        running_loss += float(loss.detach().cpu()) * grad_accum_steps
        micro_step += 1

        if micro_step % grad_accum_steps != 0:
            continue

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1

        lr = float(optimizer.param_groups[0]["lr"])
        mean_train_loss = running_loss / grad_accum_steps
        running_loss = 0.0

        if global_step % log_every == 0 or global_step == 1:
            LOGGER.info("step=%d/%d loss=%.4f lr=%.6g", global_step, max_iterations, mean_train_loss, lr)
            if wandb_run is not None:
                wandb_run.log({"train/loss": mean_train_loss, "lr": lr, "step": global_step})

        should_validate = global_step % val_every == 0 or global_step == max_iterations
        if not should_validate:
            continue

        last_metrics = _validate(cfg, model, val_loader, device)
        row = {
            "step": global_step,
            "train_loss": mean_train_loss,
            "lr": lr,
            "val_dice": last_metrics["val_dice"],
            "val_hd95": last_metrics["val_hd95"],
        }
        rows.append(row)
        _write_metrics_csv(metrics_path, rows)
        LOGGER.info(
            "val step=%d dice=%.4f hd95=%s",
            global_step,
            last_metrics["val_dice"],
            f"{last_metrics['val_hd95']:.3f}" if math.isfinite(last_metrics["val_hd95"]) else "nan",
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "val/dice": last_metrics["val_dice"],
                    "val/hd95": last_metrics["val_hd95"],
                    "step": global_step,
                }
            )

        if last_metrics["val_dice"] > best_dice:
            best_dice = last_metrics["val_dice"]
            best_hd95 = last_metrics["val_hd95"]
            best_step = global_step
            validations_without_improvement = 0
            _save_checkpoint(best_ckpt, cfg, model, optimizer, global_step, last_metrics)
        else:
            validations_without_improvement += 1

        _save_checkpoint(last_ckpt, cfg, model, optimizer, global_step, last_metrics)
        if not fixed_iterations and validations_without_improvement >= patience:
            LOGGER.info("early stopping at step=%d after %d stale validations", global_step, patience)
            break

    if not rows:
        last_metrics = _validate(cfg, model, val_loader, device)
        rows.append(
            {
                "step": global_step,
                "train_loss": float("nan"),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_dice": last_metrics["val_dice"],
                "val_hd95": last_metrics["val_hd95"],
            }
        )
        _write_metrics_csv(metrics_path, rows)
        if last_metrics["val_dice"] > best_dice:
            best_dice = last_metrics["val_dice"]
            best_hd95 = last_metrics["val_hd95"]
            best_step = global_step
            _save_checkpoint(best_ckpt, cfg, model, optimizer, global_step, last_metrics)

    _save_checkpoint(last_ckpt, cfg, model, optimizer, global_step, last_metrics)
    summary = {
        "best_step": int(best_step),
        "best_val_dice": _finite_or_none(best_dice),
        "best_val_hd95": _finite_or_none(best_hd95),
        "last_step": int(global_step),
        "checkpoint_path": str(best_ckpt),
        "last_checkpoint_path": str(last_ckpt),
        "metrics_path": str(metrics_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if wandb_run is not None:
        wandb_run.finish()
    return summary

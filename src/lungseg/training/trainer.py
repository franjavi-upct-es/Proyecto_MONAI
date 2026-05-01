# src/lungseg/training/trainer.py
"""Entrenador principal epoch-based con soporte para estrategias de Freeze/Unfreeze SSL."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler

from lungseg.inference import predict_volume
from lungseg.training.losses import build_loss
from lungseg.training.schedulers import build_cosine_warmup_scheduler, build_poly_scheduler
from lungseg.utils.logging import get_logger, wandb_enabled
from lungseg.utils.metrics import compute_segmentation_metrics
from lungseg.utils.seeds import set_global_determinism

LOGGER = get_logger(__name__)


def _select(cfg: DictConfig, key: str, default: Any = None) -> Any:
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def _int_select(cfg: DictConfig, key: str, default: int) -> int:
    return int(_select(cfg, key, default))


def _float_select(cfg: DictConfig, key: str, default: float) -> float:
    return float(_select(cfg, key, default))


def _outputs_dir(cfg: DictConfig) -> Path:
    value = _select(cfg, "paths.outputs", None)
    if value is None or "${hydra:" in str(value):
        value = "outputs/manual"
    path = Path(str(value))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        # Priorizar bfloat16 en GPUs Blackwell/Ada si está habilitado
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _model_output_for_loss(output: torch.Tensor | tuple | list) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output


def _batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        batch["image"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


def _make_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.training.optimizer
    name = str(opt_cfg.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"unknown optimizer.name={name!r}; only 'adamw' is supported")

    # Usar versión 'fused' para mayor velocidad en GPUs modernas
    use_fused = torch.cuda.is_available() and hasattr(torch.optim.AdamW, "fused")

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1.0e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1.0e-5)),
        fused=use_fused,
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _loader_len(loader: Iterable) -> int:
    try:
        return max(len(loader), 1)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            "el entrenamiento por épocas requiere un train_loader con len() definido"
        ) from exc


def _resolve_resume_path(raw_path: str, out_dir: Path) -> Path:
    if raw_path.lower() in {"last", "best"}:
        return out_dir / "checkpoints" / f"{raw_path.lower()}.pt"
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _write_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_metrics_csv(path: Path, max_step: int | None = None) -> list[dict[str, float | int]]:
    if not path.exists():
        return []

    rows: list[dict[str, float | int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            step = int(float(row["step"]))
            if max_step is not None and step > max_step:
                continue
            rows.append(
                {
                    "epoch": int(float(row.get("epoch", 0))),
                    "step": step,
                    "train_loss": float(row["train_loss"]),
                    "lr": float(row["lr"]),
                    "val_dice": float(row["val_dice"]),
                    "val_hd95": float(row["val_hd95"]),
                }
            )
    return rows


def _best_from_rows(rows: list[dict[str, float | int]]) -> tuple[int, int, float, float]:
    best_epoch = 0
    best_step = 0
    best_dice = -1.0
    best_hd95 = float("nan")

    for row in rows:
        dice = float(row["val_dice"])
        if dice > best_dice:
            best_epoch = int(row["epoch"])
            best_step = int(row["step"])
            best_dice = dice
            best_hd95 = float(row["val_hd95"])

    return best_epoch, best_step, best_dice, best_hd95


@torch.no_grad()
def _validate(
    cfg: DictConfig,
    model: torch.nn.Module,
    val_loader: Iterable,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    spacing = _select(cfg, "data.target_spacing", None)
    dice_values: list[float] = []
    hd95_values: list[float] = []

    for batch in val_loader:
        image, label = _batch_to_device(batch, device)
        with _autocast_context(device, bool(_select(cfg, "training.amp", False))):
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


def _save_checkpoint(
    path: Path,
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    *,
    epoch: int,
    step: int,
    metrics: dict[str, float],
    best_epoch: int,
    best_step: int,
    best_metrics: dict[str, float],
    epochs_without_improvement: int,
    unfreeze_done: bool,
) -> None:
    torch.save(
        {
            "epoch": int(epoch),
            "step": int(step),
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_epoch": int(best_epoch),
            "best_step": int(best_step),
            "best_metrics": best_metrics,
            "epochs_without_improvement": int(epochs_without_improvement),
            "unfreeze_done": bool(unfreeze_done),
            "cfg": OmegaConf.to_container(cfg, resolve=False),
        },
        path,
    )


class Trainer:
    """Entrenador principal epoch-based para fine-tuning de encoders SSL."""

    _ENCODER_NAME_MARKERS = (
        "encoder",
        "patch_embed",
        "swinvit",
        "swin_vit",
        "layers1",
        "layers2",
        "layers3",
        "layers4",
        "convinit",
        "down_layers",
        "downsample",
        "bottleneck",
    )

    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        loaders: tuple[Iterable, Iterable],
        *,
        unfreeze_epoch: int = -1,
        unfreeze_lr_factor: float = 1.0,
    ) -> None:
        set_global_determinism(int(_select(cfg, "seed", 42)))

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False  # Desactivado para ahorrar memoria en 8GB GPU

        self.cfg = cfg
        self.train_loader, self.val_loader = loaders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.train()

        self.grad_accum_steps = max(_int_select(cfg, "experiment.grad_accum_steps", 4), 1)
        self.steps_per_epoch = max(
            math.ceil(_loader_len(self.train_loader) / self.grad_accum_steps), 1
        )
        self.max_epochs = self._resolve_max_epochs()
        self.max_steps = self.max_epochs * self.steps_per_epoch
        self.val_every_epochs = max(
            _int_select(
                cfg, "training.val_every_epochs", _int_select(cfg, "experiment.val_every_epochs", 1)
            ),
            1,
        )
        self.val_every_steps = _int_select(cfg, "experiment.val_every", 0)
        self.log_every_steps = max(_int_select(cfg, "experiment.log_every", 50), 1)
        self.patience_epochs = max(_int_select(cfg, "experiment.patience", 20), 1)
        self.fixed_epochs = bool(
            _select(
                cfg, "experiment.fixed_epochs", _select(cfg, "experiment.fixed_iterations", False)
            )
        )

        self.unfreeze_epoch = int(unfreeze_epoch)
        self.unfreeze_lr_factor = float(unfreeze_lr_factor)
        if self.unfreeze_lr_factor <= 0.0:
            raise ValueError("unfreeze_lr_factor debe ser > 0")
        self._unfreeze_done = self.unfreeze_epoch < 0

        if self.unfreeze_epoch > 0:
            self._freeze_encoder()

        self.optimizer = _make_optimizer(cfg, self.model)

        scheduler_name = str(cfg.training.scheduler.get("name", "poly")).lower()
        if scheduler_name == "cosine_warmup":
            self.scheduler = build_cosine_warmup_scheduler(
                self.optimizer,
                max_steps=self.max_steps,
                warmup_steps=int(cfg.training.scheduler.get("warmup_steps", 500)),
            )
        else:
            self.scheduler = build_poly_scheduler(
                self.optimizer,
                max_steps=self.max_steps,
                exp=float(cfg.training.scheduler.get("exp", 0.9)),
            )

        self.base_loss = build_loss(cfg)
        self.amp_enabled = bool(cfg.training.get("amp", False)) and self.device.type == "cuda"

        # bf16 no necesita GradScaler porque tiene el mismo rango que fp32
        self.use_bf16 = self.amp_enabled and torch.cuda.is_bf16_supported()
        self.scaler = GradScaler("cuda", enabled=(self.amp_enabled and not self.use_bf16))

        self.out_dir = _outputs_dir(cfg)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.csv"
        self.summary_path = self.out_dir / "summary.json"
        self.best_ckpt = self.ckpt_dir / "best.pt"
        self.last_ckpt = self.ckpt_dir / "last.pt"
        self.wandb_run = self._init_wandb()

    def _resolve_max_epochs(self) -> int:
        configured = _select(
            self.cfg,
            "training.max_epochs",
            _select(self.cfg, "experiment.max_epochs", None),
        )
        if configured is not None:
            return max(int(configured), 1)

        max_iterations = _select(
            self.cfg,
            "training.sanity.max_iterations",
            _select(self.cfg, "experiment.max_iterations", None),
        )
        if max_iterations is not None:
            return max(math.ceil(int(max_iterations) / self.steps_per_epoch), 1)
        return 100

    def _init_wandb(self):
        if not wandb_enabled():
            return None
        try:
            import wandb

            return wandb.init(
                project="lungseg",
                config=OmegaConf.to_container(self.cfg, resolve=False),
                dir=str(self.out_dir),
            )
        except Exception as exc:
            LOGGER.warning("W&B requested but could not be initialized: %s", exc)
            return None

    def _iter_train_batches(self) -> Iterator[dict[str, Any]]:
        if bool(_select(self.cfg, "training.sanity.overfit_one_batch", False)):
            yield next(iter(self.train_loader))
            return
        yield from self.train_loader

    def _epoch_batches(self) -> int:
        if bool(_select(self.cfg, "training.sanity.overfit_one_batch", False)):
            return 1
        return _loader_len(self.train_loader)

    def _is_encoder_parameter(self, name: str) -> bool:
        lowered = name.lower()
        return any(marker in lowered for marker in self._ENCODER_NAME_MARKERS)

    def _freeze_encoder(self) -> None:
        for name, param in self.model.named_parameters():
            if self._is_encoder_parameter(name):
                param.requires_grad = False

    def _unfreeze_model(self) -> None:
        if self._unfreeze_done:
            return

        for param in self.model.parameters():
            param.requires_grad = True

        for idx, group in enumerate(self.optimizer.param_groups):
            group["lr"] = float(group["lr"]) * self.unfreeze_lr_factor
            if "initial_lr" in group:
                group["initial_lr"] = float(group["initial_lr"]) * self.unfreeze_lr_factor
            if hasattr(self.scheduler, "base_lrs") and idx < len(self.scheduler.base_lrs):
                self.scheduler.base_lrs[idx] = (
                    float(self.scheduler.base_lrs[idx]) * self.unfreeze_lr_factor
                )

        if hasattr(self.scheduler, "_last_lr"):
            self.scheduler._last_lr = [float(group["lr"]) for group in self.optimizer.param_groups]
        self._unfreeze_done = True

    def _optimizer_step(self) -> None:
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    def _load_resume_state(self) -> tuple[int, int, int, int, float, float, int, dict[str, float]]:
        resume_from = _select(self.cfg, "training.resume_from", None)
        if not resume_from:
            return (
                0,
                0,
                0,
                0,
                -1.0,
                float("nan"),
                0,
                {
                    "val_dice": float("nan"),
                    "val_hd95": float("nan"),
                },
            )

        resume_path = _resolve_resume_path(str(resume_from), self.out_dir)
        checkpoint = torch.load(resume_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _optimizer_state_to_device(self.optimizer, self.device)
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("step", 0))
        self._unfreeze_done = bool(checkpoint.get("unfreeze_done", self._unfreeze_done))
        if (
            not self._unfreeze_done
            and self.unfreeze_epoch >= 0
            and start_epoch > self.unfreeze_epoch
        ):
            self._unfreeze_model()
        elif self._unfreeze_done:
            for param in self.model.parameters():
                param.requires_grad = True

        rows = _read_metrics_csv(self.metrics_path, max_step=global_step)
        best_epoch, best_step, best_dice, best_hd95 = _best_from_rows(rows)
        if "best_metrics" in checkpoint:
            best_metrics = checkpoint["best_metrics"]
            best_epoch = int(checkpoint.get("best_epoch", best_epoch))
            best_step = int(checkpoint.get("best_step", best_step))
            best_dice = float(best_metrics.get("val_dice", best_dice))
            best_hd95 = float(best_metrics.get("val_hd95", best_hd95))
        last_metrics = dict(
            checkpoint.get("metrics", {"val_dice": float("nan"), "val_hd95": float("nan")})
        )
        stale_epochs = int(checkpoint.get("epochs_without_improvement", 0))
        LOGGER.info("resuming training from %s at epoch=%d", resume_path, start_epoch)
        return (
            start_epoch,
            global_step,
            best_epoch,
            best_step,
            best_dice,
            best_hd95,
            stale_epochs,
            last_metrics,
        )

    def fit(self) -> dict[str, Any]:
        (
            start_epoch,
            global_step,
            best_epoch,
            best_step,
            best_dice,
            best_hd95,
            stale_epochs,
            last_metrics,
        ) = self._load_resume_state()
        rows = _read_metrics_csv(self.metrics_path, max_step=global_step)

        for current_epoch in range(start_epoch, self.max_epochs):
            if current_epoch == self.unfreeze_epoch:
                self._unfreeze_model()

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            n_loss_terms = 0
            pending_backward = 0
            epoch_batches = self._epoch_batches()

            for batch_idx, batch in enumerate(self._iter_train_batches(), start=1):
                image, label = _batch_to_device(batch, self.device)
                with _autocast_context(self.device, self.amp_enabled):
                    output = _model_output_for_loss(self.model(image))
                    loss = self.base_loss(output, label) / self.grad_accum_steps

                self.scaler.scale(loss).backward()
                running_loss += float(loss.detach().cpu()) * self.grad_accum_steps
                n_loss_terms += 1
                pending_backward += 1

                if pending_backward < self.grad_accum_steps and batch_idx < epoch_batches:
                    continue

                self._optimizer_step()
                global_step += 1
                pending_backward = 0

                if global_step % self.log_every_steps == 0 or global_step == 1:
                    lr = float(self.optimizer.param_groups[0]["lr"])
                    mean_loss = running_loss / max(n_loss_terms, 1)
                    LOGGER.info(
                        "epoch=%d step=%d loss=%.4f lr=%.6g",
                        current_epoch,
                        global_step,
                        mean_loss,
                        lr,
                    )
                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "train/loss": mean_loss,
                                "lr": lr,
                                "epoch": current_epoch,
                                "step": global_step,
                            }
                        )

                # VALIDACIÓN POR PASOS (Super-convergencia)
                if self.val_every_steps > 0 and global_step % self.val_every_steps == 0:
                    self.model.eval()
                    mean_train_loss = running_loss / max(n_loss_terms, 1)
                    last_metrics = _validate(self.cfg, self.model, self.val_loader, self.device)
                    row = {
                        "epoch": current_epoch,
                        "step": global_step,
                        "train_loss": mean_train_loss,
                        "lr": float(self.optimizer.param_groups[0]["lr"]),
                        "val_dice": last_metrics["val_dice"],
                        "val_hd95": last_metrics["val_hd95"],
                    }
                    rows.append(row)
                    _write_metrics_csv(self.metrics_path, rows)

                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "val/dice": last_metrics["val_dice"],
                                "val/hd95": last_metrics["val_hd95"],
                                "epoch": current_epoch,
                                "step": global_step,
                            }
                        )

                    if last_metrics["val_dice"] > best_dice:
                        best_epoch, best_step, best_dice, best_hd95 = (
                            current_epoch,
                            global_step,
                            last_metrics["val_dice"],
                            last_metrics["val_hd95"],
                        )
                        stale_epochs = 0
                        _save_checkpoint(
                            self.best_ckpt,
                            self.cfg,
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            self.scaler,
                            epoch=current_epoch,
                            step=global_step,
                            metrics=last_metrics,
                            best_epoch=best_epoch,
                            best_step=best_step,
                            best_metrics={"val_dice": best_dice, "val_hd95": best_hd95},
                            epochs_without_improvement=stale_epochs,
                            unfreeze_done=self._unfreeze_done,
                        )
                    else:
                        stale_epochs += 1

                    self.model.train()

                    if not self.fixed_epochs and stale_epochs >= self.patience_epochs:
                        LOGGER.info("Early stopping agresivo en paso %d", global_step)
                        return {
                            "best_epoch": int(best_epoch),
                            "best_val_dice": float(best_dice),
                            "checkpoint_path": str(self.best_ckpt),
                        }

            is_final_epoch = current_epoch == self.max_epochs - 1
            if self.val_every_steps > 0:
                should_validate = is_final_epoch and global_step % self.val_every_steps != 0
            else:
                should_validate = (
                    (current_epoch + 1) % self.val_every_epochs == 0 or is_final_epoch
                )

            if should_validate:
                self.model.eval()
                mean_train_loss = running_loss / max(n_loss_terms, 1)
                last_metrics = _validate(self.cfg, self.model, self.val_loader, self.device)
                row = {
                    "epoch": current_epoch,
                    "step": global_step,
                    "train_loss": mean_train_loss,
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                    "val_dice": last_metrics["val_dice"],
                    "val_hd95": last_metrics["val_hd95"],
                }
                rows.append(row)
                _write_metrics_csv(self.metrics_path, rows)

                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "val/dice": last_metrics["val_dice"],
                            "val/hd95": last_metrics["val_hd95"],
                            "epoch": current_epoch,
                            "step": global_step,
                        }
                    )

                if last_metrics["val_dice"] > best_dice:
                    best_epoch, best_step, best_dice, best_hd95 = (
                        current_epoch,
                        global_step,
                        last_metrics["val_dice"],
                        last_metrics["val_hd95"],
                    )
                    stale_epochs = 0
                    _save_checkpoint(
                        self.best_ckpt,
                        self.cfg,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.scaler,
                        epoch=current_epoch,
                        step=global_step,
                        metrics=last_metrics,
                        best_epoch=best_epoch,
                        best_step=best_step,
                        best_metrics={"val_dice": best_dice, "val_hd95": best_hd95},
                        epochs_without_improvement=stale_epochs,
                        unfreeze_done=self._unfreeze_done,
                    )
                else:
                    stale_epochs += self.val_every_epochs

                self.model.train()

            _save_checkpoint(
                self.last_ckpt,
                self.cfg,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch=current_epoch,
                step=global_step,
                metrics=last_metrics,
                best_epoch=best_epoch,
                best_step=best_step,
                best_metrics={"val_dice": best_dice, "val_hd95": best_hd95},
                epochs_without_improvement=stale_epochs,
                unfreeze_done=self._unfreeze_done,
            )

            if not self.fixed_epochs and stale_epochs >= self.patience_epochs:
                LOGGER.info("early stopping at epoch=%d", current_epoch)
                break

        summary = {
            "best_epoch": int(best_epoch),
            "best_val_dice": _finite_or_none(best_dice),
            "checkpoint_path": str(self.best_ckpt),
            "last_step": int(global_step),
            "last_checkpoint_path": str(self.last_ckpt),
            "metrics_path": str(self.metrics_path),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if self.wandb_run is not None:
            self.wandb_run.finish()
        return summary

    def run(self) -> dict[str, Any]:
        return self.fit()


def train_iters(
    cfg: DictConfig, model: torch.nn.Module, loaders: tuple[Iterable, Iterable]
) -> dict[str, Any]:
    return Trainer(
        cfg,
        model,
        loaders,
        unfreeze_epoch=_int_select(cfg, "training.unfreeze_epoch", -1),
        unfreeze_lr_factor=_float_select(cfg, "training.unfreeze_lr_factor", 1.0),
    ).fit()

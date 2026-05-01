"""Fábrica de DataLoader respaldada por `data/splits/fold_{i}.json`.

Devuelve `(train_loader, val_loader)`. El modo de caché por defecto es conservador:
el Dataset simple mantiene bajo el uso de RAM, mientras que el almacenamiento en caché en RAM y disco sigue siendo opcional.
"""

from __future__ import annotations

import json
from pathlib import Path

from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset
from omegaconf import DictConfig, OmegaConf

from lungseg.data.splits import REPO_ROOT
from lungseg.data.transforms import build_train_transforms, build_val_transforms
from lungseg.utils.logging import get_logger
from lungseg.utils.seeds import seed_worker

LOGGER = get_logger(__name__)

_CACHE_MODE_ALIASES = {
    "0": "none",
    "false": "none",
    "off": "none",
    "no": "none",
    "none": "none",
    "uncached": "none",
    "auto": "auto",
    "1": "ram",
    "true": "ram",
    "cache": "ram",
    "cached": "ram",
    "memory": "ram",
    "ram": "ram",
    "persistent": "disk",
    "persistentdataset": "disk",
    "disk": "disk",
}


def _load_fold(splits_dir: Path, fold: int, repo_root: Path) -> tuple[list[dict], list[dict]]:
    fold_path = splits_dir / f"fold_{fold}.json"
    payload = json.loads(fold_path.read_text(encoding="utf-8"))

    def _absolutize(records: list[dict]) -> list[dict]:
        out = []
        for r in records:
            out.append(
                {
                    "image": str(repo_root / r["image"]),
                    "label": str(repo_root / r["label"]),
                    "patient_id": r["patient_id"],
                }
            )
        return out

    return _absolutize(payload["train"]), _absolutize(payload["val"])


def _select(cfg: DictConfig, key: str, default=None):
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def _cache_mode(cfg: DictConfig, cache_rate: float) -> str:
    raw_mode = _select(cfg, "data.cache.mode", "auto")
    mode = _CACHE_MODE_ALIASES.get(str(raw_mode).lower().replace("_", "").replace("-", ""))
    if mode is None:
        raise ValueError(
            f"unknown data.cache.mode={raw_mode!r}; expected one of: auto, none, ram, disk"
        )
    if mode == "auto":
        return "ram" if cache_rate > 0.0 else "none"
    return mode


def _resolve_cache_dir(cfg: DictConfig, repo_root: Path, fold: int) -> Path:
    raw_dir = _select(cfg, "data.cache.disk_dir", "data/cache/monai")
    cache_dir = Path(str(raw_dir))
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
    return cache_dir / f"fold_{fold}"


def _plain_datasets(
    train_files: list[dict],
    val_files: list[dict],
    train_tf,
    val_tf,
) -> tuple[Dataset, Dataset]:
    return Dataset(data=train_files, transform=train_tf), Dataset(data=val_files, transform=val_tf)


def build_loaders(
    cfg: DictConfig,
    fold: int,
    repo_root: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    repo_root = repo_root or REPO_ROOT
    splits_dir = repo_root / cfg.paths.splits if "paths" in cfg else repo_root / "data" / "splits"

    train_files, val_files = _load_fold(splits_dir, fold, repo_root)
    train_tf = build_train_transforms(cfg)
    val_tf = build_val_transforms(cfg)

    cache_rate = float(_select(cfg, "data.cache.rate", 1.0))
    cache_workers = int(_select(cfg, "data.cache.num_workers", 4))
    cache_mode = _cache_mode(cfg, cache_rate)

    if cache_mode == "none":
        LOGGER.info("Using standard Dataset (no cache)")
        train_ds, val_ds = _plain_datasets(train_files, val_files, train_tf, val_tf)
    elif cache_mode == "disk":
        # ... (mantener lógica de disk cache)
        cache_dir = _resolve_cache_dir(cfg, repo_root, fold)
        train_cache_dir = cache_dir / "train"
        val_cache_dir = cache_dir / "val"
        try:
            train_cache_dir.mkdir(parents=True, exist_ok=True)
            val_cache_dir.mkdir(parents=True, exist_ok=True)
            train_ds = PersistentDataset(data=train_files, transform=train_tf, cache_dir=train_cache_dir)
            val_ds = PersistentDataset(data=val_files, transform=val_tf, cache_dir=val_cache_dir)
            LOGGER.info("Using PersistentDataset disk cache at %s", cache_dir)
        except OSError:
            train_ds, val_ds = _plain_datasets(train_files, val_files, train_tf, val_tf)
    else:
        # CacheDataset por defecto para RAM con robustez
        try:
            LOGGER.info("Using CacheDataset in RAM (rate=%.2f, workers=%d)", cache_rate, cache_workers)
            train_ds = CacheDataset(
                data=train_files,
                transform=train_tf,
                cache_rate=cache_rate,
                num_workers=cache_workers,
                copy_cache=False
            )
            val_ds = CacheDataset(
                data=val_files,
                transform=val_tf,
                cache_rate=cache_rate,
                num_workers=cache_workers,
                copy_cache=False
            )
        except (PermissionError, RuntimeError, OSError) as exc:
            LOGGER.warning(
                "CacheDataset failed (%s); falling back to uncached Dataset.",
                exc,
            )
            train_ds, val_ds = _plain_datasets(train_files, val_files, train_tf, val_tf)

    num_workers = int(_select(cfg, "training.num_workers", 4))
    persistent_workers = bool(_select(cfg, "training.persistent_workers", False)) and num_workers > 0
    prefetch_factor = (
        int(_select(cfg, "training.prefetch_factor", 2)) if num_workers > 0 else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader

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

    cache_rate = float(_select(cfg, "data.cache.rate", 0.0) or 0.0)
    cache_workers = int(_select(cfg, "data.cache.num_workers", 0) or 0)
    cache_mode = _cache_mode(cfg, cache_rate)

    if cache_mode == "none":
        train_ds, val_ds = _plain_datasets(train_files, val_files, train_tf, val_tf)
    elif cache_mode == "disk":
        cache_dir = _resolve_cache_dir(cfg, repo_root, fold)
        train_cache_dir = cache_dir / "train"
        val_cache_dir = cache_dir / "val"
        try:
            train_cache_dir.mkdir(parents=True, exist_ok=True)
            val_cache_dir.mkdir(parents=True, exist_ok=True)
            train_ds = PersistentDataset(
                data=train_files,
                transform=train_tf,
                cache_dir=train_cache_dir,
            )
            val_ds = PersistentDataset(
                data=val_files,
                transform=val_tf,
                cache_dir=val_cache_dir,
            )
            LOGGER.info("Using PersistentDataset disk cache at %s", cache_dir)
        except OSError as exc:
            LOGGER.warning(
                "PersistentDataset cache unavailable at %s (%s); falling back to uncached Dataset.",
                cache_dir,
                exc,
            )
            train_ds, val_ds = _plain_datasets(train_files, val_files, train_tf, val_tf)
    elif cache_rate > 0.0:
        try:
            train_ds = CacheDataset(
                data=train_files,
                transform=train_tf,
                cache_rate=cache_rate,
                num_workers=cache_workers,
                copy_cache=False,
            )
            val_ds = CacheDataset(
                data=val_files,
                transform=val_tf,
                cache_rate=cache_rate,
                num_workers=cache_workers,
                copy_cache=False,
            )
            LOGGER.info("Using CacheDataset in RAM with cache_rate=%.3f", cache_rate)
        except PermissionError as exc:
            LOGGER.warning(
                "CacheDataset unavailable in this runtime (%s); falling back to uncached Dataset.",
                exc,
            )
            train_ds, val_ds = _plain_datasets(train_files, val_files, train_tf, val_tf)
    else:
        LOGGER.warning("data.cache.mode=%s with data.cache.rate <= 0; using uncached Dataset.", cache_mode)
        train_ds, val_ds = _plain_datasets(train_files, val_files, train_tf, val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=bool(cfg.training.pin_memory),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=bool(cfg.training.pin_memory),
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader

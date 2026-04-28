"""DataLoader factory backed by `data/splits/fold_{i}.json`.

Returns `(train_loader, val_loader)`. CacheDataset is used when
`cfg.data.cache.rate > 0`; otherwise plain Dataset to keep RAM low.
"""

from __future__ import annotations

import json
from pathlib import Path

from monai.data import CacheDataset, DataLoader, Dataset
from omegaconf import DictConfig

from lungseg.data.splits import REPO_ROOT
from lungseg.data.transforms import build_train_transforms, build_val_transforms
from lungseg.utils.seeds import seed_worker


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

    cache_rate = float(cfg.data.cache.rate)
    cache_workers = int(cfg.data.cache.num_workers)

    if cache_rate > 0.0:
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
    else:
        train_ds = Dataset(data=train_files, transform=train_tf)
        val_ds = Dataset(data=val_files, transform=val_tf)

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

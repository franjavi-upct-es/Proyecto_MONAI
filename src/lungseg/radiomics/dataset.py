"""Dataset builder for Phase 5 radiomics/classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from lungseg.radiomics.extractor import extract

_REQUIRED_COLUMNS = {"patient_id", "nodule_id", "image", "mask_gt", "malignancy_median"}


def _select(cfg: DictConfig, key: str, default=None):
    try:
        return OmegaConf.select(cfg, key, default=default)
    except Exception:
        return default


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"LIDC manifest not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["nodules"] if isinstance(payload, dict) and "nodules" in payload else payload
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def _label_from_malignancy(value: float, benign_max: float, malignant_min: float) -> int | None:
    if value >= malignant_min:
        return 1
    if value <= benign_max:
        return 0
    return None


def _resolve_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _mask_for_row(row: pd.Series, cfg: DictConfig, e2e: bool, base_dir: Path) -> Path:
    if not e2e:
        return _resolve_path(row["mask_gt"], base_dir)
    pred_dir_value = _select(cfg, "data.pred_masks_dir", None)
    if pred_dir_value is None:
        raise ValueError("e2e=True requires cfg.data.pred_masks_dir with predicted masks")
    pred_dir = Path(str(pred_dir_value))
    candidates = [
        pred_dir / f"{row['patient_id']}_{row['nodule_id']}.nii.gz",
        pred_dir / f"{row['nodule_id']}.nii.gz",
        pred_dir / str(row.get("mask_pred", "")),
    ]
    for candidate in candidates:
        if candidate.name and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "predicted mask not found for "
        f"patient_id={row['patient_id']} nodule_id={row['nodule_id']} in {pred_dir}"
    )


def build_radiomic_dataset(cfg: DictConfig, e2e: bool = False):
    """Build a nodule-level LIDC radiomics table.

    The implementation expects a manifest with one row per nodule and columns:
    ``patient_id,nodule_id,image,mask_gt,malignancy_median``. This keeps the
    Phase 5 pipeline scientifically honest: Task06 has no B/M labels and will
    fail explicitly.
    """
    data_name = str(_select(cfg, "data.name", "")).lower()
    if data_name in {"task06", "task06_lung", "msd_task06"}:
        raise ValueError(
            "MSD Task06_Lung has no benign/malignant labels; use data=lidc for Phase 5"
        )
    if data_name not in {"lidc", "lidc-idri", "lidc_idri"}:
        raise ValueError(f"unsupported classification dataset {data_name!r}; expected LIDC-IDRI")

    manifest_value = _select(cfg, "data.manifest", None)
    if manifest_value is None:
        raise FileNotFoundError(
            "cfg.data.manifest is required for Phase 5. Provide a LIDC nodule manifest with "
            "patient_id,nodule_id,image,mask_gt,malignancy_median columns."
        )

    manifest_path = Path(str(manifest_value))
    manifest = _read_manifest(manifest_path)
    missing = sorted(_REQUIRED_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"LIDC manifest is missing required columns: {missing}")

    benign_max = float(_select(cfg, "data.malignancy_consensus.benign_max", 2))
    malignant_min = float(_select(cfg, "data.malignancy_consensus.malignant_min", 4))
    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        label = _label_from_malignancy(float(row["malignancy_median"]), benign_max, malignant_min)
        if label is None:
            continue
        image_path = _resolve_path(row["image"], manifest_path.parent)
        mask_path = _mask_for_row(row, cfg, e2e=e2e, base_dir=manifest_path.parent)
        features = extract(image_path, mask_path)
        rows.append(
            {
                "patient_id": str(row["patient_id"]),
                "nodule_id": str(row["nodule_id"]),
                "label": int(label),
                "malignancy_median": float(row["malignancy_median"]),
                "image": str(image_path),
                "mask": str(mask_path),
                **features,
            }
        )

    if not rows:
        raise ValueError(
            "no benign/malignant LIDC nodules remained after discarding malignancy_median == 3"
        )
    table = pd.DataFrame(rows)
    feature_columns = [
        c
        for c in table.columns
        if c not in {"patient_id", "nodule_id", "label", "malignancy_median", "image", "mask"}
    ]
    X = table[feature_columns].to_numpy(dtype=np.float32)
    y = table["label"].to_numpy(dtype=np.int64)
    groups = table["patient_id"].to_numpy(dtype=str)
    return {"table": table, "X": X, "y": y, "groups": groups, "feature_names": feature_columns}

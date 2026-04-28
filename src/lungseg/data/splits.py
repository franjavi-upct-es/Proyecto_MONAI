"""Patient-level GroupKFold splits with rough stratification by tumor volume.

Outputs deterministic `fold_{0..k-1}.json` files under `out_dir` containing
per-case metadata so trainers in B4/B6 don't need to re-read NIfTI headers.

Stratification uses tertile bins of `tumor_volume_mm3`; with ~63 cases the
three strata hold ~21 each, more than enough for `n_splits=5`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class CaseRecord:
    image: str          # path relative to repo root
    label: str          # path relative to repo root
    patient_id: str
    tumor_volume_mm3: float
    stratum: int

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "label": self.label,
            "patient_id": self.patient_id,
            "tumor_volume_mm3": round(self.tumor_volume_mm3, 3),
            "stratum": self.stratum,
        }


def _resolve_relative(path_in_json: str, dataset_dir: Path) -> str:
    """`dataset.json` paths look like './imagesTr/lung_001.nii.gz'.
    Return path relative to REPO_ROOT for portability.
    """
    abs_path = (dataset_dir / path_in_json.lstrip("./")).resolve()
    return str(abs_path.relative_to(REPO_ROOT))


def _compute_tumor_volume(label_abs: Path) -> float:
    img = nib.load(str(label_abs))
    voxel_volume = float(np.prod(img.header.get_zooms()[:3]))
    data = np.asarray(img.dataobj)
    n_voxels = float((data > 0).sum())
    return n_voxels * voxel_volume


def _assign_strata(volumes: np.ndarray) -> np.ndarray:
    """Tertile binning. Returns int array in {0, 1, 2}."""
    edges = np.percentile(volumes, [100 / 3.0, 200 / 3.0])
    return np.digitize(volumes, edges).astype(int)


def make_splits(
    dataset_json: Path,
    out_dir: Path,
    seed: int = 42,
    k: int = 5,
) -> list[Path]:
    dataset_json = Path(dataset_json)
    out_dir = Path(out_dir)
    dataset_dir = dataset_json.parent
    contents = json.loads(dataset_json.read_text(encoding="utf-8"))

    images: list[str] = []
    labels: list[str] = []
    patient_ids: list[str] = []
    volumes: list[float] = []

    for entry in contents["training"]:
        image_rel = _resolve_relative(entry["image"], dataset_dir)
        label_rel = _resolve_relative(entry["label"], dataset_dir)
        patient = Path(entry["image"]).name.replace(".nii.gz", "")
        label_abs = REPO_ROOT / label_rel
        images.append(image_rel)
        labels.append(label_rel)
        patient_ids.append(patient)
        volumes.append(_compute_tumor_volume(label_abs))

    volumes_arr = np.asarray(volumes, dtype=np.float64)
    strata = _assign_strata(volumes_arr)

    cases = [
        CaseRecord(
            image=images[i],
            label=labels[i],
            patient_id=patient_ids[i],
            tumor_volume_mm3=float(volumes_arr[i]),
            stratum=int(strata[i]),
        )
        for i in range(len(images))
    ]

    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    dummy_x = np.zeros(len(cases))

    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(dummy_x, y=strata, groups=np.asarray(patient_ids))
    ):
        train = sorted([cases[i] for i in train_idx], key=lambda c: c.patient_id)
        val = sorted([cases[i] for i in val_idx], key=lambda c: c.patient_id)
        payload = {
            "fold": fold_idx,
            "seed": seed,
            "k": k,
            "n_train": len(train),
            "n_val": len(val),
            "train": [c.to_dict() for c in train],
            "val": [c.to_dict() for c in val],
        }
        out_path = out_dir / f"fold_{fold_idx}.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        out_paths.append(out_path)

    return out_paths

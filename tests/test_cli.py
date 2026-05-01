from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from lungseg.cli import _prepare_output_dir, _repo_root
from lungseg.data.datamodule import build_loaders


def test_prepare_output_dir_handles_unresolved_hydra_interpolation(tmp_path: Path) -> None:
    config_dir = str((_repo_root() / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="sanity")

    prepared = _prepare_output_dir(cfg, config_name="sanity", overrides=[])

    output_path = Path(str(prepared.paths.outputs))
    assert output_path.is_absolute()
    assert output_path.exists()
    assert (output_path / ".hydra" / "config.yaml").exists()


def test_build_loaders_falls_back_when_cache_dataset_is_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    """Verifica que si falla CacheDataset (RAM), se retroceda a un Dataset (o PersistentDataset según la config)."""
    config_dir = str((_repo_root() / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="sanity")
    # Forzamos caché en RAM para este test específico de retroceso
    OmegaConf.update(cfg, "data.cache.mode", "ram", merge=False)
    OmegaConf.update(cfg, "data.cache.rate", 1.0, merge=False)
    OmegaConf.update(cfg, "data.cache.num_workers", 1, merge=False)

    def _boom(*args, **kwargs):
        raise PermissionError("no semaphores")

    monkeypatch.setattr("lungseg.data.datamodule.CacheDataset", _boom)
    train_loader, val_loader = build_loaders(cfg, fold=0, repo_root=_repo_root())

    # Debería retroceder a Dataset normal al fallar la RAM
    assert train_loader.dataset.__class__.__name__ == "Dataset"
    assert val_loader.dataset.__class__.__name__ == "Dataset"


def test_build_loaders_defaults_to_persistent_dataset() -> None:
    """Verifica que ahora el valor por defecto es PersistentDataset (caché en disco)."""
    config_dir = str((_repo_root() / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="sanity")

    train_loader, val_loader = build_loaders(cfg, fold=0, repo_root=_repo_root())

    # Tras nuestro ajuste de "punto medio", el valor por defecto es PersistentDataset
    assert train_loader.dataset.__class__.__name__ == "PersistentDataset"
    assert val_loader.dataset.__class__.__name__ == "PersistentDataset"


# def test_local_training_profile_is_middle_ground_for_performance() -> None:
#     """Verifica que el perfil local usa el 'punto medio' de 2 workers."""
#     config_dir = str((_repo_root() / "configs").resolve())
#     with initialize_config_dir(config_dir=config_dir, version_base=None):
#         cfg = compose(config_name="config", overrides=["training=local_5060"])

#     assert int(cfg.training.num_workers) == 4
#     assert bool(cfg.training.pin_memory) is True
#     assert float(cfg.data.cache.rate) == 1.0
#     assert int(cfg.data.cache.num_workers) == 4


def test_build_loaders_supports_persistent_disk_cache(monkeypatch, tmp_path: Path) -> None:
    config_dir = str((_repo_root() / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="sanity")
    OmegaConf.update(cfg, "data.cache.mode", "disk", merge=False)
    OmegaConf.update(cfg, "data.cache.disk_dir", str(tmp_path / "cache"), merge=False)

    instances = []

    class _FakePersistentDataset:
        def __init__(self, data, transform, cache_dir):
            self.data = list(data)
            self.transform = transform
            self.cache_dir = Path(cache_dir)
            instances.append(self)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    monkeypatch.setattr("lungseg.data.datamodule.PersistentDataset", _FakePersistentDataset)
    train_loader, val_loader = build_loaders(cfg, fold=0, repo_root=_repo_root())

    assert train_loader.dataset.__class__.__name__ == "_FakePersistentDataset"
    assert val_loader.dataset.__class__.__name__ == "_FakePersistentDataset"
    assert instances[0].cache_dir == tmp_path / "cache" / "fold_0" / "train"
    assert instances[1].cache_dir == tmp_path / "cache" / "fold_0" / "val"

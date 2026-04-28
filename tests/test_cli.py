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


def test_build_loaders_falls_back_when_cache_dataset_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    config_dir = str((_repo_root() / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="sanity")
    OmegaConf.update(cfg, "data.cache.rate", 1.0, merge=False)
    OmegaConf.update(cfg, "data.cache.num_workers", 1, merge=False)

    def _boom(*args, **kwargs):
        raise PermissionError("no semaphores")

    monkeypatch.setattr("lungseg.data.datamodule.CacheDataset", _boom)
    train_loader, val_loader = build_loaders(cfg, fold=0, repo_root=_repo_root())

    assert train_loader.dataset.__class__.__name__ == "Dataset"
    assert val_loader.dataset.__class__.__name__ == "Dataset"

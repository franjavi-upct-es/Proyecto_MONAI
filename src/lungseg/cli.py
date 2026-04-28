"""Typer CLI entrypoint with subcommands `train`, `predict`, `ablate`.

The actual command bodies are filled in B4 (train/predict) and B6 (ablate).
B1 only wires the surface so `python -m lungseg.cli --help` works.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="lungseg",
    help="Lung tumor segmentation & classification on MSD Task06_Lung.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def train(config_name: str = typer.Option("config", "--config-name")) -> None:
    """Train a segmentation model. Implemented in B4."""
    raise NotImplementedError("B4 will wire the iteration-based trainer.")


@app.command()
def predict(checkpoint: str = typer.Option(..., "--checkpoint")) -> None:
    """Run sliding-window inference on a NIfTI volume. Implemented in B4."""
    raise NotImplementedError("B4 will wire the inference command.")


@app.command()
def ablate(config_name: str = typer.Option("config", "--config-name")) -> None:
    """Launch the Hydra-multirun ablation. Implemented in B6."""
    raise NotImplementedError("B6 will wire the ablation runner.")


if __name__ == "__main__":
    app()

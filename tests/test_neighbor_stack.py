"""Tests para el constructor de stack de slices vecinos en `ViT25D`."""

from __future__ import annotations

import torch

from lungseg.models.vit_25d import ViT25D


def _build(neighbor_context: int = 2) -> ViT25D:
    return ViT25D(
        in_channels=3,
        out_channels=2,
        neighbor_context=neighbor_context,
        encoder_name="vit_base_patch16_224.mae",
        pretrained=False,
        freeze_encoder=True,
        deep_supervision=False,
    )


def test_neighbor_stack_shape() -> None:
    model = _build(neighbor_context=2)
    x = torch.randn(1, 3, 8, 8, 4)
    out = model._build_neighbor_stack(x)
    assert tuple(out.shape) == (4, 15, 8, 8)


def test_neighbor_stack_replicate_padding_at_borders() -> None:
    """Para z=0, los canales de z=-2 y z=-1 deben replicar los de z=0."""
    model = _build(neighbor_context=2)
    # Construyo un tensor donde el valor por z es z mismo (en cada canal).
    D = 6
    x = torch.zeros(1, 3, 4, 4, D)
    for z in range(D):
        x[..., z] = float(z)

    out = model._build_neighbor_stack(x)  # (D, 15, 4, 4)

    # Layout de canales: 5 bloques de 3 canales (uno por offset -2,-1,0,+1,+2).
    # Cada bloque tiene 3 canales (igual valor) — todos los canales del bloque
    # contienen el slice indicado por el offset.
    # En z=0: offsets -2,-1 deben ser clampeados a 0, así que valor=0.
    for c in range(0, 6):  # primeros dos bloques
        assert out[0, c].mean() == 0.0, f"canal {c} de z=0 debería ser 0, fue {out[0, c].mean()}"

    # offset 0 -> valor 0 (z=0 mismo)
    assert out[0, 6].mean() == 0.0
    # offset +1 -> valor 1
    assert out[0, 9].mean() == 1.0
    # offset +2 -> valor 2
    assert out[0, 12].mean() == 2.0


def test_neighbor_stack_at_end_replicates() -> None:
    """Para z=D-1, los canales de z=D y z=D+1 deben replicar el último slice."""
    model = _build(neighbor_context=2)
    D = 5
    x = torch.zeros(1, 3, 4, 4, D)
    for z in range(D):
        x[..., z] = float(z)

    out = model._build_neighbor_stack(x)
    # En z=D-1=4: offsets +1, +2 se clampean a D-1=4.
    last = out[D - 1]
    # offset -2 -> z=2
    assert last[0].mean() == 2.0
    # offset -1 -> z=3
    assert last[3].mean() == 3.0
    # offset 0 -> z=4
    assert last[6].mean() == 4.0
    # offset +1 (clamped) -> z=4
    assert last[9].mean() == 4.0
    # offset +2 (clamped) -> z=4
    assert last[12].mean() == 4.0

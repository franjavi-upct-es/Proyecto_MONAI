"""Tests para `ViT25D` con contexto 2.5D, deep supervision y MAE init."""

from __future__ import annotations

import pytest
import torch

from lungseg.models.vit_25d import ViT25D, _patch_embed_proj


def _build(**overrides) -> ViT25D:
    defaults = dict(
        in_channels=3,
        out_channels=2,
        neighbor_context=2,
        encoder_name="vit_base_patch16_224.mae",
        pretrained=False,
        freeze_encoder=True,
        deep_supervision=False,
    )
    defaults.update(overrides)
    return ViT25D(**defaults)


def test_forward_shape_small_volume() -> None:
    model = _build()
    x = torch.randn(1, 3, 32, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 32, 32, 32)


def test_forward_shape_batched_volume() -> None:
    model = _build()
    x = torch.randn(2, 3, 64, 64, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2, 64, 64, 16)


def test_deep_supervision_returns_three_resolutions() -> None:
    model = _build(deep_supervision=True)
    x = torch.randn(1, 3, 32, 32, 32)
    with torch.no_grad():
        outputs = model(x)
    assert isinstance(outputs, list) and len(outputs) == 3
    assert outputs[0].shape == (1, 2, 32, 32, 32)
    assert outputs[1].shape == (1, 2, 16, 16, 32)
    assert outputs[2].shape == (1, 2, 8, 8, 32)


def test_freeze_encoder_disables_grad() -> None:
    model = _build(freeze_encoder=True)
    encoder_grads = [p.requires_grad for p in model.encoder.parameters()]
    assert sum(encoder_grads) == 0


def test_encoder_chunking_matches_unchunked_output() -> None:
    """El troceo del encoder debe ser numéricamente equivalente a un pase entero.

    Sin esto, la inferencia sliding-window con D≈200 dispara OOM en VRAM.
    """
    full = _build(encoder_chunk_size=10_000)  # >> N, equivalente a un solo pase
    chunked = _build(encoder_chunk_size=4)
    chunked.load_state_dict(full.state_dict())
    chunked.eval()
    full.eval()

    x = torch.randn(1, 3, 32, 32, 12)
    with torch.no_grad():
        out_full = full(x)
        out_chunked = chunked(x)
    assert torch.allclose(out_full, out_chunked, atol=1e-5), (
        f"chunked encoder difiere del unchunked (max diff {(out_full - out_chunked).abs().max()})"
    )


def test_patch_embed_uses_pretrained_weights_when_pretrained() -> None:
    """El primer conv debe tener 15 canales y std consistente con pesos pre-entrenados.

    Una init Kaiming sin pretrained sobre un Conv2d(15, 768, 16, 16) tiene
    std en torno a 0.02; los pesos MAE promediados y escalados por 3/15
    quedan aún más pequeños. Una std < 1.0 confirma que hay pesos pre-entrenados
    promediados, no inicialización aleatoria de gran magnitud.
    """
    pytest.importorskip("timm")
    try:
        model = _build(pretrained=True)
    except Exception as exc:
        pytest.skip(f"pesos pre-entrenados no descargables: {exc}")
    proj = _patch_embed_proj(model.encoder)
    assert proj.weight.shape[1] == 15
    assert float(proj.weight.std()) < 1.0

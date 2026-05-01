"""Tests para la arquitectura 2.5D ViT y el entrenamiento con slices."""

from __future__ import annotations

import torch

from lungseg.models.vit_25d import ViT25D


def test_vit_25d_forward_shape():
    """Verifica que el modelo recibe un volumen 3D y devuelve un volumen 3D del mismo tamaño."""
    # Desactivamos los pesos preentrenados para que el test sea rápido y sin red
    model = ViT25D(
        in_channels=1, out_channels=2, encoder_name="vit_base_patch16_224", pretrained=False
    )
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 32, 32, 32)


def test_vit_25d_arbitrary_shape():
    """Verifica que el upsampling/interpolación 2D ajusta formas irregulares correctamente."""
    model = ViT25D(
        in_channels=1, out_channels=2, encoder_name="vit_base_patch16_224", pretrained=False
    )
    x = torch.randn(1, 1, 48, 48, 16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2, 48, 48, 16)

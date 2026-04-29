"""Gancho (hook) de cuello de botella utilizado por los experimentos de características profundas de la Fase 5."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def extract_bottleneck_features(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Devuelve las características del cuello de botella con promedio global (global-average pooled) para un modelo 3D."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if hasattr(model, "encode"):
            encoded = model.encode(image)
            bottleneck = encoded[0] if isinstance(encoded, tuple) else encoded
        else:
            activations: list[torch.Tensor] = []

            def _capture(_module, _inputs, output):
                if isinstance(output, torch.Tensor) and output.dim() == 5:
                    activations.append(output)

            handles = [
                module.register_forward_hook(_capture)
                for module in model.modules()
                if isinstance(module, torch.nn.Conv3d)
            ]
            try:
                output = model(image)
                if not activations:
                    bottleneck = output[:, 0] if output.dim() == image.dim() + 1 else output
                else:
                    bottleneck = min(activations, key=lambda t: math.prod(t.shape[2:]))
            finally:
                for handle in handles:
                    handle.remove()
        features = functional.adaptive_avg_pool3d(bottleneck, output_size=1).flatten(1)
    if was_training:
        model.train()
    return features

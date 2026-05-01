# src/lungseg/training/losses.py
"""Configuración de funciones de pérdida para segmentación robusta."""

from __future__ import annotations

import torch.nn as nn
from monai.losses import DiceFocalLoss
from omegaconf import DictConfig


def build_loss(cfg: DictConfig | None = None) -> nn.Module:
    """
    Construye DiceFocalLoss con parámetros óptimos para segmentación de texturas difíciles.
    """
    return DiceFocalLoss(
        include_background=False,  # Ignorar el fondo en el cálculo del Dice
        softmax=True,  # Aplicar Softmax a las predicciones
        to_onehot_y=True,  # Convertir etiquetas a formato one-hot
        squared_pred=True,  # Elevar al cuadrado las predicciones en el denominador del Dice
        gamma=2.0,  # Factor de focal loss para penalizar errores en clases difíciles
        lambda_dice=1.0,  # Peso de la pérdida Dice
        lambda_focal=1.0,  # Peso de la pérdida Focal
    )

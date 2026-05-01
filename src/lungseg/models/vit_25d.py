# src/lungseg/models/vit_25d.py
"""Modelo 2.5D (Slice-wise) basado en ViT (timm) para segmentación de tumores."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as functional


class ViT25D(nn.Module):
    """
    Arquitectura de segmentación 2.5D que procesa cortes de profundidad como batch.
    Utiliza un encoder ViT de 2D (vía timm) y un decoder convolucional 2D.

    Transformación:
        3D Volume (B, C, H, W, D) -> 2D Slices (B*D, C, H, W)
        -> ViT Encoder -> CNN Decoder -> 2D Masks (B*D, out_C, H, W)
        -> 3D Volume (B, out_C, H, W, D)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        encoder_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        chunk_size: int = 2,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.chunk_size = chunk_size

        # El encoder se construye mediante timm.
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(-1,),
            dynamic_img_size=True,
        )

        if grad_checkpointing:
            if hasattr(self.encoder, "set_grad_checkpointing"):
                self.encoder.set_grad_checkpointing(True)
            else:
                self.encoder.grad_checkpointing = True

        # Obtener el embed_dim para conectar el decoder
        info = self.encoder.feature_info[-1]
        embed_dim = info["num_chs"]

        # El encoder reduce la resolución espacial por un factor (ej. 16 para patch16).
        # El decoder convolucional 2D realiza el upsampling para restaurar (H, W).
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=4),  # Upsample x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),  # Upsample x2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),  # Upsample x2 (total x16)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),  # Salida
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor volumétrico 3D de forma (B, C, H, W, D).
        Returns:
            Tensor de segmentación 3D de forma (B, out_channels, H, W, D).
        """
        # Convertir a tensor plano para evitar errores de consistencia de metadatos de MONAI
        # al realizar reshape y slicing (MetaTensor -> Tensor).
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()

        B, C, H, W, D = x.shape

        # (B, C, H, W, D) -> (B, D, C, H, W) -> (B*D, C, H, W)
        x_2d = x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)

        # Procesar por chunks para evitar OOM en GPUs con poca memoria (ej. P100 o T4)
        # Especialmente crítico si B*D es grande (ej. 2*96 = 192 slices)
        out_2d_list = []
        for i in range(0, x_2d.shape[0], self.chunk_size):
            x_chunk = x_2d[i : i + self.chunk_size]

            features = self.encoder(x_chunk)
            if isinstance(features, (list, tuple)):
                features = features[-1]

            chunk_out = self.decoder(features)

            # Interpolación si es necesario para cada chunk
            if chunk_out.shape[2:] != (H, W):
                chunk_out = functional.interpolate(
                    chunk_out, size=(H, W), mode="bilinear", align_corners=False
                )
            out_2d_list.append(chunk_out)

        out_2d = torch.cat(out_2d_list, dim=0)

        # Reensamblar el volumen 3D original
        # (B*D, out_C, H, W) -> (B, D, out_C, H, W) -> (B, out_C, H, W, D)
        out_3d = out_2d.view(B, D, -1, H, W).permute(0, 2, 3, 4, 1)

        return out_3d

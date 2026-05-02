# src/lungseg/models/vit_25d.py
"""Modelo 2.5D con contexto de slices vecinos sobre encoder ViT pre-entrenado.

Para cada corte axial `z` de la salida, el modelo construye una entrada 2D de
`C_in * (2K+1)` canales apilando los slices `[z-K, ..., z+K]` con replicate
padding en los bordes Z. Con K=2 y C_in=3 (ventanas HU), el ViT recibe 15
canales por slice; el primer conv del patch_embed se sobrescribe con pesos
inicializados como el promedio de los 3 canales RGB pre-entrenados, replicados
a los 15 canales nuevos para preservar el conocimiento MAE.

El decoder produce logits a tres resoluciones (full, 1/2, 1/4) si
`deep_supervision=True`; en inferencia (sliding window) el `_main_output`
selecciona la cabeza principal.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as functional


def _patch_embed_proj(encoder: nn.Module) -> nn.Conv2d:
    """Localiza el primer Conv2d del patch_embed en encoders timm.

    `features_only=True` envuelve la VisionTransformer en `FeatureGetterNet`,
    así que el patch_embed real vive en `encoder.model.patch_embed`. Sin la
    envoltura, está en `encoder.patch_embed`.
    """
    inner = getattr(encoder, "model", encoder)
    return inner.patch_embed.proj


class ViT25D(nn.Module):
    """Red de segmentación 2.5D con contexto multi-slice.

    Input:  ``(B, C_in, H, W, D)``
    Output: ``(B, C_out, H, W, D)`` o lista
            ``[full, half, quarter]`` con resoluciones decrecientes si
            ``deep_supervision=True``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        neighbor_context: int = 2,
        encoder_name: str = "vit_base_patch16_224.mae",
        pretrained: bool = True,
        freeze_encoder: bool = True,
        deep_supervision: bool = True,
        grad_checkpointing: bool = False,
        encoder_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if neighbor_context < 0:
            raise ValueError(f"neighbor_context debe ser >= 0, recibido {neighbor_context}")
        if encoder_chunk_size < 1:
            raise ValueError(f"encoder_chunk_size debe ser >= 1, recibido {encoder_chunk_size}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.neighbor_context = int(neighbor_context)
        self.deep_supervision = bool(deep_supervision)
        # Tope al batch interno del encoder para acotar VRAM en sliding window
        # (donde D ≈ Z completo, ~200 slices). En training con patch_d=16 cabe
        # todo en una pasada; en val con D=200 se trocea en lotes de 32.
        self.encoder_chunk_size = int(encoder_chunk_size)

        effective_in = self.in_channels * (2 * self.neighbor_context + 1)

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            in_chans=effective_in,
            features_only=True,
            out_indices=(-1,),
            dynamic_img_size=True,
        )

        if pretrained:
            self._adapt_input_channels(encoder_name, effective_in)

        if grad_checkpointing:
            if hasattr(self.encoder, "set_grad_checkpointing"):
                self.encoder.set_grad_checkpointing(True)
            else:
                self.encoder.grad_checkpointing = True

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        info = self.encoder.feature_info[-1]
        embed_dim = int(info["num_chs"])

        # Decoder: tres bloques de upsampling 2D (4x, 2x, 2x) -> total 16x.
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.head_full = nn.Conv2d(16, self.out_channels, kernel_size=1)

        if self.deep_supervision:
            self.head_half = nn.Conv2d(64, self.out_channels, kernel_size=1)
            self.head_quarter = nn.Conv2d(256, self.out_channels, kernel_size=1)
        else:
            self.head_half = None
            self.head_quarter = None

    def _adapt_input_channels(self, encoder_name: str, target_in: int) -> None:
        """Inicializa el primer conv para `target_in` canales desde pesos RGB.

        Carga un encoder pre-entrenado con `in_chans=3`, toma el promedio de los
        3 canales del primer conv, y lo replica `target_in` veces en el conv
        actual del modelo escalando por `3 / target_in` para conservar la
        magnitud de activación. Preserva el conocimiento pre-entrenado MAE en
        lugar de descartarlo cuando se cambia el número de canales de entrada.
        """
        source = timm.create_model(
            encoder_name,
            pretrained=True,
            in_chans=3,
            num_classes=0,
        )
        src_proj = source.patch_embed.proj
        src_weight = src_proj.weight.detach()  # (out, 3, kH, kW)
        avg = src_weight.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)
        new_weight = avg.repeat(1, target_in, 1, 1) * (3.0 / float(target_in))

        dst_proj = _patch_embed_proj(self.encoder)
        if dst_proj.weight.shape != new_weight.shape:
            raise RuntimeError(
                f"adaptación de canales falló: dst {tuple(dst_proj.weight.shape)} "
                f"vs new {tuple(new_weight.shape)}"
            )
        with torch.no_grad():
            dst_proj.weight.copy_(new_weight)
            if dst_proj.bias is not None and src_proj.bias is not None:
                dst_proj.bias.copy_(src_proj.bias.detach())
        del source

    def _build_neighbor_stack(self, x: torch.Tensor) -> torch.Tensor:
        """Construye el tensor 2D de vecinos.

        Args:
            x: ``(B, C_in, H, W, D)``.
        Returns:
            ``(B*D, C_in*(2K+1), H, W)`` con replicate padding en los bordes Z.
        """
        B, C, H, W, D = x.shape
        K = self.neighbor_context
        if K == 0:
            return x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)

        # Replicate padding manual a lo largo de Z mediante índices clampeados.
        base = torch.arange(D, device=x.device)
        # cada window: (B, C, H, W, D) recogido vía index_select en dim Z.
        windows = []
        for offset in range(-K, K + 1):
            idx = (base + offset).clamp(min=0, max=D - 1)
            windows.append(x.index_select(dim=4, index=idx))
        # stack -> (B, C, H, W, D, 2K+1)
        stacked = torch.stack(windows, dim=-1)
        # -> (B, D, 2K+1, C, H, W) -> (B*D, (2K+1)*C, H, W)
        stacked = stacked.permute(0, 4, 5, 1, 2, 3).contiguous()
        return stacked.reshape(B * D, (2 * K + 1) * C, H, W)

    def _decode(
        self, features: torch.Tensor, target_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Aplica el decoder y produce logits 2D a las tres resoluciones.

        Returns ``(full, half_or_None, quarter_or_None)`` con shapes
        ``(N, C_out, H, W)``, ``(N, C_out, H/2, W/2)`` y ``(N, C_out, H/4, W/4)``.
        Si la salida del up1/up2 no tiene exactamente la fracción esperada
        respecto a ``target_hw`` (caso de tamaños no-divisibles por 16), se
        reajusta con interpolación bilineal antes de pasar al siguiente bloque.
        """
        x = self.up1(features)  # 4x patch_grid
        quarter_logits = self.head_quarter(x) if self.head_quarter is not None else None

        x = self.up2(x)  # 8x patch_grid
        half_logits = self.head_half(x) if self.head_half is not None else None

        x = self.up3(x)  # 16x patch_grid
        if x.shape[-2:] != target_hw:
            x = functional.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        full_logits = self.head_full(x)

        if half_logits is not None:
            target_half = (target_hw[0] // 2, target_hw[1] // 2)
            if half_logits.shape[-2:] != target_half:
                half_logits = functional.interpolate(
                    half_logits, size=target_half, mode="bilinear", align_corners=False
                )
        if quarter_logits is not None:
            target_quarter = (target_hw[0] // 4, target_hw[1] // 4)
            if quarter_logits.shape[-2:] != target_quarter:
                quarter_logits = functional.interpolate(
                    quarter_logits, size=target_quarter, mode="bilinear", align_corners=False
                )
        return full_logits, half_logits, quarter_logits

    def _run_encoder_chunked(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Ejecuta el encoder en lotes de `encoder_chunk_size` slices.

        Necesario en sliding window inference: la ventana es
        ``(B, 3, H, W, full_D)`` y `_build_neighbor_stack` produce
        ``(B*full_D, 15, H, W)`` con full_D ≈ 200, lo que dispara OOM en
        ViT-Base si pasa entero. Concatenamos las features sin grad-graph
        cuando estamos en eval para liberar VRAM intermedia.
        """
        n = x_2d.shape[0]
        chunk = self.encoder_chunk_size
        if n <= chunk:
            features = self.encoder(x_2d)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            return features

        feats: list[torch.Tensor] = []
        for i in range(0, n, chunk):
            piece = self.encoder(x_2d[i : i + chunk])
            if isinstance(piece, (list, tuple)):
                piece = piece[-1]
            feats.append(piece)
        return torch.cat(feats, dim=0)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor]:
        if hasattr(x, "as_tensor"):
            x = x.as_tensor()
        B, C, H, W, D = x.shape
        if self.in_channels != C:
            raise ValueError(
                f"ViT25D espera {self.in_channels} canales, recibió {C}"
            )

        x_2d = self._build_neighbor_stack(x)  # (B*D, effective_in, H, W)
        features = self._run_encoder_chunked(x_2d)

        full, half, quarter = self._decode(features, (H, W))

        # Recomponer 3D: (B*D, C_out, H, W) -> (B, C_out, H, W, D).
        def _to_3d(t: torch.Tensor) -> torch.Tensor:
            _, c, h, w = t.shape
            return t.view(B, D, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

        full_3d = _to_3d(full)
        if not self.deep_supervision:
            return full_3d
        assert half is not None and quarter is not None
        return [full_3d, _to_3d(half), _to_3d(quarter)]

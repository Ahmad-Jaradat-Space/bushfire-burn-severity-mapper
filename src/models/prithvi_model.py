"""Prithvi-EO-2.0-300M as a burn-severity segmenter (frozen backbone + decoder).

Prithvi-EO-2.0 (NASA/IBM) is a geospatial foundation model pre-trained on
global HLS imagery. Its expected six bands — Blue, Green, Red, NIR, SWIR-1,
SWIR-2 — are exactly the bands in our Sentinel-2 composite, so no reordering is
needed. We load the 300M ViT via terratorch, **freeze it**, and train only a
light upsampling decoder on top: a fair, MPS-feasible "linear-probe-style"
comparison against the from-scratch U-Net, not a full fine-tune.

The ViT returns one token sequence ``[B, 1+gh*gw, 1024]`` per layer; we take the
last layer, drop the CLS token, fold the patch tokens back into a ``14x14``
feature map and decode it up to a per-pixel severity logit map — matching the
``forward(x) -> [B, num_classes, H, W]`` contract of the other model wrappers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Prithvi's expected band order == our composite band order (B02,B03,B04,B08,B11,B12)
PRITHVI_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
BACKBONE_NAME = "terratorch_prithvi_eo_v2_300"


class _Decoder(nn.Module):
    """14x14x1024 patch features -> 224x224 logits via 4 bilinear-upsample blocks."""

    def __init__(
        self, in_dim: int = 1024, num_classes: int = 4, mid: int = 256, dropout: float = 0.1
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, mid, 1)
        self.blocks = nn.ModuleList([self._up(mid, mid) for _ in range(4)])  # x16
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(mid, num_classes, 1)

    @staticmethod
    def _up(ci: int, co: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ci, co, 3, padding=1),
            nn.GroupNorm(16, co),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        return self.head(self.drop(x))


class PrithviWrapper(nn.Module):
    def __init__(
        self, num_classes: int = 4, frozen: bool = True, img_size: int = 224, dropout: float = 0.1
    ):
        super().__init__()
        from terratorch.registry import BACKBONE_REGISTRY

        self.img_size = img_size
        self.frozen = frozen
        self.embed_dim = 1024
        self.backbone = BACKBONE_REGISTRY.build(
            BACKBONE_NAME, pretrained=True, num_frames=1, bands=PRITHVI_BANDS
        )
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.decoder = _Decoder(self.embed_dim, num_classes, dropout=dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()  # frozen backbone stays in eval (no BN/dropout drift)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xi = F.interpolate(
            x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )
        with torch.no_grad() if self.frozen else torch.enable_grad():
            feats = self.backbone(xi)[-1]  # [B, 1+N, 1024]
        tok = feats[:, 1:, :]  # drop CLS -> [B, N, 1024]
        g = int(round(tok.shape[1] ** 0.5))
        fmap = tok.transpose(1, 2).reshape(B, self.embed_dim, g, g)
        logits = self.decoder(fmap)  # [B, num_classes, img, img]
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


def build_prithvi(
    num_classes: int = 4, frozen: bool = True, img_size: int = 224, dropout: float = 0.1
) -> PrithviWrapper:
    return PrithviWrapper(
        num_classes=num_classes, frozen=frozen, img_size=img_size, dropout=dropout
    )

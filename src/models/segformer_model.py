"""HuggingFace SegFormer-B0 adapted to 18-channel input.

We load the mit-b0 backbone with ImageNet pretraining (3-channel input), then
**inflate the first patch-embed conv** from 3→18 channels by tiling/averaging
the RGB weights. This preserves the pretrained spatial structure while
accepting our 18-channel pre/post/delta/topo stack.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def _inflate_first_conv(conv: nn.Conv2d, new_in_channels: int) -> nn.Conv2d:
    """Return a new Conv2d with `new_in_channels` inputs whose weights repeat the
    averaged RGB kernel scaled by 3 / new_in_channels (so output magnitude is preserved).
    """
    old_w = conv.weight.data            # [out, 3, kH, kW]
    mean_w = old_w.mean(dim=1, keepdim=True)   # [out, 1, kH, kW]
    new_w = mean_w.repeat(1, new_in_channels, 1, 1) * (3.0 / new_in_channels)
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    new_conv.weight.data.copy_(new_w)
    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data)
    return new_conv


def build_segformer(in_channels: int = 18, num_classes: int = 4,
                    backbone: str = "nvidia/mit-b0") -> nn.Module:
    """Return a SegformerForSemanticSegmentation adapted for 18ch input."""
    cfg = SegformerConfig.from_pretrained(backbone, num_labels=num_classes,
                                          ignore_mismatched_sizes=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        backbone, config=cfg, ignore_mismatched_sizes=True,
    )

    # HF SegFormer in this version stores patch embeddings under
    # model.segformer.stages[0].patch_embeddings.proj (a single 3->32 Conv2d).
    # Walk the model to locate the first 3-channel Conv2d as a defensive fallback
    # in case the attribute path changes in a future transformers release.
    patch_embed_proj = None
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "Conv2d" and mod.in_channels == 3:
            patch_embed_proj = (name, mod)
            break
    if patch_embed_proj is None:
        raise RuntimeError("Could not locate the 3-channel patch embedding Conv2d in SegFormer.")
    parent_name, old_proj = patch_embed_proj
    if old_proj.in_channels != in_channels:
        # Walk to the parent module to swap in the inflated conv
        parent_path = parent_name.rsplit(".", 1)
        parent_module = model
        for p in parent_path[0].split("."):
            parent_module = getattr(parent_module, p) if not p.isdigit() else parent_module[int(p)]
        setattr(parent_module, parent_path[1], _inflate_first_conv(old_proj, in_channels))

    return model


class SegformerWrapper(nn.Module):
    """Forward returns the upsampled logits at input resolution.

    HF SegFormer outputs logits at H/4 × W/4; we bilinear-upsample to the input
    spatial size so the training loop is identical to U-Net.
    """

    def __init__(self, in_channels: int = 18, num_classes: int = 4,
                 backbone: str = "nvidia/mit-b0"):
        super().__init__()
        self.model = build_segformer(in_channels, num_classes, backbone)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits_lo = out.logits     # [B, C, H/4, W/4]
        logits = torch.nn.functional.interpolate(
            logits_lo, size=x.shape[-2:], mode="bilinear", align_corners=False,
        )
        return logits

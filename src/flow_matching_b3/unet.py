"""ADM-style U-Net (Dhariwal & Nichol 2021) used by the paper.

This is a clean, self-contained reimplementation faithful to the ADM design:
ResBlocks with GroupNorm, FiLM-style time conditioning, optional self-attention
at chosen resolutions, strided-conv downsamplers, nearest-neighbour upsamplers,
skip concatenation across the U.

For CIFAR-10 we follow the ADM/Dhariwal-Nichol CIFAR lineage: ``base_channels=128``
(reaching 256 at the deepest level via ``channels_mult=(1, 2, 2, 2)``), ``depth=2``,
attention at resolution 16, 4 heads of width 64 — which is exactly the full channel
width (256) at that resolution. This is ~38M params, matching the paper's "not
optimized for CIFAR-10" ADM backbone. Exposed through :func:`adm_unet_cifar10`.

The 2-D MLP variant used for the checkerboard sanity experiment is also
provided here as :class:`MLPVectorField`, since both share the same interface
``model(x, t) → tensor of the same shape as x``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------


def sinusoidal_embedding(t: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Standard transformer-style sinusoidal embedding for scalar time ``t``."""
    if t.dim() != 1:
        t = t.flatten()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.to(t.dtype if t.is_floating_point() else torch.float32)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, c), num_channels=c, eps=1e-6)


class ResBlock(nn.Module):
    """GroupNorm + SiLU + Conv, with FiLM-style time injection (ADM-style)."""

    def __init__(self, in_ch: int, out_ch: int, time_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_ch, 2 * out_ch)
        self.norm2 = _gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t_emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """QKV self-attention with ``num_heads`` heads of width ``head_channels``."""

    def __init__(self, channels: int, num_heads: int = 4, head_channels: int = 64):
        super().__init__()
        if channels % (num_heads * head_channels) != 0:
            # Allow non-perfect division: fall back to channels // num_heads.
            head_channels = channels // num_heads
        self.num_heads = num_heads
        self.head_channels = head_channels
        self.norm = _gn(channels)
        self.qkv = nn.Conv2d(channels, 3 * num_heads * head_channels, kernel_size=1)
        self.proj = nn.Conv2d(num_heads * head_channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = rearrange(
            qkv, "b (three nh d) h w -> three b nh (h w) d", three=3, nh=self.num_heads
        ).unbind(0)
        # PyTorch SDPA: handles scale and broadcasting; runs FlashAttn on CUDA.
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh (h w) d -> b (nh d) h w", h=h, w=w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------


@dataclass
class UNetConfig:
    image_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 128
    channels_mult: tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16,)
    num_heads: int = 4
    head_channels: int = 64
    dropout: float = 0.0


class UNet(nn.Module):
    """ADM-style U-Net with time conditioning. Input/output shapes are identical."""

    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        c0 = cfg.base_channels
        time_ch = 4 * c0
        self.time_mlp = nn.Sequential(
            nn.Linear(c0, time_ch),
            nn.SiLU(),
            nn.Linear(time_ch, time_ch),
        )
        self.in_conv = nn.Conv2d(cfg.in_channels, c0, kernel_size=3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.skip_channels: list[int] = [c0]
        ch = c0
        res = cfg.image_size
        for level, mult in enumerate(cfg.channels_mult):
            out_ch = c0 * mult
            for _ in range(cfg.num_res_blocks):
                blocks: list[nn.Module] = [ResBlock(ch, out_ch, time_ch, cfg.dropout)]
                if res in cfg.attn_resolutions:
                    blocks.append(SelfAttention(out_ch, cfg.num_heads, cfg.head_channels))
                self.down_blocks.append(nn.ModuleList(blocks))
                ch = out_ch
                self.skip_channels.append(ch)
            if level != len(cfg.channels_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                self.skip_channels.append(ch)
                res //= 2

        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_ch, cfg.dropout)
        self.mid_attn = SelfAttention(ch, cfg.num_heads, cfg.head_channels)
        self.mid_block2 = ResBlock(ch, ch, time_ch, cfg.dropout)

        # Decoder (mirror)
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(cfg.channels_mult))):
            out_ch = c0 * mult
            for _ in range(cfg.num_res_blocks + 1):
                skip_ch = self.skip_channels.pop()
                blocks = [ResBlock(ch + skip_ch, out_ch, time_ch, cfg.dropout)]
                if res in cfg.attn_resolutions:
                    blocks.append(SelfAttention(out_ch, cfg.num_heads, cfg.head_channels))
                self.up_blocks.append(nn.ModuleList(blocks))
                ch = out_ch
            if level != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))
                res *= 2

        self.out_norm = _gn(ch)
        self.out_conv = nn.Conv2d(ch, cfg.out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _embed_time(self, t: Tensor) -> Tensor:
        return self.time_mlp(sinusoidal_embedding(t, self.cfg.base_channels))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self._embed_time(t)
        h = self.in_conv(x)
        skips: list[Tensor] = [h]
        for blocks in self.down_blocks:
            for b in blocks:
                h = b(h, t_emb) if isinstance(b, ResBlock) else b(h)
            skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for blocks in self.up_blocks:
            first = blocks[0]
            if isinstance(first, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
            for b in blocks:
                h = b(h, t_emb) if isinstance(b, ResBlock) else b(h)

        return self.out_conv(F.silu(self.out_norm(h)))


def adm_unet_cifar10(dropout: float = 0.0) -> UNet:
    """ADM CIFAR-10 config: base 128 ch (→256 at res 16), depth 2, attn@16, 4 heads × 64.

    ~38.3M params. base=128 with mult (1,2,2,2) reaches 256 channels at resolution 16,
    where the 4×64=256 attention covers the full width (no bottleneck).
    """
    return UNet(
        UNetConfig(
            image_size=32,
            in_channels=3,
            out_channels=3,
            base_channels=128,
            channels_mult=(1, 2, 2, 2),
            num_res_blocks=2,
            attn_resolutions=(16,),
            num_heads=4,
            head_channels=64,
            dropout=dropout,
        )
    )


# ---------------------------------------------------------------------------
# 2-D MLP for the checkerboard sanity experiment (paper §6, Fig. 4)
# ---------------------------------------------------------------------------


class MLPVectorField(nn.Module):
    """5-layer MLP of width 512, used for the 2-D toy experiment in the paper."""

    def __init__(self, dim: int = 2, hidden: int = 512, num_layers: int = 5):
        super().__init__()
        self.time_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        layers: list[nn.Module] = [nn.Linear(dim + hidden, hidden), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        return self.net(torch.cat([x, t_emb], dim=-1))

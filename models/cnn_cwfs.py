import torch
import torch.nn as nn

try:
    from .common import CrossAttentionBlock, MLPHead, encode_and_pool
except ImportError:
    from models.common import CrossAttentionBlock, MLPHead, encode_and_pool


# ──────────────────────────────────────────────────────────────────────
# ResNet building blocks
# ──────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """
    Standard ResNet-18/34 BasicBlock:

        x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → + shortcut → ReLU

    A 1×1 projection is applied to the shortcut whenever the spatial size
    or channel count changes (stride > 1 or in_ch ≠ out_ch).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int = 2) -> nn.Sequential:
    """Build one ResNet stage: first block downsamples, rest keep spatial size."""
    blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
    for _ in range(n_blocks - 1):
        blocks.append(BasicBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*blocks)


class ResNetBackbone(nn.Module):
    """
    Lightweight ResNet-style feature extractor for single-channel PSF images.

    Spatial progression from a 256×256 input:
        Stem  (stride 1) : 256×256, base_ch
        Stage 1 (stride 2): 128×128, base_ch×2
        Stage 2 (stride 2):  64×64, base_ch×4
        Stage 3 (stride 2):  32×32, base_ch×8
        Stage 4 (stride 2):  16×16, base_ch×8   ← output spatial map

    With the default base_ch=32:
        output shape: [B, 256, 16, 16]  →  256 spatial tokens of dim 256

    Parameters
    ----------
    base_ch     : int  — stem output channels (default 32)
    stage_blocks: int  — number of BasicBlocks per stage (default 2)
    """

    def __init__(self, base_ch: int = 32, stage_blocks: int = 2):
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_stage(c,     c * 2, stage_blocks, stride=2)   # 128
        self.stage2 = _make_stage(c * 2, c * 4, stage_blocks, stride=2)   #  64
        self.stage3 = _make_stage(c * 4, c * 8, stage_blocks, stride=2)   #  32
        self.stage4 = _make_stage(c * 8, c * 8, stage_blocks, stride=2)   #  16
        self.out_channels = c * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[B, 1, H, W]

        Returns
        -------
        Tensor[B, out_channels, H/16, W/16]
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


# ──────────────────────────────────────────────────────────────────────
# CNNCWFS
# ──────────────────────────────────────────────────────────────────────

class CNNCWFS(nn.Module):
    """
    Siamese ResNet + cross-attention Curvature Wavefront Sensor network.

    Architecture
    ------------
    Shared backbone (Siamese weights):
        I1 [B,1,H,W] ─┐
        I2 [B,1,H,W] ──┤ ResNetBackbone → [B, C, H', W'] → flatten spatial → [B, N, C]
        r  [B,1,H,W] ─┘

    Cross-stream attention (Stage 3):
        Block 0 : q = CrossAttn(query=F1, kv=F2)   — intra vs. extra-focal
        Block 1 : q = CrossAttn(query=q,  kv=Fr)   — refine with Roddier features
        Additional blocks alternate kv=F2 / kv=Fr.

    Regression head (Stage 4):
        Global average pool over tokens → MLPHead → Z2..Z15

    Spatial token count with default settings (base_ch=32, 256×256 input):
        C = 256 channels,  N = 16×16 = 256 tokens

    Parameters
    ----------
    base_ch        : int   — ResNet stem output channels (default 32)
    stage_blocks   : int   — BasicBlocks per stage (default 2)
    n_cross_blocks : int   — cross-attention blocks (default 2; minimum 1 for
                             two_stream/pairs, 0 allowed for r_stack)
    n_heads        : int   — attention heads; out_channels must be divisible by n_heads
    ffn_mult       : int   — FFN hidden-dim multiplier in cross-attention blocks
    dropout        : float — dropout probability
    n_outputs      : int   — number of Zernike coefficients (default 14)
    input_mode     : str   — 'two_stream' (default) | 'r_stack' | 'pairs'
    """

    def __init__(
        self,
        base_ch: int = 32,
        stage_blocks: int = 2,
        n_cross_blocks: int = 2,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        n_outputs: int = 14,
        input_mode: str = 'two_stream',
    ):
        super().__init__()
        if input_mode not in ('two_stream', 'r_stack', 'pairs'):
            raise ValueError(f"Unknown input_mode '{input_mode}'")
        if input_mode in ('pairs', 'two_stream') and n_cross_blocks < 1:
            raise ValueError("n_cross_blocks must be at least 1")
        self.input_mode = input_mode

        # Shared Siamese backbone
        self.backbone = ResNetBackbone(base_ch=base_ch, stage_blocks=stage_blocks)
        dim = self.backbone.out_channels

        if dim % n_heads != 0:
            raise ValueError(
                f"backbone output channels ({dim}) must be divisible by n_heads ({n_heads})"
            )

        # Cross-stream attention blocks
        self.cross_attn = nn.ModuleList(
            [CrossAttentionBlock(dim, n_heads, ffn_mult, dropout)
             for _ in range(n_cross_blocks)]
        )

        # Regression head
        self.head = MLPHead(dim, [dim // 2], n_outputs, dropout)

    # ------------------------------------------------------------------

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the shared backbone and reshape spatial map to token sequence.

        Parameters
        ----------
        x : Tensor[B, 1, H, W]

        Returns
        -------
        Tensor[B, N_tokens, C]  where N_tokens = (H/16) * (W/16)
        """
        feat = self.backbone(x)              # [B, C, H', W']
        B, C, Hp, Wp = feat.shape
        return feat.view(B, C, Hp * Wp).transpose(1, 2)   # [B, N, C]

    def _encode_and_pool(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, H, W] → per-frame backbone + mean pool → [B, N, C]"""
        return encode_and_pool(frames, self._extract)

    def _forward_pairs(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
        r:  torch.Tensor,
    ) -> torch.Tensor:
        # I1, I2, r: [B, 1, H, W]
        F1 = self._extract(I1)
        F2 = self._extract(I2)
        Fr = self._extract(r)
        kv_sources = [F2, Fr]
        q = F1
        for k, cross_block in enumerate(self.cross_attn):
            kv = kv_sources[k % 2]   # even k → F2; odd k → Fr
            q = cross_block(q, kv)
        return self.head(q.mean(dim=1))

    def _forward_two_stream(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
    ) -> torch.Tensor:
        # I1, I2: [B, T, H, W]
        F1 = self._encode_and_pool(I1)   # [B, N, C]
        F2 = self._encode_and_pool(I2)   # [B, N, C]
        q = F1
        for cross_block in self.cross_attn:
            q = cross_block(q, F2)       # all blocks use F2 as kv
        return self.head(q.mean(dim=1))

    def _forward_r_stack(
        self,
        R: torch.Tensor,
    ) -> torch.Tensor:
        # R: [B, T², H, W] — each frame is an independent sample
        B, TT, H, W = R.shape
        flat = R.reshape(B * TT, 1, H, W)     # [B*T², 1, H, W]
        tokens = self._extract(flat)           # [B*T², N, C]
        return self.head(tokens.mean(dim=1))   # [B*T², n_outputs]

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.input_mode == 'pairs':
            return self._forward_pairs(*args, **kwargs)
        elif self.input_mode == 'two_stream':
            return self._forward_two_stream(*args, **kwargs)
        else:
            return self._forward_r_stack(*args, **kwargs)

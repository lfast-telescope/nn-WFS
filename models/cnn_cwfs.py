import torch
import torch.nn as nn
from typing import Optional

try:
    from .common import CrossAttentionBlock, MLPHead, encode_and_pool, RoddierSignal
except ImportError:
    from models.common import CrossAttentionBlock, MLPHead, encode_and_pool, RoddierSignal


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
        When stem_stride=1 (default):
            Stem  (stride 1) : 256×256, base_ch
            Stage 1 (stride 2): 128×128, base_ch×2
            Stage 2 (stride 2):  64×64, base_ch×4
            Stage 3 (stride 2):  32×32, base_ch×8
            Stage 4 (stride 2):  16×16, base_ch×8   ← output spatial map
        When stem_stride=2:
            Stem  (stride 2) : 128×128, base_ch
            Stage 1 (stride 2):  64×64, base_ch×2
            Stage 2 (stride 2):  32×32, base_ch×4
            Stage 3 (stride 2):  16×16, base_ch×8
            Stage 4 (stride 2):   8×8,   base_ch×8   ← output spatial map

    With the default base_ch=32:
        output shape (stem_stride=1): [B, 256, 16, 16]  →  256 spatial tokens of dim 256
        output shape (stem_stride=2): [B, 256,  8,  8]  →   64 spatial tokens of dim 256

    Parameters
    ----------
    base_ch     : int  — stem output channels (default 32)
    stage_blocks: int  — number of BasicBlocks per stage (default 2)
    stem_stride : int  — stride of the initial stem convolution (default 1; set to 2 for downsampled stem)
    """

    def __init__(self, base_ch: int = 32, stage_blocks: int = 2, stem_stride: int = 1):
        super().__init__()
        if stem_stride not in (1, 2):
            raise ValueError(f"stem_stride must be 1 or 2, got {stem_stride}")
        c = base_ch
        kernel_size = 5 if stem_stride == 2 else 3
        padding = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=kernel_size, stride=stem_stride, padding=padding, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_stage(c,     c * 2, stage_blocks, stride=2)   # 128 (or 64)
        self.stage2 = _make_stage(c * 2, c * 4, stage_blocks, stride=2)   #  64 (or 32)
        self.stage3 = _make_stage(c * 4, c * 8, stage_blocks, stride=2)   #  32 (or 16)
        self.stage4 = _make_stage(c * 8, c * 8, stage_blocks, stride=2)   #  16 (or  8)
        self.out_channels = c * 8
        self.stem_stride = stem_stride

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
# SIAMCNN (formerly CNNCWFS)
# ──────────────────────────────────────────────────────────────────────

class SIAMCNN(nn.Module):
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
    stem_stride    : int   — stride of the initial stem convolution (default 1)
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
        stem_stride: int = 1,
    ):
        super().__init__()
        if input_mode not in ('two_stream', 'r_stack', 'pairs'):
            raise ValueError(f"Unknown input_mode '{input_mode}'")
        if input_mode in ('pairs', 'two_stream') and n_cross_blocks < 1:
            raise ValueError("n_cross_blocks must be at least 1")
        self.input_mode = input_mode

        # Shared Siamese backbone
        self.backbone = ResNetBackbone(
            base_ch=base_ch, stage_blocks=stage_blocks, stem_stride=stem_stride
        )
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


# Backward-compatibility alias
CNNCWFS = SIAMCNN


# ──────────────────────────────────────────────────────────────────────
# RODCNN — Cross-batch Roddier CNN
# ──────────────────────────────────────────────────────────────────────

class RODCNN(nn.Module):
    """
    Cross-batch Roddier CNN for curvature wavefront sensing.

    Architecture
    ------------
    Input: the B=T intra-focal frames (I1) and B=T extra-focal frames (I2)
    of a *single* HDF5 example (same mirror state, independent atmospheric
    realisations per frame — see CWFSDataset's return_stacks=True mode).
    train.py feeds one example per iteration (no cross-example batching),
    so every (I1_i, I2_j) pair formed below is guaranteed to share the same
    Zernike label.

    B² expansion (inside forward):
        All B×B combinations of (I1_i, I2_j) are formed, yielding B² Roddier
        signals  r_ij = (I1_i − I2_j) / (I1_i + I2_j + ε).

    Single-stream backbone (no cross-attention):
        r_all [B², 1, H, W] → ResNetBackbone → [B², C, Hp, Wp]
        → flatten → [B², N, C] → global average pool → [B², C]
        → MLPHead → [B², n_outputs]

    Training objective:
        Loss on pred.mean(dim=0) vs the example's (single) label vector.
        Training the mean of B² predictions forces the network to extract
        the invariant mirror wavefront while averaging out diverse
        atmospheric realisations.

    Parameters
    ----------
    base_ch      : int   — ResNet stem output channels (default 32)
    stage_blocks : int   — BasicBlocks per stage (default 2)
    dropout      : float — dropout probability in regression head
    n_outputs    : int   — number of Zernike coefficients to predict (default 14)
    stem_stride  : int   — stride of the initial stem convolution (default 1)
    """

    def __init__(
        self,
        base_ch: int = 32,
        stage_blocks: int = 2,
        dropout: float = 0.0,
        n_outputs: int = 14,
        stem_stride: int = 1,
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.roddier   = RoddierSignal()
        self.backbone  = ResNetBackbone(
            base_ch=base_ch, stage_blocks=stage_blocks, stem_stride=stem_stride
        )
        dim            = self.backbone.out_channels
        self.head      = MLPHead(dim, [dim // 2], n_outputs, dropout)

    def forward(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
        k_pairs: Optional[int] = None,
        return_all_pairs: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        I1 : Tensor[B, T, 1, H, W] or [T, 1, H, W] or [B, T, H, W]
            Intra-focal frame stacks.
        I2 : Tensor[B, T, 1, H, W] or [T, 1, H, W] or [B, T, H, W]
            Extra-focal frame stacks.
        k_pairs : int, optional
            Number of (I1_i, I2_j) pairs to randomly subsample per example.
            Only applied during training (self.training=True).
            If None or during evaluation, evaluates all T² combinations.
        return_all_pairs : bool
            If True, returns [B, K, n_outputs] before averaging over pairs.
            If False (default), returns [B, n_outputs].

        Returns
        -------
        Tensor[B, n_outputs] (or [B, K, n_outputs] if return_all_pairs=True)
        """
        # Standardise input to [B, T, 1, H, W]
        if I1.dim() == 3:  # [T, H, W]
            I1 = I1.unsqueeze(0).unsqueeze(2)
            I2 = I2.unsqueeze(0).unsqueeze(2)
        elif I1.dim() == 4:
            if I1.shape[1] == 1:  # [T, 1, H, W]
                I1 = I1.unsqueeze(0)
                I2 = I2.unsqueeze(0)
            else:  # [B, T, H, W]
                I1 = I1.unsqueeze(2)
                I2 = I2.unsqueeze(2)

        B, T, C, H, W = I1.shape
        use_subsample = (self.training and k_pairs is not None and k_pairs < T * T)

        if use_subsample:
            # Sample k_pairs random (i, j) frame index pairs per example
            idx_i = torch.randint(0, T, (B, k_pairs), device=I1.device)
            idx_j = torch.randint(0, T, (B, k_pairs), device=I2.device)
            batch_idx = torch.arange(B, device=I1.device).unsqueeze(1).expand(B, k_pairs)

            I1_sel = I1[batch_idx, idx_i]  # [B, k_pairs, 1, H, W]
            I2_sel = I2[batch_idx, idx_j]  # [B, k_pairs, 1, H, W]

            I1_flat = I1_sel.reshape(B * k_pairs, C, H, W)
            I2_flat = I2_sel.reshape(B * k_pairs, C, H, W)
            n_pairs_per_ex = k_pairs
        else:
            # Vectorised full T² expansion
            I1_rep = I1.unsqueeze(2).expand(B, T, T, C, H, W).reshape(B * T * T, C, H, W)
            I2_rep = I2.unsqueeze(1).expand(B, T, T, C, H, W).reshape(B * T * T, C, H, W)
            I1_flat = I1_rep
            I2_flat = I2_rep
            n_pairs_per_ex = T * T

        r_all = self.roddier(I1_flat, I2_flat)               # [B*K, 1, H, W]
        feat = self.backbone(r_all)                          # [B*K, dim, Hp, Wp]
        BK, ch, Hp, Wp = feat.shape
        tokens = feat.view(BK, ch, Hp * Wp).transpose(1, 2)  # [B*K, N, ch]
        pooled = tokens.mean(dim=1)                          # [B*K, ch]
        preds_flat = self.head(pooled)                       # [B*K, n_outputs]

        preds = preds_flat.view(B, n_pairs_per_ex, self.n_outputs)
        if return_all_pairs:
            return preds
        return preds.mean(dim=1)  # [B, n_outputs]

    def predict(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference helper: average predictions over all T² available pairs.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.forward(I1, I2, k_pairs=None, return_all_pairs=False)
        if was_training:
            self.train()
        return out

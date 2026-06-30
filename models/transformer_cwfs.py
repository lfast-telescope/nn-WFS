import torch
import torch.nn as nn

try:
    from .common import PatchEmbed, TransformerBlock, CrossAttentionBlock, MLPHead, encode_and_pool
except ImportError:
    from models.common import PatchEmbed, TransformerBlock, CrossAttentionBlock, MLPHead, encode_and_pool


class TransformerCWFS(nn.Module):
    """
    Transformer-based Curvature Wavefront Sensor network.

    Architecture
    ------------
    Stage 1 — Spatial encoder (shared weights, applied independently to each stream):
        I1 [B,1,H,W] ─┐
        I2 [B,1,H,W] ──┤ PatchEmbed + pos_embed → N patches → 4×TransformerBlock
        r  [B,1,H,W] ─┘
        → three token sets f1, f2, fr each [B, N_patches, D]

    Stage 3 — Cross-stream attention:
        Block 0 : q = CrossAttn(query=f1, kv=f2)   — fuse intra vs extra-focal
        Block 1 : q = CrossAttn(query=q,  kv=fr)   — refine with Roddier signal
        Additional blocks (if n_cross_blocks > 2) alternate kv=f2 / kv=fr.

    Stage 4 — Regression head:
        LayerNorm → global average pool over patches → MLPHead → Z2..Z15

    Parameters
    ----------
    img_size       : int   — spatial side length (must be divisible by patch_size)
    patch_size     : int   — patch side length; 16 → 256 tokens, 32 → 64 tokens
    embed_dim      : int   — token dimension D
    n_enc_blocks   : int   — Stage-1 self-attention blocks (default 4)
    n_cross_blocks : int   — Stage-3 cross-attention blocks (default 2; minimum 1 for
                             two_stream/pairs, 0 allowed for r_stack)
    n_heads        : int   — attention heads (embed_dim must be divisible by n_heads)
    ffn_mult       : int   — FFN hidden-dim multiplier
    dropout        : float — dropout probability in attention and FFN layers
    n_outputs      : int   — number of Zernike coefficients to predict (default 14)
    input_mode     : str   — 'two_stream' (default) | 'r_stack' | 'pairs'
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 256,
        n_enc_blocks: int = 4,
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

        # --- Stage 1: shared patch tokeniser + positional embedding ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=1, embed_dim=embed_dim)
        n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- Stage 1: shared spatial encoder ---
        self.encoder = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, ffn_mult, dropout)
             for _ in range(n_enc_blocks)]
        )

        # --- Stage 3: cross-stream attention ---
        self.cross_attn = nn.ModuleList(
            [CrossAttentionBlock(embed_dim, n_heads, ffn_mult, dropout)
             for _ in range(n_cross_blocks)]
        )

        # --- Stage 4: regression head ---
        self.norm = nn.LayerNorm(embed_dim)
        self.head = MLPHead(embed_dim, [embed_dim, embed_dim // 4], n_outputs, dropout)

    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenise and spatially encode a single PSF image.

        Parameters
        ----------
        x : Tensor[B, 1, H, W]

        Returns
        -------
        Tensor[B, N_patches, embed_dim]
        """
        tokens = self.patch_embed(x) + self.pos_embed
        for block in self.encoder:
            tokens = block(tokens)
        return tokens

    def _encode_and_pool(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, H, W] → per-frame encode + mean pool → [B, N, D]"""
        return encode_and_pool(frames, self._encode)

    def _forward_pairs(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
        r:  torch.Tensor,
    ) -> torch.Tensor:
        # I1, I2, r: [B, 1, H, W]
        f1 = self._encode(I1)
        f2 = self._encode(I2)
        fr = self._encode(r)
        kv_sources = [f2, fr]
        q = f1
        for k, cross_block in enumerate(self.cross_attn):
            kv = kv_sources[k % 2]   # even k → f2; odd k → fr
            q = cross_block(q, kv)
        return self.head(self.norm(q).mean(dim=1))

    def _forward_two_stream(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
    ) -> torch.Tensor:
        # I1, I2: [B, T, H, W]
        f1 = self._encode_and_pool(I1)   # [B, N, D]
        f2 = self._encode_and_pool(I2)   # [B, N, D]
        q = f1
        for cross_block in self.cross_attn:
            q = cross_block(q, f2)       # all blocks use f2 as kv
        return self.head(self.norm(q).mean(dim=1))

    def _forward_r_stack(
        self,
        R: torch.Tensor,
    ) -> torch.Tensor:
        # R: [B, T², H, W] — each frame is an independent sample
        B, TT, H, W = R.shape
        flat = R.reshape(B * TT, 1, H, W)              # [B*T², 1, H, W]
        tokens = self._encode(flat)                     # [B*T², N, D]
        return self.head(self.norm(tokens).mean(dim=1)) # [B*T², n_outputs]

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.input_mode == 'pairs':
            return self._forward_pairs(*args, **kwargs)
        elif self.input_mode == 'two_stream':
            return self._forward_two_stream(*args, **kwargs)
        else:
            return self._forward_r_stack(*args, **kwargs)

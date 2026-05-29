import torch
import torch.nn as nn

try:
    from .common import PatchEmbed, TransformerBlock, CrossAttentionBlock, MLPHead
except ImportError:
    from models.common import PatchEmbed, TransformerBlock, CrossAttentionBlock, MLPHead


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
    n_cross_blocks : int   — Stage-3 cross-attention blocks (default 2; minimum 2)
    n_heads        : int   — attention heads (embed_dim must be divisible by n_heads)
    ffn_mult       : int   — FFN hidden-dim multiplier
    dropout        : float — dropout probability in attention and FFN layers
    n_outputs      : int   — number of Zernike coefficients to predict (default 14)
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
    ):
        super().__init__()
        if n_cross_blocks < 2:
            raise ValueError("n_cross_blocks must be at least 2")

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

    def forward(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        I1 : Tensor[B, 1, H, W]  — intra-focal mean PSF
        I2 : Tensor[B, 1, H, W]  — extra-focal mean PSF
        r  : Tensor[B, 1, H, W]  — Roddier normalised-difference signal

        Returns
        -------
        Tensor[B, n_outputs]  — predicted Zernike coefficients Z2..Z15
        """
        f1 = self._encode(I1)   # [B, N, D]
        f2 = self._encode(I2)
        fr = self._encode(r)

        # Cross-stream attention cascade.
        # Block 0:      Q = f1,  KV = f2   (intra vs. extra-focal)
        # Block 1:      Q = out, KV = fr   (refine with Roddier signal)
        # Block k >= 2: alternate KV source (f2 for even k, fr for odd k)
        kv_sources = [f2, fr]
        q = f1
        for k, cross_block in enumerate(self.cross_attn):
            kv = kv_sources[min(k, 1)]   # block 0 → f2; blocks 1+ → fr
            q = cross_block(q, kv)

        pooled = self.norm(q).mean(dim=1)   # [B, D]
        return self.head(pooled)

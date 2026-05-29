import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────
# RoddierSignal
# ──────────────────────────────────────────────────────────────────────

class RoddierSignal(nn.Module):
    """
    Computes the normalised curvature-sensing signal

        r = (I1 - I2) / (I1 + I2 + eps)

    as a differentiable nn.Module so it can sit inside a model graph
    when normalisation refinement or gradient-based inspection is needed.
    For the default pipeline the signal is pre-computed in the dataset;
    this module is provided for in-model use and ablation.

    Parameters
    ----------
    eps : float
        Small stabilisation constant (default 1e-6).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, I1: torch.Tensor, I2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        I1, I2 : Tensor[..., H, W]  — intra- and extra-focal images

        Returns
        -------
        Tensor[..., H, W] ∈ (−1, 1)
        """
        return (I1 - I2) / (I1 + I2 + self.eps)


# ──────────────────────────────────────────────────────────────────────
# PatchEmbed
# ──────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """
    Converts a single-channel 2-D image into a sequence of patch tokens
    using a non-overlapping convolutional projection (ViT-style).

    Parameters
    ----------
    img_size   : int   — spatial side length of the square input (pixels)
    patch_size : int   — side length of each square patch (pixels)
    in_chans   : int   — number of input channels (default 1)
    embed_dim  : int   — output token dimension D

    Attributes
    ----------
    n_patches : int  — number of tokens  = (img_size / patch_size)²
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 256,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[B, C, H, W]

        Returns
        -------
        Tensor[B, N_patches, embed_dim]
        """
        x = self.proj(x)           # [B, D, H/P, W/P]
        x = x.flatten(2)           # [B, D, N]
        x = x.transpose(1, 2)      # [B, N, D]
        return x


# ──────────────────────────────────────────────────────────────────────
# TransformerBlock  (self-attention, used in Stage 1 spatial encoder)
# ──────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Standard pre-LayerNorm transformer encoder block:

        x ← x + Attn(LN(x))
        x ← x + FFN(LN(x))

    Parameters
    ----------
    dim      : int   — token dimension D
    n_heads  : int   — number of attention heads
    ffn_mult : int   — FFN hidden-dim multiplier (default 4 → 4D hidden)
    dropout  : float — dropout probability applied inside attention and FFN
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[B, N, D]

        Returns
        -------
        Tensor[B, N, D]
        """
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────
# CrossAttentionBlock  (used in Stage 3 cross-stream fusion)
# ──────────────────────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    Pre-LayerNorm cross-attention block:

        q ← q + CrossAttn(LN_q(q), LN_kv(kv))
        q ← q + FFN(LN(q))

    The query stream attends to the key/value stream.  Both streams are
    normalised independently before attention.

    Parameters
    ----------
    dim      : int   — token dimension D (same for both streams)
    n_heads  : int   — number of attention heads
    ffn_mult : int   — FFN hidden-dim multiplier (default 4)
    dropout  : float — dropout probability
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff      = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        q  : Tensor[B, N_q,  D] — query stream (updated in-place residually)
        kv : Tensor[B, N_kv, D] — key/value stream (read-only)

        Returns
        -------
        Tensor[B, N_q, D]
        """
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(self.norm_q(q), kv_norm, kv_norm)
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


# ──────────────────────────────────────────────────────────────────────
# MLPHead
# ──────────────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    """
    MLP regression head:  in_dim → hidden_dims[0] → ... → out_dim

    GELU activations and optional dropout between hidden layers;
    no activation on the final linear layer.

    Parameters
    ----------
    in_dim      : int
    hidden_dims : list[int]  — may be empty for a single linear layer
    out_dim     : int
    dropout     : float
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[B, in_dim]

        Returns
        -------
        Tensor[B, out_dim]
        """
        return self.net(x)

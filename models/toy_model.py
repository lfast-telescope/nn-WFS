import torch
import torch.nn as nn

try:
    from .common import PatchEmbed
except ImportError:
    from models.common import PatchEmbed

EPS_RODDIER = 1e-6


class SLPCWFS(nn.Module):
    """
    Fully-connected baseline for CWFS Zernike estimation with patch embedding.

    Embeds image patches, mean-pools tokens, then applies a hidden layer with
    150 nodes (GeLU activation + dropout) before the output layer.

    Parameters
    ----------
    img_size   : int   — spatial size H = W of input (default 256)
    patch_size : int   — side length of each square patch (default 16)
    embed_dim  : int   — token dimension after patch embedding (default 128)
    n_outputs  : int   — number of Zernike coefficients (default 36)
    dropout    : float — dropout rate for hidden layer (default 0.0)
    input_mode : str   — one of 'two_stream' (default) | 'r_stack' | 'pairs'

        two_stream : forward(I1:[B,T,H,W], I2:[B,T,H,W])
                     Computes all T² Roddier combinations, embeds, pools, then fc.
        r_stack    : forward(R:[B,T²,H,W])
                     Treats each of the T² frames as an independent sample;
                     returns [B*T², n_outputs] (labels tiled in the training loop).
        pairs      : forward(I1:[B,1,H,W], I2:[B,1,H,W], r:[B,1,H,W])
                     Embeds r, pools, applies fc (backward-compatible).
    """

    def __init__(
        self,
        img_size: int   = 256,
        patch_size: int = 16,
        embed_dim: int  = 128,
        n_outputs: int  = 36,
        dropout: float  = 0.0,
        input_mode: str = 'two_stream',
    ):
        super().__init__()
        if input_mode not in ('two_stream', 'r_stack', 'pairs'):
            raise ValueError(f"Unknown input_mode '{input_mode}'")
        self.input_mode = input_mode
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=1, embed_dim=embed_dim)
        
        # Head: hidden layer (150 nodes) + GeLU + dropout + output layer
        self.fc_hidden = nn.Linear(embed_dim, 150)
        self.gelu = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_output = nn.Linear(150, n_outputs)

    # ------------------------------------------------------------------

    def _embed_and_pool(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, H, W] → patch embed → [B, N, D] → mean pool → [B, D]"""
        tokens = self.patch_embed(x)   # [B, N, D]
        return tokens.mean(dim=1)       # [B, D]

    def _apply_head(self, pooled: torch.Tensor) -> torch.Tensor:
        """Apply hidden layer, GeLU, dropout, and output layer."""
        hidden = self.fc_hidden(pooled)
        hidden = self.gelu(hidden)
        hidden = self.dropout_layer(hidden)
        return self.fc_output(hidden)

    def _forward_pairs(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
        r:  torch.Tensor,
    ) -> torch.Tensor:
        # r: [B, 1, H, W]
        return self._apply_head(self._embed_and_pool(r))

    def _forward_two_stream(
        self,
        I1: torch.Tensor,
        I2: torch.Tensor,
    ) -> torch.Tensor:
        # I1, I2: [B, T, H, W]
        B, T, H, W = I1.shape
        I1_exp = I1.unsqueeze(2)   # [B, T, 1, H, W]
        I2_exp = I2.unsqueeze(1)   # [B, 1, T, H, W]
        R = (I1_exp - I2_exp) / (I1_exp + I2_exp + EPS_RODDIER)  # [B, T, T, H, W]
        R = R.reshape(B, T * T, H, W)                             # [B, T², H, W]
        # Embed each T² frame and pool
        R_flat = R.reshape(B * T * T, 1, H, W)                    # [B*T², 1, H, W]
        embedded = self._embed_and_pool(R_flat)                   # [B*T², D]
        pooled = embedded.reshape(B, T * T, -1).mean(dim=1)       # [B, D]
        return self._apply_head(pooled)

    def _forward_r_stack(
        self,
        R: torch.Tensor,
    ) -> torch.Tensor:
        # R: [B, T², H, W] — each frame is an independent sample
        B, TT, H, W = R.shape
        R_flat = R.reshape(B * TT, 1, H, W)                       # [B*T², 1, H, W]
        return self._apply_head(self._embed_and_pool(R_flat))     # [B*T², n_outputs]

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.input_mode == 'pairs':
            return self._forward_pairs(*args, **kwargs)
        elif self.input_mode == 'two_stream':
            return self._forward_two_stream(*args, **kwargs)
        else:
            return self._forward_r_stack(*args, **kwargs)


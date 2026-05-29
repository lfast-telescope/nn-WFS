import math
import torch

# ──────────────────────────────────────────────────────────────────────
# Per-mode and total wavefront error metrics
# ──────────────────────────────────────────────────────────────────────
# All functions accept *denormalised* Zernike coefficients in physical
# units (metres of optical path difference).  If the model was trained
# on z-scored labels, denormalise with the training-set statistics before
# calling these functions:
#
#     pred_phys = pred_norm * std + mean
#
# where std / mean come from dataset.compute_label_stats().
# ──────────────────────────────────────────────────────────────────────


def per_mode_rms(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Root-mean-square error for each Zernike mode, averaged over the batch.

    Parameters
    ----------
    pred   : Tensor[B, 14]  — predicted Zernike coefficients Z2..Z15
    target : Tensor[B, 14]  — ground-truth coefficients (same units)

    Returns
    -------
    Tensor[14]  — RMS error per mode (same units as input)
    """
    return (pred - target).pow(2).mean(dim=0).sqrt()


def total_wfe_rms(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Total wavefront error RMS, averaged over the batch.

    Assumes the Zernike basis is orthonormal over the pupil, so the total
    WFE variance equals the sum of per-mode variances (Noll 1976):

        σ_total² = Σ_j σ_j²

    Each sample's total WFE is the quadrature sum of its per-mode errors;
    the returned scalar is the mean over the batch.

    Parameters
    ----------
    pred   : Tensor[B, 14]
    target : Tensor[B, 14]

    Returns
    -------
    Scalar tensor  — mean total WFE RMS across the batch (same units as input)
    """
    diff = pred - target                           # [B, 14]
    return diff.pow(2).sum(dim=1).sqrt().mean()    # scalar


def strehl_proxy(
    wfe_rms: torch.Tensor,
    wavelength: float = 550e-9,
) -> torch.Tensor:
    """
    Maréchal approximation to the Strehl ratio:

        S ≈ exp[−(2π σ / λ)²]

    Valid for σ/λ ≲ 0.1 (diffraction-limited regime, Strehl > 0.8).
    The approximation underestimates Strehl for larger aberrations, but
    remains a useful monotone proxy for ranking model performance.

    Parameters
    ----------
    wfe_rms    : scalar tensor  — total RMS wavefront error in metres
    wavelength : float          — reference wavelength in metres (default 550 nm)

    Returns
    -------
    Scalar tensor ∈ (0, 1]
    """
    return torch.exp(
        -(2.0 * math.pi * wfe_rms / wavelength) ** 2
    )

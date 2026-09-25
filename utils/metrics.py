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


# ──────────────────────────────────────────────────────────────────────
# Zernike Radial Order Grouping
# ──────────────────────────────────────────────────────────────────────

def noll_radial_order(j: int) -> int:
    """
    Return the Zernike radial order n for Noll index j (1-based).

    In Noll indexing, modes are ordered by increasing radial order n:
        n=0: j=1 (piston)
        n=1: j=2, 3 (tip, tilt)
        n=2: j=4, 5, 6 (defocus, astigmatisms)
        n=3: j=7..10 (coma, trefoil)
        n=4: j=11..15 (spherical, secondary astigmatism, quadrafoil)
        n=5: j=16..21
        n=6: j=22..28
        n=7: j=29..36

    The radial order n is given analytically by:
        n = ceil((-3 + sqrt(1 + 8j)) / 2)
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    return int(math.ceil((-3.0 + math.sqrt(1.0 + 8.0 * j)) / 2.0))


def _format_mode_range(modes: list[int]) -> str:
    """Format a list of Noll mode indices as a compact string (e.g. 'Z4–Z6' or 'Z4, Z6')."""
    if not modes:
        return ""
    if len(modes) == 1:
        return f"Z{modes[0]}"
    if modes == list(range(modes[0], modes[-1] + 1)):
        return f"Z{modes[0]}–Z{modes[-1]}"
    return ", ".join(f"Z{m}" for m in modes)


def group_modes_by_radial_order(
    trained_modes: list[int],
    mode_rms: list[float] | tuple[float, ...] | torch.Tensor,
    scale: float = 1e9,
) -> list[dict]:
    """
    Group per-mode RMS errors by Zernike radial order n.

    Parameters
    ----------
    trained_modes : list of int
        Noll indices (1-based) corresponding to each element in mode_rms.
    mode_rms : sequence of float or Tensor
        RMS error for each mode.
    scale : float, default 1e9
        Scale factor applied to mode_rms values (1e9 converts metres to nm).

    Returns
    -------
    list of dict, sorted by order n, with keys:
        'order': int (radial order n)
        'modes': list[int] (Noll indices in this order)
        'modes_str': str (compact label, e.g. "Z4–Z6")
        'n_modes': int
        'avg': float (mean RMS in scaled units)
        'min': float (min RMS in scaled units)
        'max': float (max RMS in scaled units)
    """
    if isinstance(mode_rms, torch.Tensor):
        vals = mode_rms.detach().cpu().flatten().tolist()
    else:
        vals = list(mode_rms)

    if len(trained_modes) != len(vals):
        raise ValueError(
            f"Length mismatch: {len(trained_modes)} trained_modes vs {len(vals)} mode_rms values"
        )

    by_order: dict[int, list[tuple[int, float]]] = {}
    for mode_j, rms in zip(trained_modes, vals):
        n = noll_radial_order(mode_j)
        by_order.setdefault(n, []).append((mode_j, float(rms) * scale))

    results = []
    for n in sorted(by_order.keys()):
        pairs = by_order[n]
        modes = [p[0] for p in pairs]
        v_list = [p[1] for p in pairs]
        results.append({
            'order': n,
            'modes': modes,
            'modes_str': _format_mode_range(modes),
            'n_modes': len(modes),
            'avg': sum(v_list) / len(v_list),
            'min': min(v_list),
            'max': max(v_list),
        })
    return results


def format_order_grouped_rms(
    trained_modes: list[int],
    mode_rms: list[float] | tuple[float, ...] | torch.Tensor,
    scale: float = 1e9,
    prefix: str = "    ",
) -> list[str]:
    """
    Format per-mode RMS errors grouped by Zernike radial order n.

    Returns aligned lines showing:
        Order n=<n> (<modes>, <count>): avg= <avg>  min= <min>  max= <max>
    """
    grouped = group_modes_by_radial_order(trained_modes, mode_rms, scale=scale)
    if not grouped:
        return []

    items = []
    for g in grouped:
        m_word = "mode" if g['n_modes'] == 1 else "modes"
        label = f"Order n={g['order']} ({g['modes_str']}, {g['n_modes']} {m_word})"
        items.append((label, g))

    max_label_w = max(len(label) for label, _ in items)
    lines = []
    for label, g in items:
        lines.append(
            f"{prefix}{label:<{max_label_w}} : avg={g['avg']:5.1f}  min={g['min']:5.1f}  max={g['max']:5.1f}"
        )
    return lines


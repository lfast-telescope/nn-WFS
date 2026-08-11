import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Noll Zernike index → (radial order n, signed azimuthal frequency m)
# ──────────────────────────────────────────────────────────────────────
# Noll (1976) convention: for a given radial order n, modes are ordered so
# that even j → m >= 0 (cosine term), odd j → m <= 0 (sine term); m == 0
# modes are radially symmetric (defocus Z4, spherical Z11, ...) and are
# invariant under all D4 operations.


def noll_to_nm(j: int) -> tuple[int, int]:
    """
    Convert a 1-based Noll Zernike index to (radial order n, signed
    azimuthal frequency m).

    Standard Noll (1976) algorithm.  Verified against the historical
    hardcoded Z2-Z15 table this module used to ship with:
        Z2:(1,+1) Z3:(1,-1) Z4:(2,0) Z5:(2,-2) Z6:(2,+2) Z7:(3,-1) Z8:(3,+1)
        Z9:(3,-3) Z10:(3,+3) Z11:(4,0) Z12:(4,+2) Z13:(4,-2) Z14:(4,+4) Z15:(4,-4)
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * ((j1 + (n + 1) % 2) // 2))
    return n, m


def _find_pairs(trained_modes: list[int]) -> list[tuple[int, int, int]]:
    """
    For an ascending list of Noll indices, resolve the cos/sin partner of
    every non-singleton (m != 0) mode.

    Returns a list of (cos_position, sin_position, m) triples, where
    positions are indices into `trained_modes` (i.e. into the truncated
    label vector), and m > 0 (the pair's positive azimuthal frequency).

    Raises
    ------
    ValueError
        If a mode with m != 0 is present without its cos/sin partner also
        present in `trained_modes`.  D4 rotation mixes a (cos, sin) pair
        together, so both must be part of the trained output vector.
    """
    position = {mode: i for i, mode in enumerate(trained_modes)}
    max_j = max(trained_modes)
    nm_table = {j: noll_to_nm(j) for j in range(1, max_j + 1)}
    by_nm = {}
    for j, (n, m) in nm_table.items():
        by_nm.setdefault((n, m), j)

    pairs: list[tuple[int, int, int]] = []
    handled: set[int] = set()
    for mode in trained_modes:
        if mode in handled:
            continue
        n, m = nm_table[mode]
        if m == 0:
            handled.add(mode)
            continue
        partner = by_nm.get((n, -m))
        if partner is None or partner not in position:
            partner_label = f"Z{partner}" if partner is not None else "its partner mode"
            raise ValueError(
                f"trained_modes={trained_modes} is missing the cos/sin partner of "
                f"Z{mode} (n={n}, m={m}): {partner_label} (n={n}, m={-m}) must also "
                f"be included.  D4 rotation mixes cos/sin pairs together, so the "
                f"model may only be trained on complete pairs (radially symmetric "
                f"modes such as Z4/Z11 are exempt)."
            )
        cos_mode, sin_mode = (mode, partner) if m > 0 else (partner, mode)
        pairs.append((position[cos_mode], position[sin_mode], abs(m)))
        handled.add(mode)
        handled.add(partner)

    return pairs


def validate_trained_modes_pairing(trained_modes: list[int]) -> None:
    """
    Raise ValueError if `trained_modes` does not form complete cos/sin pairs
    for every non-singleton mode.  Call this unconditionally at config
    validation time (independent of whether augmentation is enabled) since
    the model may only be trained on pairing-complete mode sets.
    """
    _find_pairs(list(trained_modes))


# ──────────────────────────────────────────────────────────────────────
# D4 label-transformation matrix construction
# ──────────────────────────────────────────────────────────────────────

def _zernike_block_2x2(m, flip, rot_k):
    """
    2×2 matrix acting on the (cos-coeff, sin-coeff) pair of a Zernike mode
    with azimuthal order m under a given D4 image operation.

    Image operation: optionally flip-LR first, then rotate rot_k × 90° CCW.

    This maps the physical pupil angle as
        no flip : θ_in(θ_out) = θ_out − rot_k·90°
        with flip: θ_in(θ_out) = π + rot_k·90° − θ_out

    The coefficient (a, b) for  a cos(mθ) + b sin(mθ)  transforms as
        [a_new, b_new]^T = M @ [a_old, b_old]^T

    For no-flip:  M = R_m(φ) = [[ cos mφ, −sin mφ], [sin mφ,  cos mφ]]
    For flip:     M = F_m(φ) = [[ cos mψ,  sin mψ], [sin mψ, −cos mψ]]
                  where ψ = m(π + φ)

    All entries are exact integers {−1, 0, 1} for D4 multiples of 90°.
    """
    phi_deg = rot_k * 90.0

    if not flip:
        psi = np.deg2rad(m * phi_deg)
        c = int(round(float(np.cos(psi))))
        s = int(round(float(np.sin(psi))))
        return np.array([[c, -s],
                         [s,  c]], dtype=np.float32)
    else:
        # reflection composed with rotation: ψ = m * (180° + φ)
        psi = np.deg2rad(m * (180.0 + phi_deg))
        c = int(round(float(np.cos(psi))))
        s = int(round(float(np.sin(psi))))
        return np.array([[ c, s],
                         [ s, -c]], dtype=np.float32)


def build_d4_label_matrices(trained_modes: list[int]) -> list[torch.Tensor]:
    """
    Build the 8 D4 dihedral label-transformation matrices for an arbitrary
    ascending list of Noll Zernike indices `trained_modes`.

    Each matrix is sized `len(trained_modes) x len(trained_modes)` and acts
    on a label vector whose entries follow the order of `trained_modes`.

    Ordering of the 8 D4 operations:
        0 : identity
        1 : rot 90° CCW
        2 : rot 180°
        3 : rot 270° CCW
        4 : flip-LR
        5 : flip-LR → rot 90° CCW
        6 : flip-LR → rot 180°   (= flip-UD)
        7 : flip-LR → rot 270° CCW

    The flip flag and rot_k (0–3) index are encoded as:
        op_idx < 4  →  flip=False, rot_k = op_idx
        op_idx >= 4 →  flip=True,  rot_k = op_idx % 4

    Raises ValueError (via `_find_pairs`) if any non-singleton mode's
    partner is missing from `trained_modes`.
    """
    trained_modes = list(trained_modes)
    n_out = len(trained_modes)
    pairs = _find_pairs(trained_modes)

    ops = [
        (False, 0), (False, 1), (False, 2), (False, 3),
        (True,  0), (True,  1), (True,  2), (True,  3),
    ]
    matrices = []
    for (flip, rot_k) in ops:
        M = np.eye(n_out, dtype=np.float32)
        for (cos_idx, sin_idx, m) in pairs:
            block = _zernike_block_2x2(m, flip, rot_k)
            M[cos_idx, cos_idx] = block[0, 0]
            M[cos_idx, sin_idx] = block[0, 1]
            M[sin_idx, cos_idx] = block[1, 0]
            M[sin_idx, sin_idx] = block[1, 1]
        # singleton rows (radially symmetric) remain on the identity diagonal
        matrices.append(torch.from_numpy(M))
    return matrices


# ──────────────────────────────────────────────────────────────────────
# Public transform
# ──────────────────────────────────────────────────────────────────────

class D4Augment:
    """
    Randomly apply one of the 8 D4 dihedral symmetry operations to a CWFS
    sample, transforming the PSF images and the Zernike label vector
    consistently.

    Rotation and reflection of the label vector is derived analytically from
    the irradiance transport equation (Roddier 1993): under a coordinate
    rotation by φ, an azimuthal-order-m Zernike pair (cos, sin) transforms
    as a 2D rotation by m·φ; under a left-right reflection the pair transforms
    as a reflection matrix parameterised by m.

    Parameters
    ----------
    trained_modes : list[int]
        Ascending list of Noll Zernike indices the label vector represents,
        in order.  Must be pairing-complete (see `validate_trained_modes_pairing`).
    p : float
        Probability of applying *any* non-identity transform on each call.
        Default 1.0 samples uniformly from all 8 operations (including
        identity, each with probability 1/8).
    """

    def __init__(self, trained_modes: list[int], p: float = 1.0):
        self.trained_modes = list(trained_modes)
        self.p = float(p)
        self._label_matrices = build_d4_label_matrices(self.trained_modes)

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample : dict with keys 'I1', 'I2' (each Tensor[..., H, W]),
                                  optionally 'r' and/or 'R' (Tensor[..., H, W]),
                                  'labels' (Tensor[len(trained_modes)])

        Returns
        -------
        dict with the same keys, consistently transformed.
        """
        if torch.rand(1).item() >= self.p:
            return sample

        op_idx = torch.randint(0, 8, (1,)).item()
        if op_idx == 0:
            return sample   # identity — skip copies

        flip  = op_idx >= 4
        rot_k = op_idx % 4

        out = dict(sample)
        for key in ('I1', 'I2', 'r', 'R'):
            if key in sample:
                out[key] = _apply_image_op(sample[key], flip, rot_k)

        M = self._label_matrices[op_idx].to(sample['labels'].device)
        out['labels'] = M @ sample['labels']

        return out


# ──────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────

def _apply_image_op(img, flip, rot_k):
    """
    Apply a D4 image operation to a [..., H, W] tensor.

    Operation order: flip-LR first (if requested), then rotate rot_k × 90° CCW.
    This matches the angular-coordinate derivation used in _zernike_block_2x2.

    torch.rot90(img, k=1, dims=[-2,-1]) rotates 90° CCW, mapping
    physical coordinates (x, y) → (−y, x), i.e. θ → θ + 90°.
    torch.flip(img, dims=[-1]) flips LR, mapping x → −x, i.e. θ → π − θ.
    """
    if flip:
        img = torch.flip(img, dims=[-1])
    if rot_k:
        img = torch.rot90(img, k=rot_k, dims=[-2, -1])
    return img

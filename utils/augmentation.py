import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Noll Zernike mode metadata for Z2..Z15  (label vector indices 0..13)
# ──────────────────────────────────────────────────────────────────────
# Noll convention: even j → cos term (m > 0 or m = 0); odd j → sin term (m < 0)
#
# idx  Noll-j  (n, m)   angular form
# ---  ------  ------   ------------
#   0    2     (1,+1)   cos  θ      x-tilt
#   1    3     (1,-1)   sin  θ      y-tilt
#   2    4     (2, 0)   —           defocus          ← radially symmetric
#   3    5     (2,-2)   sin 2θ      oblique astig
#   4    6     (2,+2)   cos 2θ      vertical astig
#   5    7     (3,-1)   sin  θ      vertical coma
#   6    8     (3,+1)   cos  θ      horizontal coma
#   7    9     (3,-3)   sin 3θ      vertical trefoil
#   8   10     (3,+3)   cos 3θ      oblique trefoil
#   9   11     (4, 0)   —           primary spherical ← radially symmetric
#  10   12     (4,+2)   cos 2θ      2nd-order vert. astig
#  11   13     (4,-2)   sin 2θ      2nd-order obl. astig
#  12   14     (4,+4)   cos 4θ
#  13   15     (4,-4)   sin 4θ
#
# Each entry: (cos_label_idx, sin_label_idx, azimuthal_order_m)
_PAIRS = [
    ( 0,  1, 1),   # Z2  (cos θ),   Z3  (sin θ)   — m=1
    ( 6,  5, 1),   # Z8  (cos θ),   Z7  (sin θ)   — m=1
    ( 4,  3, 2),   # Z6  (cos 2θ),  Z5  (sin 2θ)  — m=2
    (10, 11, 2),   # Z12 (cos 2θ),  Z13 (sin 2θ)  — m=2
    ( 8,  7, 3),   # Z10 (cos 3θ),  Z9  (sin 3θ)  — m=3
    (12, 13, 4),   # Z14 (cos 4θ),  Z15 (sin 4θ)  — m=4
]

# Radially-symmetric modes: invariant under all D4 operations
_SINGLETONS = [2, 9]   # Z4 (defocus), Z11 (primary spherical)


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


def _build_d4_label_matrices():
    """
    Pre-compute the 8 D4 label-transformation matrices (14×14, float32).

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
    """
    ops = [
        (False, 0), (False, 1), (False, 2), (False, 3),
        (True,  0), (True,  1), (True,  2), (True,  3),
    ]
    matrices = []
    for (flip, rot_k) in ops:
        M = np.eye(14, dtype=np.float32)
        for (cos_idx, sin_idx, m) in _PAIRS:
            block = _zernike_block_2x2(m, flip, rot_k)
            M[cos_idx, cos_idx] = block[0, 0]
            M[cos_idx, sin_idx] = block[0, 1]
            M[sin_idx, cos_idx] = block[1, 0]
            M[sin_idx, sin_idx] = block[1, 1]
        # singleton rows (radially symmetric) remain on the identity diagonal
        matrices.append(M)
    return matrices


# Pre-computed at module load time (cost: 8 × 14×14 multiplications)
_D4_LABEL_MATRICES = _build_d4_label_matrices()
_D4_LABEL_TENSORS  = [torch.from_numpy(M) for M in _D4_LABEL_MATRICES]


# ──────────────────────────────────────────────────────────────────────
# Public transform
# ──────────────────────────────────────────────────────────────────────

class D4Augment:
    """
    Randomly apply one of the 8 D4 dihedral symmetry operations to a CWFS
    sample, transforming both the PSF images and the Zernike label vector
    consistently.

    The 8 operations are defined in _build_d4_label_matrices (above).
    Rotation and reflection of the label vector is derived analytically from
    the irradiance transport equation (Roddier 1993): under a coordinate
    rotation by φ, an azimuthal-order-m Zernike pair (cos, sin) transforms
    as a 2D rotation by m·φ; under a left-right reflection the pair transforms
    as a reflection matrix parameterised by m.

    Parameters
    ----------
    p : float
        Probability of applying *any* non-identity transform on each call.
        Default 7/8 gives a uniform distribution over all 8 operations
        (each op is picked with probability 1/8).  Set to 1.0 to sample
        uniformly from all 8 including identity.
    """

    def __init__(self, p=1.0):
        self.p = float(p)

    def __call__(self, sample):
        """
        Parameters
        ----------
        sample : dict with keys 'I1', 'I2', 'r'  (each Tensor[1, H, W])
                                  'labels'          (Tensor[14])

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

        I1     = _apply_image_op(sample['I1'], flip, rot_k)
        I2     = _apply_image_op(sample['I2'], flip, rot_k)
        r      = _apply_image_op(sample['r'],  flip, rot_k)
        M      = _D4_LABEL_TENSORS[op_idx].to(sample['labels'].device)
        labels = M @ sample['labels']

        return {'I1': I1, 'I2': I2, 'r': r, 'labels': labels}


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

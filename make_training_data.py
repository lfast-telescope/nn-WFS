#!/usr/bin/env python3
"""
make_training_data.py — Synthetic CWFS Training Data Generator

Produces multi-plane defocused PSF images and in-focus focal PSF images for
nonlinear curvature wavefront sensor (nlCWFS) training, following Roddier & Roddier (1993)
and Guyon (2010).

Tensor output shape
-------------------
psfs   : float16  [N, 5, T, H, W]
         Channel 0 = I_i1  (Channel I intra-focal,  +dz1)
         Channel 1 = I_i2  (Channel I extra-focal,  -dz1, rotated 180° + dtheta)
         Channel 2 = I_ii1 (Channel II intra-focal, +dz2)
         Channel 3 = I_ii2 (Channel II extra-focal, -dz2, rotated 180° + dtheta)
         Channel 4 = I_foc (Focal plane in-focus PSF, dz=0)

labels : float32  [N, n_modes]     Zernike coefficients Z1..Z{n_modes}, metres OPD
                                    indices 0–2 (Z1 piston, Z2 tip, Z3 tilt) are always zero

attributes/
         dz1_nominal, dz2_nominal, dz_asymmetry, dtheta_deg, r0,
         dz_i1, dz_i2, dz_ii1, dz_ii2, regime

Storage & Execution
-------------------
Supports two-tier data staging: workers write directly to node-local NVMe SSD (/tmp),
then atomically sync to persistent shared storage (/rental/cbender, /groups/cbender, or /xdisk/cbender).
Supports both standalone interactive execution and Slurm array job packaging.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import yaml
from scipy.ndimage import rotate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.sparse_recorder import SparseRecorder

try:
    from hcipy import (
        Apodizer,
        Configuration,
        Field,
        FraunhoferPropagator,
        InfiniteAtmosphericLayer,
        MultiLayerAtmosphere,
        Wavefront,
        make_focal_grid,
        make_las_campanas_atmospheric_layers,
        make_obstructed_circular_aperture,
        make_pupil_grid,
        make_zernike_basis,
    )
    from hcipy.atmosphere import Cn_squared_from_fried_parameter

    # Optimization 2: Precompute MFT transformation matrices and allocate intermediate buffers
    try:
        Configuration().fourier.mft.precompute_matrices = True
        Configuration().fourier.mft.allocate_intermediate = True
    except Exception:
        pass
except ImportError as e:
    sys.exit(
        f"hcipy is required to generate training data. "
        f"Install it with: pip install hcipy\n ({e})"
    )

# Monkeypatch Grid.__hash__ for Python 3.14 xxhash byte-encoding compatibility
try:
    from hcipy.field.grid import Grid
    import xxhash
    _orig_grid_hash = Grid.__hash__
    def _safe_grid_hash(self):
        try:
            return _orig_grid_hash(self)
        except TypeError:
            h = xxhash.xxh64()
            coord_sys = self._coordinate_system
            if isinstance(coord_sys, str):
                coord_sys = coord_sys.encode('utf-8')
            h.update(coord_sys)
            if self.is_regular:
                h.update(np.ascontiguousarray(self.delta))
                h.update(np.ascontiguousarray(self.dims))
                h.update(np.ascontiguousarray(self.zero))
            elif self.is_separated:
                for s in self.separated_coords:
                    h.update(np.ascontiguousarray(s))
            else:
                for p in self.points:
                    h.update(np.ascontiguousarray(p))
            return h.intdigest()
    Grid.__hash__ = _safe_grid_hash
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Config loading & Dot-access dictionary
# ──────────────────────────────────────────────────────────────────────────────

class _NS(dict):
    """Dot-access dict for nested config."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, _NS):
                self[k] = _NS(v)
            elif isinstance(v, list):
                self[k] = [_NS(item) if isinstance(item, dict) and not isinstance(item, _NS) else item for item in v]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, _NS):
            value = _NS(value)
        self[key] = value


def load_config(path: str) -> _NS:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _NS(raw)


def apply_overrides(cfg: _NS, overrides: list) -> None:
    """Apply dotted key=value overrides in-place, e.g. 'simulation.n_examples=100'."""
    for item in overrides:
        if '=' not in item:
            continue
        key_path, _, value_str = item.partition('=')
        key_path = key_path.lstrip('-')
        parts = key_path.split('.')
        node = cfg
        for part in parts[:-1]:
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                if value_str.lower() in ('true', 'yes'):
                    value = True
                elif value_str.lower() in ('false', 'no'):
                    value = False
                elif value_str.lower() in ('none', 'null'):
                    value = None
                else:
                    value = value_str
        node[parts[-1]] = value


# ──────────────────────────────────────────────────────────────────────────────
# Wavelength grid
# ──────────────────────────────────────────────────────────────────────────────

def make_wavelength_grid(cfg: _NS):
    """
    Return (wavelengths, weights) arrays.
    Sampled uniformly in wavenumber between lambda_min and lambda_max.
    """
    lmin = cfg.wavelengths.lambda_min
    lmax = cfg.wavelengths.lambda_max
    n    = cfg.wavelengths.n_wavelengths
    wavelengths = 1.0 / np.linspace(1.0 / lmax, 1.0 / lmin, n)

    scheme = str(cfg.wavelengths.weights).lower()
    if scheme == 'flat':
        weights = np.ones(n, dtype=np.float64) / n
    else:
        weights = np.ones(n, dtype=np.float64) / n
    return wavelengths, weights


# ──────────────────────────────────────────────────────────────────────────────
# Optical system builder
# ──────────────────────────────────────────────────────────────────────────────

def build_optics(cfg: _NS):
    """
    Build pupil grid, focal grid, Fraunhofer propagator, and unit defocus OPD.
    """
    OD           = cfg.optics.OD
    ID           = cfg.optics.ID
    focal_ratio  = cfg.optics.focal_ratio
    wl_ref       = cfg.optics.wavelength_ref
    q            = cfg.optics.q
    num_airy     = cfg.optics.num_airy
    n_pupil      = cfg.optics.pupil_samples
    focal_length = OD * focal_ratio

    pupil_grid = make_pupil_grid(n_pupil, OD)
    aperture   = make_obstructed_circular_aperture(OD, ID / OD)(pupil_grid)
    aperture   = np.array(aperture, dtype=np.float64)

    focal_grid = make_focal_grid(
        q, num_airy,
        spatial_resolution=wl_ref * focal_ratio,
    )
    prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

    # Unit Noll Z4 (defocus) mode from hcipy:
    basis_3      = make_zernike_basis(3, OD, pupil_grid, starting_mode=2)
    defocus_mode = np.array(basis_3[2])  # shape (N_pupil²,)

    # c4 factor: c4 = delta_z / (16.0 * (f/#)² * sqrt(3))
    c4_factor = 1.0 / (16.0 * (focal_ratio**2) * math.sqrt(3.0))

    return pupil_grid, focal_grid, prop, aperture, defocus_mode, c4_factor, focal_length


def build_zernike_basis(cfg: _NS, pupil_grid):
    """Return list of n_modes hcipy Zernike modes as ndarrays on pupil_grid."""
    n_modes = cfg.zernike.n_modes
    basis   = make_zernike_basis(n_modes, cfg.optics.OD, pupil_grid, starting_mode=1)
    return [np.array(b) for b in basis]


def draw_coefficients(cfg: _NS, rng: np.random.Generator) -> np.ndarray:
    """Draw one set of Zernike coefficients in metres OPD, shape (n_modes,)."""
    n             = cfg.zernike.n_modes
    distribution  = cfg.zernike.distribution
    normalization = str(cfg.zernike.get('amplitude_normalization', 'none')).lower()
    override      = cfg.zernike.per_mode_rms_nm

    if override is not None:
        amplitudes = np.asarray(override, dtype=np.float64) * 1e-9
    else:
        amp_rms = float(cfg.zernike.amplitude_rms)
        if normalization == 'radial_order':
            factors = np.array([
                1.0 / max(1, int(math.ceil((-3.0 + math.sqrt(1.0 + 8.0 * j)) / 2.0)))
                for j in range(1, n + 1)
            ], dtype=np.float64)
            amplitudes = amp_rms * factors
        else:
            amplitudes = np.full(n, amp_rms, dtype=np.float64)

    if distribution == 'gaussian':
        coeffs = rng.normal(loc=0.0, scale=amplitudes, size=n)
    elif distribution == 'uniform':
        half_width = amplitudes * math.sqrt(3.0)
        coeffs = rng.uniform(low=-half_width, high=half_width, size=n)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Tip, tilt, piston are uncorrectable / always zeroed
    coeffs[0] = 0.0
    if n > 1:
        coeffs[1] = 0.0
    if n > 2:
        coeffs[2] = 0.0
    return coeffs


# ──────────────────────────────────────────────────────────────────────────────
# Modular Defocus & Tolerancing parameter sampler
# ──────────────────────────────────────────────────────────────────────────────

def get_defocus_distances(cfg: _NS) -> list[float]:
    """Return list of nominal defocus distances (metres) from config."""
    if 'defocus_distances_m' in cfg.optics:
        distances = [float(d) for d in cfg.optics.defocus_distances_m]
    elif 'delta_z1' in cfg.optics:
        dz1 = float(cfg.optics.delta_z1)
        dz2 = float(cfg.optics.get('delta_z2', 2.0 * dz1))
        distances = [dz1, dz2]
    else:
        dz = float(cfg.optics.get('delta_z', 0.369e-3))
        distances = [dz, 2.0 * dz]
    return distances


def get_channel_metadata(cfg: _NS) -> tuple[list[str], list[str], list[float], bool]:
    """
    Return (channel_names, metric_tags, nominal_distances, include_focal).
    
    Channel names follow the requested format:
      [Ii1, -Ii1, Iii1, -Iii1, Iiii1, -Iiii1, ..., Ifocal]
    Metric tags follow signed metric notation:
      ['+0.369mm', '-0.369mm', ..., '0.000mm']
    """
    distances = get_defocus_distances(cfg)
    include_focal = bool(cfg.optics.get('include_focal', True))

    roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    channel_names = []
    metric_tags = []

    for k, d in enumerate(distances):
        r = roman_numerals[k] if k < len(roman_numerals) else f"p{k+1}"
        channel_names.append(f"I{r}1")
        channel_names.append(f"-I{r}1")
        d_mm = d * 1e3
        metric_tags.append(f"+{d_mm:.3f}mm")
        metric_tags.append(f"-{d_mm:.3f}mm")

    if include_focal:
        channel_names.append("Ifocal")
        metric_tags.append("0.000mm")

    return channel_names, metric_tags, distances, include_focal


def sample_tolerancing_parameters(cfg: _NS, rng: np.random.Generator, c4_factor: float) -> dict:
    """
    Sample physical optical and atmospheric tolerancing parameters for one example.
    Supports arbitrary defocus planes and absolute ±40 um uncertainty.
    """
    distances = get_defocus_distances(cfg)
    include_focal = bool(cfg.optics.get('include_focal', True))
    tol_cfg = cfg.get('tolerancing', {})
    is_enabled = tol_cfg.get('enabled', False)

    dz_unc = float(tol_cfg.get('dz_uncertainty_m', 40.0e-6))
    dtheta_max = float(tol_cfg.get('dtheta', {}).get('max_deg', 2.0))
    seeing_cfg = tol_cfg.get('seeing', {})
    r0_min = float(seeing_cfg.get('r0_min', 0.08))
    r0_max = float(seeing_cfg.get('r0_max', 0.16))

    if not is_enabled:
        asym = np.zeros(len(distances), dtype=np.float64)
        dtheta = 0.0
        r0 = float(cfg.atmosphere.r0_500nm)
        regime = "nominal"
    else:
        # Absolute uncertainty: uniform [-dz_unc, +dz_unc] per plane
        asym = rng.uniform(-dz_unc, +dz_unc, size=len(distances))
        dtheta = rng.uniform(-dtheta_max, +dtheta_max)
        r0 = rng.uniform(r0_min, r0_max)
        regime = "toleranced"

    sampled_dz = []
    sampled_c4 = []

    for k, d_nom in enumerate(distances):
        dz_intra = float(d_nom + 0.5 * asym[k])
        dz_extra = float(d_nom - 0.5 * asym[k])
        sampled_dz.append(dz_intra)
        sampled_dz.append(dz_extra)
        sampled_c4.append(dz_intra * c4_factor)
        sampled_c4.append(dz_extra * c4_factor)

    if include_focal:
        sampled_dz.append(0.0)
        sampled_c4.append(0.0)

    res = {
        'nominal_distances': distances,
        'dz_asymmetry': [float(a) for a in asym],
        'dtheta_deg': float(dtheta),
        'r0': float(r0),
        'sampled_dz': sampled_dz,
        'sampled_c4': sampled_c4,
        'regime': regime,
    }

    # Backward compatibility keys for 2-distance configs:
    if len(distances) >= 2:
        res['dz1_nominal']  = float(distances[0])
        res['dz2_nominal']  = float(distances[1])
        res['dz_asymmetry'] = float(asym[0])
        res['dz_i1']        = sampled_dz[0]
        res['dz_i2']        = sampled_dz[1]
        res['dz_ii1']       = sampled_dz[2]
        res['dz_ii2']       = sampled_dz[3]
        res['c4_i1']        = sampled_c4[0]
        res['c4_i2']        = sampled_c4[1]
        res['c4_ii1']       = sampled_c4[2]
        res['c4_ii2']       = sampled_c4[3]

    return res


# ──────────────────────────────────────────────────────────────────────────────
# Atmospheric turbulence builder
# ──────────────────────────────────────────────────────────────────────────────

def build_atmosphere(cfg: _NS, pupil_grid, seed: int):
    """Construct reference hcipy atmospheric object, or return None if disabled."""
    if not cfg.atmosphere.enabled:
        return None

    r0_ref = float(cfg.atmosphere.r0_500nm)
    wl_ref = float(cfg.optics.wavelength_ref)

    if cfg.atmosphere.use_las_campanas:
        cn_squared = Cn_squared_from_fried_parameter(r0_ref, wl_ref)
        layers = make_las_campanas_atmospheric_layers(
            pupil_grid,
            cn_squared=cn_squared,
            outer_scale=cfg.atmosphere.L0,
        )
        atm = MultiLayerAtmosphere(layers, scintillation=False)
    else:
        sl = cfg.atmosphere.single_layer
        v = sl.wind_speed
        theta = sl.wind_direction
        velocity = [v * math.cos(theta), v * math.sin(theta)]
        layer = InfiniteAtmosphericLayer(
            pupil_grid, sl.Cn2, sl.L0, velocity, seed=seed
        )
        atm = MultiLayerAtmosphere([layer], scintillation=False)

    atm.t = 0.0
    return atm


def reset_atm_seed(atm):
    """Advance each layer to a new independent realization and reset t to 0."""
    if atm is None:
        return None
    for layer in atm.layers:
        layer.reset(make_independent_realization=True)
    atm._t = 0.0
    return atm


def get_atm_opd(atmosphere, t: float, wavelength_ref: float, atm_scale: float = 1.0):
    """Return atmospheric OPD (metres) scaled by Kolmogorov factor (r0_ref/r0)^(5/6)."""
    if atmosphere is None:
        return None
    atmosphere.t = t
    phase_rad = np.array(atmosphere.phase_for(wavelength_ref))
    opd = phase_rad * wavelength_ref / (2.0 * math.pi)
    if atm_scale != 1.0:
        opd = opd * atm_scale
    return opd


# ──────────────────────────────────────────────────────────────────────────────
# Polychromatic propagation
# ──────────────────────────────────────────────────────────────────────────────

def _centre_crop_or_pad(arr: np.ndarray, target: int) -> np.ndarray:
    """Centre-crop or zero-pad a 2-D array to (target, target)."""
    h, w = arr.shape
    if h == target and w == target:
        return arr

    if h < target or w < target:
        ph = max(0, target - h)
        pw = max(0, target - w)
        arr = np.pad(arr, ((ph // 2, ph - ph // 2),
                           (pw // 2, pw - pw // 2)))
        h, w = arr.shape

    r0 = (h - target) // 2
    c0 = (w - target) // 2
    return arr[r0:r0 + target, c0:c0 + target]


def propagate_polychromatic(
    mirror_opd:   np.ndarray,
    defocus_sign: float,
    defocus_opd:  np.ndarray,
    c4_defocus:   float,
    cfg:          _NS,
    aperture:     np.ndarray,
    prop:         FraunhoferPropagator,
    pupil_grid,
    wavelengths:  np.ndarray,
    weights:      np.ndarray,
    img_size:     int,
    atm,
    atm_scale:    float = 1.0,
) -> np.ndarray:
    """
    Propagate polychromatic wavefront through atmosphere to detector plane.
    """
    wl_ref   = cfg.optics.wavelength_ref
    n_frames = cfg.simulation.t_frames
    t_frame  = 1.0 / cfg.simulation.frame_rate
    pixel_oversample = cfg.optics.pixel_oversample
    raw_size = img_size * pixel_oversample

    tau_0 = None
    if cfg.atmosphere.enabled and atm is not None:
        v_wind = cfg.atmosphere.single_layer.wind_speed if not cfg.atmosphere.use_las_campanas else 10.0
        tau_0  = 0.31 * float(cfg.atmosphere.r0_500nm) / v_wind

    n_sub = max(1, round(t_frame / tau_0)) if tau_0 is not None else 1

    static_opd = mirror_opd + (defocus_sign * c4_defocus * defocus_opd if c4_defocus != 0.0 else 0.0)
    frames = np.zeros((n_frames, raw_size, raw_size), dtype=np.float64)
    Ngrid_foc = None

    for frame_i in range(n_frames):
        broadband_integrated = np.zeros((raw_size, raw_size), dtype=np.float64)
        for sub_j in range(n_sub):
            t_sample  = 2 * frame_i * t_frame + sub_j * tau_0 if tau_0 is not None else 0.0 #add factor 2 to ensure decorrelation
            atm_opd   = get_atm_opd(atm, t_sample, wl_ref, atm_scale=atm_scale)
            total_opd = static_opd + atm_opd if atm_opd is not None else static_opd

            broadband_sub = np.zeros((raw_size, raw_size), dtype=np.float64)
            for wl, wt in zip(wavelengths, weights):
                phase     = (2.0 * math.pi / wl) * total_opd
                amplitude = aperture * np.exp(1j * phase)
                wf        = Wavefront(Field(amplitude.astype(np.complex128), pupil_grid), wl)
                psf_field = prop.forward(wf).power
                if Ngrid_foc is None:
                    Ngrid_foc = int(round(math.sqrt(len(psf_field))))
                psf_2d = np.array(psf_field).reshape(Ngrid_foc, Ngrid_foc)
                broadband_sub += wt * _centre_crop_or_pad(psf_2d, raw_size)
            broadband_integrated += broadband_sub

        frames[frame_i] = broadband_integrated

    # Bin sub-pixels into detector pixels:
    binned_frames = frames.reshape(
        n_frames, img_size, pixel_oversample, img_size, pixel_oversample
    ).mean(axis=(2, 4))
    total = binned_frames.sum()
    if total > 0:
        binned_frames /= total
    return binned_frames


def _apply_rotation(seq: np.ndarray, dtheta_deg: float) -> np.ndarray:
    """
    Rotate extra-focal sequence by 180° + dtheta_deg around center.
    Preserves total normalized flux.
    """
    total_rot = 180.0 + dtheta_deg
    if abs(dtheta_deg) < 1e-6:
        return np.rot90(seq, k=2, axes=(1, 2))
    
    rotated = rotate(seq, total_rot, axes=(1, 2), reshape=False, order=3, mode='constant', cval=0.0)
    # Re-normalize flux to prevent interpolation losses
    orig_sum = seq.sum()
    rot_sum = rotated.sum()
    if rot_sum > 0:
        rotated = rotated * (orig_sum / rot_sum)
    return rotated


def save_verification_plot(
    psfs_frame0: np.ndarray,
    channel_names: list[str],
    metric_tags: list[str],
    ex_idx: int,
    n_examples: int,
    ex_time: float,
    out_path: str | Path,
) -> None:
    """
    Save multi-plane verification PNG plot for the current example.

    Matches the format of modular_dry_run_verification.png with 2 rows of subplots,
    square-root intensity stretch, magma colormap, and channel titles with metric tags.
    Overwrites out_path on each invocation.
    """
    clean_ch_names = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in channel_names]
    clean_metric_tags = [m.decode('utf-8') if isinstance(m, bytes) else str(m) for m in metric_tags]
    n_channels = len(clean_ch_names)
    n_cols = max(1, math.ceil(n_channels / 2))

    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_channels):
        img = psfs_frame0[i].astype(np.float32)
        axes[i].imshow(np.sqrt(np.maximum(img, 0.0)), cmap='magma', origin='lower')
        axes[i].set_title(f"{clean_ch_names[i]}\n({clean_metric_tags[i]})", fontsize=10)
        axes[i].axis('off')

    for j in range(n_channels, len(axes)):
        axes[j].axis('off')

    plt.suptitle(
        f"Modular Multi-Plane CWFS Verification — Example {ex_idx + 1}/{n_examples} "
        f"(Gen Time: {ex_time:.2f}s, Sqrt Scale)",
        fontsize=13,
    )
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────

def main(
    cfg: _NS,
    task_id: int = 0,
    n_tasks: int = 1,
    output_path: Optional[str] = None,
    dry_run: bool = False,
):
    n_examples_total = int(cfg.simulation.n_examples)
    t_frames         = int(cfg.simulation.t_frames)
    img_size         = int(cfg.simulation.img_size)
    n_modes          = int(cfg.zernike.n_modes)
    chunk            = int(cfg.simulation.hdf5_chunk_size)
    seed_base        = int(cfg.simulation.random_seed) + task_id * 10000

    output_tmp_plots = bool(cfg.simulation.get('output_tmp_plots', cfg.get('output_tmp_plots', False)))
    tmp_plot_path_cfg = cfg.simulation.get('tmp_plot_path', cfg.get('tmp_plot_path', None))
    if tmp_plot_path_cfg is not None:
        tmp_plot_path = Path(tmp_plot_path_cfg)
    else:
        prefix = "modular_dry_run_verification" if dry_run else "modular_verification"
        plot_name = f"{prefix}.png" if n_tasks == 1 else f"{prefix}_task{task_id:03d}.png"
        tmp_plot_path = Path("tmp") / plot_name

    # Number of examples for this task/shard
    n_examples = n_examples_total // n_tasks if n_tasks > 1 else n_examples_total
    if dry_run:
        n_examples = 2
        print("DRY RUN: generating 2 examples only, verifying shapes.")

    # Two-tier storage path resolution
    storage_cfg  = cfg.get('storage', {})
    staging_dir  = storage_cfg.get('staging_dir', None)
    target_root  = storage_cfg.get('target_root', '/rental/cbender')
    sub_dir      = storage_cfg.get('sub_dir', 'cwfs_shards')
    cleanup_tmp  = storage_cfg.get('cleanup_staging', True)

    shard_filename = f"cwfs_shard_task{task_id:03d}.h5" if n_tasks > 1 else "cwfs_synthetic.h5"

    if output_path is not None:
        final_target_path = Path(output_path).resolve()
        use_staging = False
        local_write_path = final_target_path
    elif staging_dir and not dry_run:
        user = os.environ.get("USER", "hpc_user")
        job_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", str(int(time.time()))))
        scratch_dir = Path(staging_dir) / user / f"cwfs_{job_id}"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        local_write_path = scratch_dir / shard_filename
        final_target_path = Path(target_root) / sub_dir / shard_filename
        final_target_path.parent.mkdir(parents=True, exist_ok=True)
        use_staging = True
    else:
        final_target_path = Path(target_root) / sub_dir / shard_filename
        final_target_path.parent.mkdir(parents=True, exist_ok=True)
        local_write_path = final_target_path
        use_staging = False

    channel_names, metric_tags, nominal_distances, include_focal = get_channel_metadata(cfg)
    n_channels = len(channel_names)

    print(f"\n{'='*70}")
    print(f"CWFS DATA GENERATION (Task {task_id + 1}/{n_tasks})")
    print(f"{'='*70}")
    print(f"Target Examples  : {n_examples:,}")
    print(f"Output Channels  : {n_channels} ({', '.join(channel_names)})")
    print(f"Metric Tags      : {', '.join(metric_tags)}")
    print(f"Frames / Example : {t_frames}")
    print(f"Resolution       : {img_size}×{img_size} px")
    print(f"Random Seed      : {seed_base}")
    print(f"Local Write Path : {local_write_path}")
    if output_tmp_plots:
        print(f"Output Tmp Plots : Enabled -> {tmp_plot_path} (overwritten each example)")
    if use_staging:
        print(f"Final Target Path: {final_target_path}")
    print(f"{'='*70}\n")

    print("Building optical system...")
    (pupil_grid, focal_grid, prop,
     aperture, defocus_opd_unit,
     c4_factor, focal_length) = build_optics(cfg)

    print("Building Zernike basis & atmosphere...")
    zernike_basis = build_zernike_basis(cfg, pupil_grid)
    wavelengths, weights = make_wavelength_grid(cfg)
    rng = np.random.default_rng(seed_base)

    r0_ref = float(cfg.atmosphere.r0_500nm)
    atm = build_atmosphere(cfg, pupil_grid, seed=seed_base)

    recorder = None
    if not dry_run and not getattr(cfg, 'no_sparse_record', False):
        sparse_dir = getattr(cfg, 'sparse_dir', None)
        recorder = SparseRecorder(task_name=f"datagen_task{task_id:03d}", output_dir=sparse_dir, config=dict(cfg))

    local_write_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(local_write_path, 'w') as f:
            # Multi-channel PSF dataset: [N, n_channels, T, H, W]
            ds_psfs = f.create_dataset(
                'psfs',
                shape=(n_examples, n_channels, t_frames, img_size, img_size),
                dtype='float16',
                chunks=(1, n_channels, t_frames, img_size, img_size),
                compression='gzip', compression_opts=4,
            )
            ds_psfs.attrs['channels'] = [c.encode('utf-8') for c in channel_names]
            ds_psfs.attrs['metric_tags'] = [m.encode('utf-8') for m in metric_tags]
            ds_psfs.attrs['channel_indices'] = list(range(n_channels))
            ds_psfs.attrs['nominal_defocus_m'] = np.array(nominal_distances, dtype='float32')
            ds_psfs.attrs['include_focal'] = include_focal
            ds_psfs.attrs['has_focal_psf'] = include_focal

            # Labels dataset: [N, n_modes]
            ds_labels = f.create_dataset(
                'labels',
                shape=(n_examples, n_modes),
                dtype='float32',
                chunks=(max(1, min(64, n_examples)), n_modes),
            )
            ds_labels.attrs['label_units'] = cfg.output.label_units
            ds_labels.attrs['noll_start']  = 1
            ds_labels.attrs['n_modes']     = n_modes
            f.attrs['config'] = json.dumps(dict(cfg), default=str)

            # /attributes group for physical tolerancing & regime tracking
            grp_attr = f.create_group('attributes')
            ds_nom_dz   = grp_attr.create_dataset('nominal_defocus_m', data=np.array(nominal_distances, dtype='float32'))
            ds_samp_dz  = grp_attr.create_dataset('sampled_defocus_m', shape=(n_examples, n_channels), dtype='float32')
            ds_samp_c4  = grp_attr.create_dataset('sampled_c4_m', shape=(n_examples, n_channels), dtype='float32')
            ds_dtheta   = grp_attr.create_dataset('dtheta_deg', shape=(n_examples,), dtype='float32')
            ds_r0       = grp_attr.create_dataset('r0', shape=(n_examples,), dtype='float32')
            ds_regime   = grp_attr.create_dataset('regime', shape=(n_examples,), dtype=h5py.string_dtype(encoding='utf-8'))

            # Backward compatibility attributes for legacy 2-plane readers
            if len(nominal_distances) >= 2:
                ds_dz1_nom = grp_attr.create_dataset('dz1_nominal', shape=(n_examples,), dtype='float32')
                ds_dz2_nom = grp_attr.create_dataset('dz2_nominal', shape=(n_examples,), dtype='float32')
                ds_asym    = grp_attr.create_dataset('dz_asymmetry', shape=(n_examples,), dtype='float32')
                ds_dzi1    = grp_attr.create_dataset('dz_i1', shape=(n_examples,), dtype='float32')
                ds_dzi2    = grp_attr.create_dataset('dz_i2', shape=(n_examples,), dtype='float32')
                ds_dzii1   = grp_attr.create_dataset('dz_ii1', shape=(n_examples,), dtype='float32')
                ds_dzii2   = grp_attr.create_dataset('dz_ii2', shape=(n_examples,), dtype='float32')

            t0 = time.time()
            last_time = time.time()

            for ex_idx in range(n_examples):
                t_ex_start = time.time()
                labels_ex = draw_coefficients(cfg, rng)
                mirror_opd = sum(float(c) * m for c, m in zip(labels_ex, zernike_basis))

                # Tolerancing parameters
                params = sample_tolerancing_parameters(cfg, rng, c4_factor)
                atm_scale = (r0_ref / params['r0'])**(5.0 / 6.0)

                # Iterate through each defocus distance pair (intra, extra)
                ch_idx = 0
                for k in range(len(nominal_distances)):
                    c4_intra = params['sampled_c4'][2 * k]
                    c4_extra = params['sampled_c4'][2 * k + 1]

                    # Intra-focal channel: +c4
                    atm = reset_atm_seed(atm)
                    I_intra = propagate_polychromatic(
                        mirror_opd, +1.0, defocus_opd_unit, c4_intra,
                        cfg, aperture, prop, pupil_grid,
                        wavelengths, weights, img_size,
                        atm, atm_scale=atm_scale,
                    )
                    ds_psfs[ex_idx, ch_idx] = I_intra.astype(np.float16)
                    ch_idx += 1

                    # Extra-focal channel: -c4 (rotated 180° + dtheta)
                    atm = reset_atm_seed(atm)
                    I_extra_raw = propagate_polychromatic(
                        mirror_opd, -1.0, defocus_opd_unit, c4_extra,
                        cfg, aperture, prop, pupil_grid,
                        wavelengths, weights, img_size,
                        atm, atm_scale=atm_scale,
                    )
                    I_extra = _apply_rotation(I_extra_raw, params['dtheta_deg'])
                    ds_psfs[ex_idx, ch_idx] = I_extra.astype(np.float16)
                    ch_idx += 1

                # In-focus channel if included (c4 = 0.0)
                if include_focal:
                    atm = reset_atm_seed(atm)
                    I_focal = propagate_polychromatic(
                        mirror_opd, 0.0, defocus_opd_unit, 0.0,
                        cfg, aperture, prop, pupil_grid,
                        wavelengths, weights, img_size,
                        atm, atm_scale=atm_scale,
                    )
                    ds_psfs[ex_idx, ch_idx] = I_focal.astype(np.float16)
                    ch_idx += 1

                ds_labels[ex_idx] = labels_ex.astype(np.float32)

                ds_samp_dz[ex_idx] = params['sampled_dz']
                ds_samp_c4[ex_idx] = params['sampled_c4']
                ds_dtheta[ex_idx]  = params['dtheta_deg']
                ds_r0[ex_idx]      = params['r0']
                ds_regime[ex_idx]  = params['regime']

                if len(nominal_distances) >= 2:
                    ds_dz1_nom[ex_idx] = params['dz1_nominal']
                    ds_dz2_nom[ex_idx] = params['dz2_nominal']
                    ds_asym[ex_idx]    = params['dz_asymmetry']
                    ds_dzi1[ex_idx]    = params['dz_i1']
                    ds_dzi2[ex_idx]    = params['dz_i2']
                    ds_dzii1[ex_idx]   = params['dz_ii1']
                    ds_dzii2[ex_idx]   = params['dz_ii2']

                ex_time = time.time() - t_ex_start

                if output_tmp_plots:
                    save_verification_plot(
                        psfs_frame0=ds_psfs[ex_idx, :, 0],
                        channel_names=channel_names,
                        metric_tags=metric_tags,
                        ex_idx=ex_idx,
                        n_examples=n_examples,
                        ex_time=ex_time,
                        out_path=tmp_plot_path,
                    )

                if (ex_idx + 1) % 5 == 0 or ex_idx == n_examples - 1:
                    elapsed = time.time() - t0
                    done = ex_idx + 1
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_sec = (n_examples - done) / rate if rate > 0 else 0
                    eta_td = timedelta(seconds=int(eta_sec))
                    print(f"  [{done:>{len(str(n_examples))}}/{n_examples}] "
                          f"{int(elapsed):>4}s | {rate:.2f} ex/s | ETA: {str(eta_td)}")

                    if recorder is not None and ((done % 20 == 0) or (done == n_examples)):
                        recorder.record_step(
                            step=f"{done}/{n_examples}",
                            metrics={"rate_ex_per_sec": rate, "eta_sec": eta_sec},
                            phase="gen",
                            step_name="example",
                            elapsed_s=int(elapsed),
                        )

        print(f"\nLocal Generation Complete: {local_write_path}")
        if recorder is not None:
            recorder.log_message(f"Task {task_id} generated {n_examples} examples.")
            recorder.close(status="COMPLETED")

        # Two-tier staging: move to shared target
        if use_staging:
            print(f"Staging to persistent storage: {final_target_path} ...")
            tmp_final = final_target_path.with_suffix(".tmp")
            shutil.copy2(local_write_path, tmp_final)
            tmp_final.replace(final_target_path)
            print(f"Successfully staged to {final_target_path}")

            if cleanup_tmp:
                try:
                    local_write_path.unlink(missing_ok=True)
                    if local_write_path.parent.exists() and not list(local_write_path.parent.iterdir()):
                        local_write_path.parent.rmdir()
                    print("Local staging cleanup complete.")
                except Exception as e:
                    print(f"[WARN] Staging cleanup warning: {e}")

    except Exception as e:
        if recorder is not None:
            recorder.log_message(f"[ERROR] Data generation aborted: {e}")
            recorder.close(status=f"FAILED ({type(e).__name__})")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic 5-channel CWFS training data with tolerancing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', required=True,
                        help='Path to data_generation.yaml.')
    parser.add_argument('--task_id', type=int, default=0,
                        help='Slurm array task ID (0-indexed, default: 0).')
    parser.add_argument('--n_tasks', type=int, default=1,
                        help='Total number of tasks in the array (default: 1).')
    parser.add_argument('--n_examples', type=int, default=None,
                        help='Override total number of examples to generate.')
    parser.add_argument('--output', default=None,
                        help='Override output HDF5 path.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate 2 examples only to verify shapes.')
    parser.add_argument('--sparse_dir', default=None,
                        help='Directory for sparse HPC logs.')
    parser.add_argument('--no_sparse_record', action='store_true',
                        help='Disable sparse HPC logging.')
    parser.add_argument('--output_tmp_plots', action='store_true', default=None,
                        help='Save verification plot PNG at the conclusion of each example.')
    parser.add_argument('--tmp_plot_path', default=None,
                        help='Override path for temporary verification plot PNG.')

    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, overrides)

    if args.n_examples is not None:
        cfg.simulation.n_examples = args.n_examples
    if args.sparse_dir:
        cfg.sparse_dir = args.sparse_dir
    if args.no_sparse_record:
        cfg.no_sparse_record = True
    if args.output_tmp_plots is not None:
        cfg.simulation.output_tmp_plots = args.output_tmp_plots
    if args.tmp_plot_path is not None:
        cfg.simulation.tmp_plot_path = args.tmp_plot_path

    main(
        cfg=cfg,
        task_id=args.task_id,
        n_tasks=args.n_tasks,
        output_path=args.output,
        dry_run=args.dry_run,
    )

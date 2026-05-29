"""
make_training_data.py — Synthetic CWFS Training Data Generator

Produces paired defocused PSF images (I1 intra-focal, I2 extra-focal) for
curvature wavefront sensor (CWFS) training, following Roddier & Roddier (1993).

Method: Option B — λ/D focal grid (single FraunhoferPropagator anchored at the
reference wavelength).  For each wavelength, the PSF is propagated on the λ/D
grid and then rescaled by (λ / λ_ref) via scipy.ndimage.zoom before summing into
the broadband image.  This correctly reproduces the wavelength-dependent physical
PSF size on a real detector with fixed pixel pitch.

Output HDF5 schema
------------------
psfs   : float16  [N, 2, H, W]   channel 0 = I1 (intra-focal), 1 = I2 (extra-focal)
labels : float32  [N, n_modes]   Zernike coefficients Z2..Z(n_modes+1), metres OPD
Attributes on 'labels': label_units = 'metres_opd'

Usage
-----
python make_training_data.py --config config/data_generation.yaml [--output path.h5] [--dry-run]

Override any config value with --section.key=value:
    python make_training_data.py --config config/data_generation.yaml --simulation.n_examples=100
"""

from __future__ import annotations

import argparse
import math
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import yaml
from scipy.ndimage import zoom

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from hcipy import (
        Apodizer,
        Field,
        FraunhoferPropagator,
        InfiniteAtmosphericLayer,
        MultiLayerAtmosphere,
        Wavefront,
        evaluate_supersampled,
        make_circular_aperture,
        make_focal_grid,
        make_las_campanas_atmospheric_layers,
        make_obstructed_circular_aperture,
        make_pupil_grid,
        make_zernike_basis,
    )
except ImportError as e:
    sys.exit(
        f"hcipy is required to generate training data.  "
        f"Install it with:  pip install hcipy\n  ({e})"
    )

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        total = kwargs.get('total', '?')
        for i, x in enumerate(it):
            print(f"\r  {i+1}/{total}", end='', flush=True)
            yield x
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

class _NS(dict):
    """Dot-access dict for nested config."""
    def __getattr__(self, key):
        try:
            val = self[key]
            return _NS(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
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
                value = value_str
        node[parts[-1]] = value


# ──────────────────────────────────────────────────────────────────────────────
# Wavelength grid
# ──────────────────────────────────────────────────────────────────────────────

def make_wavelength_grid(cfg: _NS):
    """
    Return (wavelengths, weights) arrays.

    Wavelengths are sampled uniformly in wavenumber between lambda_min and
    lambda_max (equivalent to uniform photon-energy spacing).
    Weights are normalised to sum to 1.
    """
    lmin = cfg.wavelengths.lambda_min
    lmax = cfg.wavelengths.lambda_max
    n    = cfg.wavelengths.n_wavelengths

    wavelengths = 1.0 / np.linspace(1.0 / lmax, 1.0 / lmin, n)

    scheme = str(cfg.wavelengths.weights).lower()
    if scheme != 'flat':
        print(f"Warning: unknown weight scheme '{scheme}', falling back to flat.")
    weights = np.ones(n, dtype=np.float14)
    weights /= weights.sum()
    return wavelengths, weights


# ──────────────────────────────────────────────────────────────────────────────
# Optical system
# ──────────────────────────────────────────────────────────────────────────────

def build_optics(cfg: _NS):
    """
    Construct hcipy optical system objects.

    Returns
    -------
    pupil_grid        : CartesianGrid
    focal_grid        : CartesianGrid  (λ_ref/D units, anchored at wavelength_ref)
    prop              : FraunhoferPropagator
    aperture          : ndarray        (annular aperture amplitude, shape N_pupil²)
    defocus_opd_unit  : ndarray        (Noll Z4, unit amplitude, metres; shape N_pupil²)
    c4_defocus        : float          (Noll Z4 coefficient for delta_z, metres)
    focal_length      : float
    """
    OD           = cfg.optics.OD
    ID           = cfg.optics.ID
    focal_ratio  = cfg.optics.focal_ratio
    wl_ref       = cfg.optics.wavelength_ref
    q            = cfg.optics.q
    num_airy     = cfg.optics.num_airy
    n_pupil      = cfg.optics.pupil_samples
    focal_length = OD * focal_ratio
    delta_z      = cfg.optics.delta_z

    pupil_grid = make_pupil_grid(n_pupil, OD)
    aperture   = make_obstructed_circular_aperture(OD, ID / OD)(pupil_grid)
    aperture   = np.array(aperture, dtype=np.float64)

    focal_grid = make_focal_grid(
        q, num_airy,
        spatial_resolution=wl_ref / OD,
    )
    prop = FraunhoferPropagator(pupil_grid, focal_grid, focal_length=focal_length)

    # Noll Z4 (defocus) mode from hcipy.
    # make_zernike_basis(n, D, grid, starting_mode=2) → Z2, Z3, Z4, ...
    # Index 2 = Z4 (defocus).  D = aperture diameter in same units as grid.
    basis_3      = make_zernike_basis(3, OD, pupil_grid, starting_mode=2)
    defocus_mode = np.array(basis_3[2])          # shape (N_pupil²,)

    # c4 = delta_z / (8 * (f/#)² * sqrt(3))   [metres OPD, Noll convention]
    c4_defocus = delta_z / (8.0 * focal_ratio**2 * math.sqrt(3))

    return pupil_grid, focal_grid, prop, aperture, defocus_mode, c4_defocus, focal_length


def build_zernike_basis(cfg: _NS, pupil_grid):
    """
    Return list of n_modes hcipy Zernike modes as ndarrays on pupil_grid.
    Modes are Z2..Z(n_modes+1) in Noll ordering (piston Z1 excluded).
    """
    n_modes = cfg.zernike.n_modes
    basis   = make_zernike_basis(n_modes, cfg.optics.OD, pupil_grid, starting_mode=2)
    modes   = [np.array(b) for b in basis]

    print(f"  Zernike basis: {n_modes} modes Z2–Z{n_modes+1}, "
          f"mode[0] (tip) peak={modes[0].max():.3f}, "
          f"mode[2] (defocus) peak={modes[2].max():.3f}")
    return modes


# ──────────────────────────────────────────────────────────────────────────────
# Zernike coefficient sampling
# ──────────────────────────────────────────────────────────────────────────────

def make_amplitude_array(cfg: _NS) -> np.ndarray:
    """Return per-mode RMS amplitudes in metres OPD, shape (n_modes,)."""
    n        = cfg.zernike.n_modes
    override = cfg.zernike.per_mode_rms_nm

    if override is not None:
        arr = np.asarray(override, dtype=np.float64) * 1e-9
        if len(arr) != n:
            raise ValueError(
                f"per_mode_rms_nm has {len(arr)} entries but n_modes={n}"
            )
        return arr

    scalar = cfg.zernike.amplitude_rms   # metres
    return np.full(n, scalar, dtype=np.float64)


def draw_coefficients(amplitudes: np.ndarray, distribution: str,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Draw one set of Zernike coefficients in metres OPD.

    Parameters
    ----------
    amplitudes   : ndarray[n_modes]  — per-mode RMS in metres
    distribution : 'gaussian' | 'uniform'
    rng          : numpy Generator
    """
    n = len(amplitudes)
    if distribution == 'gaussian':
        return rng.standard_normal(n) * amplitudes
    elif distribution == 'uniform':
        half = amplitudes * math.sqrt(3)
        return rng.uniform(-half, half)
    else:
        raise ValueError(f"Unknown distribution '{distribution}'. Use 'gaussian' or 'uniform'.")


# ──────────────────────────────────────────────────────────────────────────────
# Atmosphere
# ──────────────────────────────────────────────────────────────────────────────

def build_atmosphere(cfg: _NS, pupil_grid, seed: int):
    """
    Construct an hcipy atmospheric object, or return None if disabled.
    """
    if not cfg.atmosphere.enabled:
        return None

    r0   = cfg.atmosphere.r0_500nm
    wl_r = cfg.optics.wavelength_ref

    if cfg.atmosphere.use_las_campanas:
        # hcipy's Las Campanas profile does not expose per-layer seeds;
        # seed numpy's legacy random state as a workaround for reproducibility.
        np.random.seed(seed)
        layers = make_las_campanas_atmospheric_layers(
            pupil_grid,
            r0=r0,
            L0=cfg.atmosphere.L0,
            wavelength=wl_r,
        )
        np.random.seed(None)          # restore to non-deterministic
        atm = MultiLayerAtmosphere(layers, scintillation=False)
    else:
        sl    = cfg.atmosphere.single_layer
        v     = sl.wind_speed
        theta = sl.wind_direction
        velocity = [v * math.cos(theta), v * math.sin(theta)]
        layer = InfiniteAtmosphericLayer(
            pupil_grid, sl.Cn2, sl.L0, velocity, seed=seed
        )
        atm = MultiLayerAtmosphere([layer], scintillation=False)

    atm.t = 0.0
    return atm


def get_atm_opd(atmosphere, t: float, wavelength_ref: float):
    """
    Return atmospheric OPD (metres) at time t as ndarray, or None.
    """
    if atmosphere is None:
        return None
    atmosphere.t = t
    phase_rad = np.array(atmosphere.phase_for(wavelength_ref))
    opd = phase_rad * wavelength_ref / (2.0 * math.pi)
    return opd


# ──────────────────────────────────────────────────────────────────────────────
# Polychromatic propagation — Option B
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
    atm_opd,
    aperture:     np.ndarray,
    prop:         FraunhoferPropagator,
    pupil_grid,
    wavelengths:  np.ndarray,
    weights:      np.ndarray,
    wl_ref:       float,
    img_size:     int,
) -> np.ndarray:
    """
    Propagate a polychromatic wavefront to the focal plane.

    Option B: propagate each wavelength on the fixed λ_ref/D grid, zoom
    each PSF by (λ / λ_ref) to convert to physical pixel scale, then sum.

    Returns broadband PSF (img_size, img_size), normalised to unit total flux.
    """
    broadband = np.zeros((img_size, img_size), dtype=np.float64)

    total_opd = mirror_opd + defocus_sign * c4_defocus * defocus_opd
    if atm_opd is not None:
        total_opd = total_opd + atm_opd

    # Ngrid_foc: focal grid side length, derived from the propagated field size
    # (NOT the pupil grid size — they differ: pupil has pupil_samples², focal
    # has (2*num_airy*q)² ≈ 384² for q=8, num_airy=24).
    Ngrid_foc = None

    for wl, wt in zip(wavelengths, weights):
        phase     = (2.0 * math.pi / wl) * total_opd
        amplitude = aperture * np.exp(1j * phase)
        wf        = Wavefront(Field(amplitude.astype(np.complex128), pupil_grid), wl)
        psf_field = prop.forward(wf).power
        if Ngrid_foc is None:
            Ngrid_foc = int(round(math.sqrt(len(psf_field))))
        psf_2d    = np.array(psf_field).reshape(Ngrid_foc, Ngrid_foc)

        scale = wl / wl_ref
        if abs(scale - 1.0) < 1e-9:
            psf_phys = psf_2d
        else:
            psf_phys = zoom(psf_2d, scale, order=3, mode='constant', cval=0.0)

        broadband += wt * _centre_crop_or_pad(psf_phys, img_size)

    total = broadband.sum()
    if total > 0:
        broadband /= total

    return broadband


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg: _NS, output_path: Optional[str] = None, dry_run: bool = False):
    """
    Generate the full synthetic dataset and write to HDF5.
    """
    out_path    = Path(output_path or cfg.output.path)
    n_examples  = cfg.simulation.n_examples
    n_atm_seeds = cfg.simulation.n_atm_seeds
    n_total     = n_examples * n_atm_seeds
    img_size    = cfg.simulation.img_size
    n_modes     = cfg.zernike.n_modes
    chunk       = cfg.simulation.hdf5_chunk_size
    seed_base   = cfg.simulation.random_seed

    if dry_run:
        n_total = 2
        print("DRY RUN: generating 2 examples only, no HDF5 written.")

    print("Building optical system...")
    (pupil_grid, focal_grid, prop,
     aperture, defocus_opd_unit,
     c4_defocus, focal_length) = build_optics(cfg)

    print(f"  OD={cfg.optics.OD} m, f/{cfg.optics.focal_ratio}, "
          f"delta_z={cfg.optics.delta_z*1e3:.2f} mm → c4={c4_defocus*1e9:.1f} nm")
    print(f"  Focal grid: q={cfg.optics.q}, num_airy={cfg.optics.num_airy}, "
          f"output {img_size}×{img_size} px")

    print("Building Zernike basis...")
    zernike_basis = build_zernike_basis(cfg, pupil_grid)
    amplitudes    = make_amplitude_array(cfg)
    wavelengths, weights = make_wavelength_grid(cfg)
    rng = np.random.default_rng(seed_base)

    print(f"  {len(wavelengths)} wavelengths "
          f"[{wavelengths.min()*1e9:.0f}–{wavelengths.max()*1e9:.0f}] nm")
    print(f"  {n_modes} modes, amplitude RMS "
          f"[{amplitudes.min()*1e9:.0f}–{amplitudes.max()*1e9:.0f}] nm")

    # ── Dry run ──────────────────────────────────────────────────────────────
    if dry_run:
        for i in range(n_total):
            atm_seed = seed_base + i
            atm      = build_atmosphere(cfg, pupil_grid, seed=atm_seed)
            labels   = draw_coefficients(amplitudes, cfg.zernike.distribution, rng)
            mirror_opd = sum(float(c) * m for c, m in zip(labels, zernike_basis))

            I1_acc = np.zeros((img_size, img_size))
            I2_acc = np.zeros((img_size, img_size))
            dt = 1.0 / cfg.simulation.frame_rate
            for k in range(cfg.simulation.t_frames):
                atm_opd = get_atm_opd(atm, k * dt, cfg.optics.wavelength_ref)
                I1_acc += propagate_polychromatic(
                    mirror_opd, +1.0, defocus_opd_unit, c4_defocus,
                    atm_opd, aperture, prop, pupil_grid,
                    wavelengths, weights, cfg.optics.wavelength_ref, img_size,
                )
                I2_acc += propagate_polychromatic(
                    mirror_opd, -1.0, defocus_opd_unit, c4_defocus,
                    atm_opd, aperture, prop, pupil_grid,
                    wavelengths, weights, cfg.optics.wavelength_ref, img_size,
                )
            I1 = (I1_acc / cfg.simulation.t_frames).astype(np.float16)
            I2 = (I2_acc / cfg.simulation.t_frames).astype(np.float16)
            print(f"  Example {i}: I1 {I1.shape} sum={float(I1.astype(np.float32).sum()):.4f}  "
                  f"labels [{labels.min()*1e9:.1f}..{labels.max()*1e9:.1f}] nm OPD")
        print("Dry run complete.")
        return

    # ── HDF5 allocation ───────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {n_total} examples to {out_path} ...")

    with h5py.File(out_path, 'w') as f:
        ds_psfs = f.create_dataset(
            'psfs',
            shape=(n_total, 2, img_size, img_size),
            dtype='float16',
            chunks=(chunk, 2, img_size, img_size),
            compression='gzip', compression_opts=4,
        )
        ds_labels = f.create_dataset(
            'labels',
            shape=(n_total, n_modes),
            dtype='float32',
            chunks=(chunk, n_modes),
        )
        ds_labels.attrs['label_units'] = cfg.output.label_units
        ds_labels.attrs['noll_start']  = 2
        ds_labels.attrs['n_modes']     = n_modes
        f.attrs['config'] = json.dumps(dict(cfg), default=str)

        row = 0
        t0  = time.time()
        dt  = 1.0 / cfg.simulation.frame_rate

        for ex_idx in range(n_examples):
            labels_ex  = draw_coefficients(amplitudes, cfg.zernike.distribution, rng)
            mirror_opd = sum(float(c) * m for c, m in zip(labels_ex, zernike_basis))

            for seed_offset in range(n_atm_seeds):
                atm_seed = seed_base + ex_idx * n_atm_seeds + seed_offset
                atm      = build_atmosphere(cfg, pupil_grid, seed=atm_seed)

                I1_acc = np.zeros((img_size, img_size), dtype=np.float64)
                I2_acc = np.zeros((img_size, img_size), dtype=np.float64)

                for k in range(cfg.simulation.t_frames):
                    atm_opd = get_atm_opd(atm, k * dt, cfg.optics.wavelength_ref)
                    I1_acc += propagate_polychromatic(
                        mirror_opd, +1.0, defocus_opd_unit, c4_defocus,
                        atm_opd, aperture, prop, pupil_grid,
                        wavelengths, weights, cfg.optics.wavelength_ref, img_size,
                    )
                    I2_acc += propagate_polychromatic(
                        mirror_opd, -1.0, defocus_opd_unit, c4_defocus,
                        atm_opd, aperture, prop, pupil_grid,
                        wavelengths, weights, cfg.optics.wavelength_ref, img_size,
                    )

                ds_psfs[row, 0] = (I1_acc / cfg.simulation.t_frames).astype(np.float16)
                ds_psfs[row, 1] = (I2_acc / cfg.simulation.t_frames).astype(np.float16)
                ds_labels[row]  = labels_ex.astype(np.float32)
                row += 1

            if (ex_idx + 1) % max(1, n_examples // 20) == 0 or ex_idx == n_examples - 1:
                elapsed = time.time() - t0
                done    = row
                rate    = done / elapsed if elapsed > 0 else 0
                eta     = (n_total - done) / rate if rate > 0 else float('inf')
                print(f"  [{done:>{len(str(n_total))}}/{n_total}]  "
                      f"{elapsed:.0f}s  {rate:.2f} ex/s  ETA {eta:.0f}s")

    print(f"\nDone.  HDF5: {out_path}")
    print(f"  psfs   {ds_psfs.shape}   float16")
    print(f"  labels {ds_labels.shape} float32")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate synthetic CWFS training data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config',   required=True,
                        help='Path to data_generation.yaml.')
    parser.add_argument('--output',   default=None,
                        help='Override output HDF5 path from config.')
    parser.add_argument('--dry-run',  action='store_true',
                        help='Generate 2 examples and print shapes, no HDF5 written.')

    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, overrides)
    main(cfg, output_path=args.output, dry_run=args.dry_run)


# ── Legacy constants kept for reference ──────────────────────────────────────
# OD, ID = 0.76, 0.152               # metres
# FOCAL_LENGTH = OD * 3.33
# WAVELENGTHS = 1.0 / np.linspace(1/700e-9, 1/400e-9, 35)
# DEFOCUS_WFE = 4.5 * 545e-9
# N_ZERNIKE   = 36
# N_EXAMPLES  = 1000
# T_FRAMES    = 32
# N_ATM_SEEDS = 1
# DT          = 1/50

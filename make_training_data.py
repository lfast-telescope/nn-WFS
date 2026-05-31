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
labels : float32  [N, n_modes]   Zernike coefficients Z1..Z{n_modes}, metres OPD
                                  indices 0–2 (Z1 piston, Z2 tip, Z3 tilt) are always zero
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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
        imshow_field
    )
    from hcipy.atmosphere import Cn_squared_from_fried_parameter
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
    weights = np.ones(n, dtype=np.float64)
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

    # c4 = delta_z / (16 * (f/#)² * sqrt(3))   [metres OPD, Noll convention]
    c4_defocus = delta_z / (16.0 * focal_ratio**2 * math.sqrt(3))

    return pupil_grid, focal_grid, prop, aperture, defocus_mode, c4_defocus, focal_length


def build_zernike_basis(cfg: _NS, pupil_grid):
    """
    Return list of n_modes hcipy Zernike modes as ndarrays on pupil_grid.
    Modes are Z1..Z{n_modes} in Noll ordering (includes piston Z1).
    """
    n_modes = cfg.zernike.n_modes
    basis   = make_zernike_basis(n_modes, cfg.optics.OD, pupil_grid, starting_mode=1)
    modes   = [np.array(b) for b in basis]

    print(f"  Zernike basis: {n_modes} modes Z1–Z{n_modes}, "
          f"mode[0] (tip) peak={modes[1].max():.3f}, "
          f"mode[2] (defocus) peak={modes[3].max():.3f}")
    return modes


# ──────────────────────────────────────────────────────────────────────────────
# Zernike coefficient sampling
# ──────────────────────────────────────────────────────────────────────────────

def _noll_radial_order(j: int) -> int:
    """Return the radial order n for Noll index j (1-based)."""
    return int(math.ceil((-3.0 + math.sqrt(1.0 + 8.0 * j)) / 2.0))


def draw_coefficients(cfg: _NS, rng: np.random.Generator) -> np.ndarray:
    """
    Draw one set of Zernike coefficients in metres OPD, shape (n_modes,).

    Builds per-mode amplitudes from config (honouring amplitude_normalization
    and per_mode_rms_nm), then samples from the requested distribution.
    Modes 0–2 (Z2–Z4, tip/tilt/piston equivalents) are zeroed after drawing.
    """
    n             = cfg.zernike.n_modes
    distribution  = cfg.zernike.distribution
    normalization = str(cfg.zernike.get('amplitude_normalization', 'none')).lower()
    override      = cfg.zernike.per_mode_rms_nm

    if override is not None:
        amplitudes = np.asarray(override, dtype=np.float64) * 1e-9
        if len(amplitudes) != n:
            raise ValueError(
                f"per_mode_rms_nm has {len(amplitudes)} entries but n_modes={n}"
            )
    else:
        scalar = cfg.zernike.amplitude_rms   # metres
        amplitudes = np.full(n, scalar, dtype=np.float64)
        if normalization == 'radial_order':
            # Z1 (piston) has radial order 0; clamp to 1 since its coeff is always zeroed.
            radial_orders = np.array(
                [max(1, _noll_radial_order(j)) for j in range(1, n + 1)],
                dtype=np.float64,
            )
            amplitudes = amplitudes / radial_orders
        elif normalization != 'none':
            raise ValueError(
                f"Unknown amplitude_normalization '{normalization}'. "
                "Use 'none' or 'radial_order'."
            )

    if distribution == 'gaussian':
        dist = rng.standard_normal(n) * amplitudes
    elif distribution == 'uniform':
        half = amplitudes * math.sqrt(3)
        dist = rng.uniform(-half, half)
    else:
        raise ValueError(f"Unknown distribution '{distribution}'. Use 'gaussian' or 'uniform'.")
    dist[:3] = 0  # Don't train for first three modes
    return dist


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
        cn_squared = Cn_squared_from_fried_parameter(r0, wl_r)
        layers = make_las_campanas_atmospheric_layers(
            pupil_grid,
            cn_squared=cn_squared,
            outer_scale=cfg.atmosphere.L0,
        )
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

def reset_atm_seed(atm):
    """Advance each layer to a new independent realization and reset t to 0."""
    if atm is None:
        return None
    for layer in atm.layers:
        layer.reset(make_independent_realization=True)
    atm._t = 0.0
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
    cfg:          _NS,
    aperture:     np.ndarray,
    prop:         FraunhoferPropagator,
    pupil_grid,
    wavelengths:  np.ndarray,
    weights:      np.ndarray,
    img_size:     int,
    atm,
) -> np.ndarray:
    """
    Propagate a polychromatic wavefront to the focal plane.

    Simulates a long exposure as n_frames sequential frames at the configured
    frame rate.  Each frame is the average of n_sub atmospheric samples spaced
    by one coherence time tau_0 = 0.31 * r0 / v_wind.  The atmosphere is
    advanced monotonically through time via atm.t assignment (hcipy evolve_until).

    The caller must call reset_atm_seed(atm) before each call so that I1 and I2
    start from independent atmospheric realizations.

    Returns broadband PSF (img_size, img_size), normalised to unit total flux.
    """
    wl_ref   = cfg.optics.wavelength_ref
    n_frames = cfg.simulation.t_frames
    t_frame  = 1.0 / cfg.simulation.frame_rate
    pixel_oversample = cfg.optics.pixel_oversample
    raw_size = img_size * pixel_oversample

    tau_0 = None
    if cfg.atmosphere.enabled and atm is not None:
        v_wind = cfg.atmosphere.single_layer.wind_speed
        tau_0  = 0.31 * cfg.atmosphere.r0_500nm / v_wind

    n_sub   = max(1, round(t_frame / tau_0)) if tau_0 is not None else 1
    n_total = n_frames * n_sub

    static_opd = mirror_opd + defocus_sign * c4_defocus * defocus_opd
    broadband = np.zeros((raw_size, raw_size), dtype=np.float64)
    broadband_t_integrated = broadband.copy()
    frames    = np.zeros((n_frames, raw_size, raw_size), dtype=np.float64)

    # Ngrid_foc: focal grid side length, derived from the propagated field size
    # (NOT the pupil grid size — they differ
    Ngrid_foc = None

    for frame_i in range(n_frames):
        for sub_j in range(n_sub):
            t_sample  = frame_i * t_frame + sub_j * tau_0 if tau_0 is not None else 0.0
            atm_opd   = get_atm_opd(atm, t_sample, wl_ref) if atm is not None else None
            total_opd = static_opd + atm_opd if atm_opd is not None else static_opd

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
                broadband += wt * _centre_crop_or_pad(psf_phys, raw_size)
            broadband_t_integrated += broadband
            broadband = broadband*0

        frames[frame_i] = broadband_t_integrated
        broadband_t_integrated = broadband_t_integrated*0        

    # Bin pixel_oversample × pixel_oversample sub-pixels into detector pixels:
    # frames (n_frames, raw_size, raw_size) → binned_frames (n_frames, img_size, img_size)
    binned_frames = frames.reshape(
        n_frames, img_size, pixel_oversample, img_size, pixel_oversample
    ).mean(axis=(2, 4))
    total = binned_frames.sum()
    if total > 0:
        binned_frames /= total

    return binned_frames


# ──────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg: _NS, output_path: Optional[str] = None, dry_run: bool = False):
    """
    Generate the full synthetic dataset and write to HDF5.
    """
    out_path    = Path(output_path or cfg.output.path)
    n_examples  = cfg.simulation.n_examples
    n_total     = n_examples
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
    wavelengths, weights = make_wavelength_grid(cfg)
    rng = np.random.default_rng(seed_base)

    print("Building atmosphere...")
    atm = build_atmosphere(cfg, pupil_grid, seed=seed_base)

    print(f"  {len(wavelengths)} wavelengths "
          f"[{wavelengths.min()*1e9:.0f}–{wavelengths.max()*1e9:.0f}] nm")
    print(f"  {n_modes} modes, amplitude_rms={cfg.zernike.amplitude_rms*1e9:.0f} nm "
          f"(normalization={cfg.zernike.get('amplitude_normalization', 'none')})")

    # ── Dry run ──────────────────────────────────────────────────────────────
    if dry_run:
        for i in range(n_total):
            labels     = draw_coefficients(cfg, rng)
            mirror_opd = sum(float(c) * m for c, m in zip(labels, zernike_basis))

            atm = reset_atm_seed(atm)
            I1 = propagate_polychromatic(
                mirror_opd, +1.0, defocus_opd_unit, c4_defocus,
                cfg, aperture, prop, pupil_grid,
                wavelengths, weights, img_size,
                atm,
            )
            atm = reset_atm_seed(atm)
            I2 = propagate_polychromatic(
                mirror_opd, -1.0, defocus_opd_unit, c4_defocus,
                cfg, aperture, prop, pupil_grid,
                wavelengths, weights, img_size,
                atm,
            )
            fig, ax = plt.subplots(1, 3)
            I1m, I2m = I1.mean(axis=0), I2.mean(axis=0)
            I2m = np.rot90(I2m,k=2) #Roddier, yo!
            S = (I2m - I1m)/(I2m + I1m)

            if False:
                ax[0].imshow(I1m)
                ax[1].imshow(I2m)
                ax[2].imshow((I1m - I2m) / (I1m + I2m + 1e-12))
                plt.suptitle(f"c4={c4_defocus*1e9:.1f} nm")
                fname = f"rot_150nm_1as_c4_{c4_defocus*1e9:.0f}nm.png"
                plt.savefig('imgs/' + fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved {fname}")
   
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
        ds_labels.attrs['noll_start']  = 1
        ds_labels.attrs['n_modes']     = n_modes
        ds_signal = f.create_dataset(
            'signal',
            shape=(n_total, img_size, img_size),
            dtype='float32',
            chunks=(chunk, img_size, img_size),
            compression='gzip', compression_opts=4,
        )
        ds_signal.attrs['description'] = 'Roddier curvature signal S = (I2_rot - I1) / (I2_rot + I1)'
        f.attrs['config'] = json.dumps(dict(cfg), default=str)

        row = 0
        t0  = time.time()

        for ex_idx in range(n_examples):
            labels_ex  = draw_coefficients(cfg, rng)
            mirror_opd = sum(float(c) * m for c, m in zip(labels_ex, zernike_basis))

            atm = reset_atm_seed(atm)
            I1m = propagate_polychromatic(
                mirror_opd, +1.0, defocus_opd_unit, c4_defocus,
                cfg, aperture, prop, pupil_grid,
                wavelengths, weights, img_size,
                atm,
            ).mean(axis=0)
            atm = reset_atm_seed(atm)
            I2m = np.rot90(propagate_polychromatic(
                mirror_opd, -1.0, defocus_opd_unit, c4_defocus,
                cfg, aperture, prop, pupil_grid,
                wavelengths, weights, img_size,
                atm,
            ).mean(axis=0), k=2)
            S = (I2m - I1m) / (I2m + I1m + 1e-12)

            ds_psfs[row, 0]  = I1m.astype(np.float16)
            ds_psfs[row, 1]  = I2m.astype(np.float16)
            ds_signal[row]   = S.astype(np.float32)
            ds_labels[row]   = labels_ex.astype(np.float32)
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
    print(f"  signal {ds_signal.shape} float32")
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

"""
test_optical_scaling.py — Unit tests for optical propagation scaling,
focal grid coordinate units, and geometric defocus footprint invariance.
"""

import math
import unittest
import numpy as np

from hcipy import (
    Field,
    FraunhoferPropagator,
    Wavefront,
    make_focal_grid,
    make_obstructed_circular_aperture,
    make_pupil_grid,
    make_zernike_basis,
)

from make_training_data import _NS, build_optics, propagate_polychromatic


class TestOpticalScaling(unittest.TestCase):
    def setUp(self):
        self.mock_cfg = _NS({
            "optics": {
                "OD": 0.76,
                "ID": 0.152,
                "focal_ratio": 3.33,
                "wavelength_ref": 550.0e-9,
                "q": 3.025,
                "num_airy": 138,
                "pixel_oversample": 4,
                "pupil_samples": 256,
                "delta_z1": 0.369e-3,
                "delta_z2": 0.738e-3,
                "delta_z": 0.369e-3,
            },
            "wavelengths": {
                "lambda_min": 380.0e-9,
                "lambda_max": 700.0e-9,
                "n_wavelengths": 11,
                "weights": "flat",
            },
            "simulation": {
                "t_frames": 1,
                "frame_rate": 50.0,
                "img_size": 256,
            },
            "atmosphere": {
                "enabled": False,
            },
        })

    def test_focal_grid_pixel_pitch(self):
        """Verify that focal grid delta correctly matches LFAST ZWO ASI183 2.4 um pixel pitch."""
        pupil_grid, focal_grid, prop, aperture, defocus_mode, c4_factor, focal_length = build_optics(self.mock_cfg)

        subpixel_pitch = focal_grid.delta[0]  # metres
        oversample = self.mock_cfg.optics.pixel_oversample
        binned_pitch = subpixel_pitch * oversample

        target_pitch = 2.40e-6  # 2.4 um ASI183 pitch
        rel_diff = abs(binned_pitch - target_pitch) / target_pitch

        self.assertLess(rel_diff, 0.015,
                        f"Binned pixel pitch {binned_pitch*1e6:.3f} um does not match target {target_pitch*1e6:.3f} um (<1.5% expected)")

    def test_geometric_defocus_invariance(self):
        """Verify that defocused geometric pupil radius does NOT scale with wavelength."""
        pupil_grid, focal_grid, prop, aperture, defocus_mode, c4_factor, focal_length = build_optics(self.mock_cfg)

        dz2 = self.mock_cfg.optics.delta_z2
        c4 = dz2 * c4_factor
        defocus_opd = c4 * defocus_mode

        radii = []
        for wl in [380e-9, 550e-9, 700e-9]:
            phase = (2.0 * math.pi / wl) * defocus_opd
            amplitude = aperture * np.exp(1j * phase)
            wf = Wavefront(Field(amplitude.astype(np.complex128), pupil_grid), wl)
            psf_field = prop.forward(wf).power
            psf_2d = np.array(psf_field).reshape(focal_grid.shape)

            mid = psf_2d.shape[0] // 2
            radial_slice = psf_2d[mid, mid:]
            plateau_level = np.median(radial_slice[80:160])
            edge_idx = 80 + np.where(radial_slice[80:250] < 0.2 * plateau_level)[0][0]
            radii.append(edge_idx)

        # The difference across wavelengths should be at most 2 grid pixels (Fresnel edge ring width only)
        # Without the fix (with zoom), this difference was > 40 pixels!
        max_diff = max(radii) - min(radii)
        self.assertLessEqual(max_diff, 2,
                             f"Defocused beam radius varied across wavelengths by {max_diff} pixels: {radii}")

    def test_no_boundary_clipping(self):
        """Verify that Channel II defocused PSF fits completely inside the detector field without edge clipping."""
        pupil_grid, focal_grid, prop, aperture, defocus_mode, c4_factor, focal_length = build_optics(self.mock_cfg)

        dz2 = self.mock_cfg.optics.delta_z2
        c4 = dz2 * c4_factor

        wls = np.linspace(380e-9, 700e-9, 5)
        weights = np.ones(5) / 5.0
        mirror_opd = np.zeros(pupil_grid.size)

        img = propagate_polychromatic(
            mirror_opd, +1.0, defocus_mode, c4,
            self.mock_cfg, aperture, prop, pupil_grid,
            wls, weights, self.mock_cfg.simulation.img_size,
            atm=None,
        )

        frame = img[0]  # [256, 256]
        # Perimeter pixels (top, bottom, left, right edges) should have zero flux
        edge_max = max(
            frame[0, :].max(),
            frame[-1, :].max(),
            frame[:, 0].max(),
            frame[:, -1].max(),
        )
        self.assertEqual(edge_max, 0.0,
                         f"Channel II beam is clipped at detector boundary: edge max is {edge_max:.2e}")


if __name__ == "__main__":
    unittest.main()

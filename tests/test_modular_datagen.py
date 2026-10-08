"""
test_modular_datagen.py — Unit tests for modular multi-plane CWFS data generation.
Tests channel naming, signed metric metadata tags, absolute ±40 um tolerancing,
and zero-padding elimination with num_airy=170.
"""

import math
import unittest
import numpy as np
import yaml

from make_training_data import (
    _NS,
    load_config,
    get_defocus_distances,
    get_channel_metadata,
    sample_tolerancing_parameters,
    _centre_crop_or_pad,
    save_verification_plot,
)
from hcipy import Configuration


class TestModularDataGen(unittest.TestCase):
    def setUp(self):
        self.cfg_5planes = _NS({
            'optics': {
                'OD': 0.76,
                'ID': 0.152,
                'focal_ratio': 3.33,
                'defocus_distances_m': [0.369e-3, 0.738e-3, 1.107e-3, 1.476e-3, 1.845e-3],
                'include_focal': True,
                'q': 3.025,
                'num_airy': 170,
                'pupil_samples': 512,
                'pixel_oversample': 4,
                'wavelength_ref': 550.0e-9,
            },
            'tolerancing': {
                'enabled': True,
                'dz_uncertainty_m': 40.0e-6,
                'dtheta': {'distribution': 'uniform', 'max_deg': 2.0},
                'seeing': {'distribution': 'uniform', 'r0_min': 0.08, 'r0_max': 0.16},
            },
            'atmosphere': {
                'enabled': True,
                'r0_500nm': 0.10,
            },
        })

    def test_channel_metadata_5planes_plus_focal(self):
        ch_names, metric_tags, nominal_dists, include_focal = get_channel_metadata(self.cfg_5planes)
        self.assertEqual(len(ch_names), 11)
        self.assertEqual(len(metric_tags), 11)
        self.assertEqual(len(nominal_dists), 5)
        self.assertTrue(include_focal)

        expected_channels = [
            'Ii1', '-Ii1',
            'Iii1', '-Iii1',
            'Iiii1', '-Iiii1',
            'Iiv1', '-Iiv1',
            'Iv1', '-Iv1',
            'Ifocal',
        ]
        self.assertEqual(ch_names, expected_channels)

        expected_tags = [
            '+0.369mm', '-0.369mm',
            '+0.738mm', '-0.738mm',
            '+1.107mm', '-1.107mm',
            '+1.476mm', '-1.476mm',
            '+1.845mm', '-1.845mm',
            '0.000mm',
        ]
        self.assertEqual(metric_tags, expected_tags)

    def test_channel_metadata_without_focal(self):
        import copy
        cfg_no_focal = copy.deepcopy(self.cfg_5planes)
        cfg_no_focal['optics']['include_focal'] = False

        ch_names, metric_tags, _, include_focal = get_channel_metadata(cfg_no_focal)
        self.assertEqual(len(ch_names), 10)
        self.assertEqual(len(metric_tags), 10)
        self.assertFalse(include_focal)
        self.assertNotIn('Ifocal', ch_names)
        self.assertNotIn('0.000mm', metric_tags)

    def test_absolute_tolerancing_bounds(self):
        rng = np.random.default_rng(12345)
        c4_factor = 1.0 / (16.0 * (3.33**2) * math.sqrt(3.0))

        for _ in range(50):
            params = sample_tolerancing_parameters(self.cfg_5planes, rng, c4_factor)
            sampled_dz = params['sampled_dz']
            self.assertEqual(len(sampled_dz), 11)

            # Check each of the 5 planes (intra and extra)
            nominals = [0.369e-3, 0.738e-3, 1.107e-3, 1.476e-3, 1.845e-3]
            for k, d_nom in enumerate(nominals):
                dz_intra = sampled_dz[2 * k]
                dz_extra = sampled_dz[2 * k + 1]
                # Absolute deviation from nominal must not exceed 40 um / 2 = 20 um
                self.assertLessEqual(abs(dz_intra - d_nom), 20.001e-6)
                self.assertLessEqual(abs(dz_extra - d_nom), 20.001e-6)
                # Asymmetry between intra and extra must not exceed 40 um
                self.assertLessEqual(abs(dz_intra - dz_extra), 40.001e-6)

            # Check focal plane is always zero
            self.assertEqual(sampled_dz[10], 0.0)

    def test_centre_crop_or_pad_num_airy_170(self):
        # num_airy = 170, q = 3.025 -> 2 * 170 * 3.025 = 1028.5 -> 1028 x 1028
        arr_1028 = np.ones((1028, 1028), dtype=np.float64)
        target = 1024
        cropped = _centre_crop_or_pad(arr_1028, target)
        self.assertEqual(cropped.shape, (1024, 1024))
        # Ensure it is a sub-region view / identical values without padding zeros
        self.assertEqual(cropped[0, 0], 1.0)
        self.assertEqual(cropped[-1, -1], 1.0)

    def test_mft_precomputation_settings(self):
        # Verify HCIPy MFT optimization settings are active
        self.assertTrue(Configuration().fourier.mft.precompute_matrices)
        self.assertTrue(Configuration().fourier.mft.allocate_intermediate)

    def test_config_output_tmp_plots_default_false(self):
        cfg = load_config('config/data_generation.yaml')
        self.assertFalse(cfg.simulation.output_tmp_plots)

    def test_save_verification_plot_and_overwrite(self):
        import tempfile
        from pathlib import Path

        ch_names = ['Ii1', '-Ii1', 'Iii1', '-Iii1', 'Ifocal']
        metric_tags = ['+0.369mm', '-0.369mm', '+0.738mm', '-0.738mm', '0.000mm']
        dummy_psfs = np.zeros((len(ch_names), 32, 32), dtype=np.float32)
        dummy_psfs[:, 10:22, 10:22] = 1.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_plot = Path(tmp_dir) / "test_verification.png"

            # Example 0
            save_verification_plot(
                psfs_frame0=dummy_psfs,
                channel_names=ch_names,
                metric_tags=metric_tags,
                ex_idx=0,
                n_examples=2,
                ex_time=1.23,
                out_path=test_plot,
            )
            self.assertTrue(test_plot.exists())
            size1 = test_plot.stat().st_size
            self.assertGreater(size1, 1000)

            # Example 1: Verify overwrite
            save_verification_plot(
                psfs_frame0=dummy_psfs,
                channel_names=ch_names,
                metric_tags=metric_tags,
                ex_idx=1,
                n_examples=2,
                ex_time=0.98,
                out_path=test_plot,
            )
            self.assertTrue(test_plot.exists())
            size2 = test_plot.stat().st_size
            self.assertGreater(size2, 1000)


if __name__ == '__main__':
    unittest.main()

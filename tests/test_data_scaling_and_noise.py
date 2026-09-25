"""
test_data_scaling_and_noise.py — Unit tests for dataset size subsampling,
amplitude range filtering, on-the-fly detector noise injection, and sweep manifests.
"""

import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from nn_WFS.dataset import (
    CWFSDataset,
    apply_detector_noise,
    subsample_indices,
    train_val_test_split,
)
from nn_WFS.sweep import (
    estimate_trial_duration,
    expand_sweep_trials,
    format_override_slug,
    load_sweep_manifest,
)
from nn_WFS.train import (
    _resolve_amplitude_range,
    _resolve_noise_config,
    _resolve_train_subsample,
)


class TestDataScalingAndNoise(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # ── Task 1: Subsampling & Nested Subsets ──────────────────────────

    def test_subsample_indices_ratio(self):
        indices = np.arange(100, dtype=np.int64)
        sub_20 = subsample_indices(indices, ratio=0.20, seed=42)
        self.assertEqual(len(sub_20), 20)
        self.assertTrue(np.all(np.diff(sub_20) > 0))  # sorted ascending

        # Bounds checks
        with self.assertRaises(ValueError):
            subsample_indices(indices, ratio=0.0)
        with self.assertRaises(ValueError):
            subsample_indices(indices, ratio=1.5)

    def test_subsample_indices_count(self):
        indices = np.arange(1000, dtype=np.int64)
        sub_50 = subsample_indices(indices, count=50, seed=42)
        self.assertEqual(len(sub_50), 50)
        self.assertTrue(np.all(np.diff(sub_50) > 0))

        sub_large = subsample_indices(indices, count=2000, seed=42)
        self.assertEqual(len(sub_large), 1000)

        with self.assertRaises(ValueError):
            subsample_indices(indices, count=0)
        with self.assertRaises(ValueError):
            subsample_indices(indices, count=-5)

    def test_subsample_indices_nested_prefix(self):
        """Verify that smaller sample counts are strictly nested subsets of larger counts."""
        indices = np.arange(10000, dtype=np.int64)
        s100 = subsample_indices(indices, count=100, seed=123)
        s500 = subsample_indices(indices, count=500, seed=123)
        s2000 = subsample_indices(indices, count=2000, seed=123)

        self.assertTrue(set(s100).issubset(set(s500)))
        self.assertTrue(set(s500).issubset(set(s2000)))

    # ── Task 2: Detector Noise Injection ──────────────────────────────

    def test_detector_noise_disabled_or_infinite(self):
        I = torch.full((1, 32, 32), 0.05, dtype=torch.float32)
        # Disabled config
        out1 = apply_detector_noise(I, T=8, noise_cfg={'enabled': False}, is_train=True, sample_id=0)
        self.assertTrue(torch.all(out1 == I))

        # None config
        out2 = apply_detector_noise(I, T=8, noise_cfg=None, is_train=True, sample_id=0)
        self.assertTrue(torch.all(out2 == I))

        # Infinite flux
        out3 = apply_detector_noise(I, T=8, noise_cfg={'enabled': True, 'photons_per_frame': float('inf')}, is_train=True, sample_id=0)
        self.assertTrue(torch.all(out3 == I))

    def test_detector_noise_poisson_statistics(self):
        # Create uniform image of 10000 pixels with sum across frame = 1.0 (so with T=8, I = 1.0 / 8)
        H, W = 100, 100
        T = 8
        val_per_px = 1.0 / (T * H * W)
        I = torch.full((1, H, W), val_per_px, dtype=torch.float32)

        n_ph = 1.0e6  # 1 million photons per frame -> 100 photons per pixel on average
        noise_cfg = {
            'enabled': True,
            'photons_per_frame': n_ph,
            'read_noise_e': 0.0,
            'seed': 42,
        }

        noisy_I = apply_detector_noise(I, T=T, noise_cfg=noise_cfg, is_train=True, sample_id=1)

        # Expected counts per pixel = 100
        # Recover counts = noisy_I * T * n_ph
        counts = (noisy_I * T * n_ph).cpu().numpy().flatten()
        mean_c = float(np.mean(counts))
        var_c = float(np.var(counts))

        # Poisson distribution should have mean ≈ 100, var ≈ 100 (relative tolerance ~ 10%)
        self.assertAlmostEqual(mean_c, 100.0, delta=5.0)
        self.assertAlmostEqual(var_c, 100.0, delta=15.0)

    def test_detector_noise_deterministic_eval(self):
        """In eval mode (is_train=False), noise must be 100% reproducible for the same sample_id."""
        I = torch.rand((1, 32, 32), dtype=torch.float32)
        noise_cfg = {'enabled': True, 'photons_per_frame': 1.0e5, 'seed': 42}

        # Two calls in eval mode with same sample_id
        eval1 = apply_detector_noise(I, T=8, noise_cfg=noise_cfg, is_train=False, sample_id=99)
        eval2 = apply_detector_noise(I, T=8, noise_cfg=noise_cfg, is_train=False, sample_id=99)
        self.assertTrue(torch.all(eval1 == eval2))

        # Different sample_id produces different noise realization
        eval3 = apply_detector_noise(I, T=8, noise_cfg=noise_cfg, is_train=False, sample_id=100)
        self.assertFalse(torch.all(eval1 == eval3))

        # In train mode, calling twice produces different realizations
        train1 = apply_detector_noise(I, T=8, noise_cfg=noise_cfg, is_train=True, sample_id=99)
        train2 = apply_detector_noise(I, T=8, noise_cfg=noise_cfg, is_train=True, sample_id=99)
        self.assertFalse(torch.all(train1 == train2))

    # ── Task 3: Amplitude Filtering & Mock Dataset ───────────────────

    def test_amplitude_range_filtering(self):
        h5_path = self.test_dir / "mock_amplitude.h5"
        n_samples = 100
        n_modes = 14

        # Generate synthetic labels in metres OPD with known WFE:
        # Col 0, 1, 2 = 0
        # Sample i has WFE = (i + 1) * 2 nm (from 2 nm to 200 nm)
        labels = np.zeros((n_samples, n_modes), dtype=np.float32)
        for i in range(n_samples):
            wfe_nm = (i + 1) * 2.0  # 2, 4, ..., 200 nm
            wfe_m = wfe_nm * 1e-9
            # Place in mode 4 (index 3)
            labels[i, 3] = wfe_m

        psfs = np.zeros((n_samples, 2, 8, 32, 32), dtype=np.float16)

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('labels', data=labels)
            f.create_dataset('psfs', data=psfs)

        # Filter for [50, 100] nm -> should select indices where wfe_nm in [50, 100]
        # (i+1)*2 in [50, 100] -> i+1 in [25, 50] -> 26 samples
        tr, val, te = train_val_test_split(str(h5_path), ratios=(0.5, 0.25, 0.25), seed=42, amplitude_range_nm=[50.0, 100.0])
        total_selected = len(tr) + len(val) + len(te)
        self.assertEqual(total_selected, 26)

        # Inverted range raises ValueError
        with self.assertRaises(ValueError):
            train_val_test_split(str(h5_path), amplitude_range_nm=[150.0, 50.0])

        # Non-overlapping range raises ValueError (no valid examples)
        with self.assertRaises(ValueError):
            train_val_test_split(str(h5_path), amplitude_range_nm=[500.0, 600.0])

    # ── Resolvers in train.py ─────────────────────────────────────────

    def test_train_resolvers(self):
        # Subsample resolver
        c1, r1 = _resolve_train_subsample({'data': {'train_sample_count': 5000}})
        self.assertEqual(c1, 5000)
        self.assertIsNone(r1)

        c2, r2 = _resolve_train_subsample({'data': {'train_sample_ratio': 0.25}})
        self.assertIsNone(c2)
        self.assertEqual(r2, 0.25)

        # Amplitude range resolver
        a1 = _resolve_amplitude_range({'data': {'amplitude_range_nm': [100.0, 150.0]}})
        self.assertEqual(a1, (100.0, 150.0))

        a2 = _resolve_amplitude_range({'data': {'amplitude_range_nm': "150, 200"}})
        self.assertEqual(a2, (150.0, 200.0))

        a3 = _resolve_amplitude_range({'data': {'amplitude_range_nm': None}})
        self.assertIsNone(a3)

        # Noise resolver
        n1 = _resolve_noise_config({'data': {'noise': {'enabled': True, 'photons_per_frame': 1e5, 'read_noise_e': 1.5}}})
        self.assertTrue(n1['enabled'])
        self.assertEqual(n1['photons_per_frame'], 1e5)
        self.assertEqual(n1['read_noise_e'], 1.5)

        n2 = _resolve_noise_config({'data': {'noise': {'enabled': True, 'photons_per_frame': 'clean'}}})
        self.assertFalse(n2['enabled'])

    # ── Manifest Expansion & Sweep Helpers ────────────────────────────

    def test_sweep_manifests_expansion(self):
        manifest_paths = [
            Path("nn_WFS/experiments/dataset_size_sweep.yaml"),
            Path("nn_WFS/experiments/snr_sweep.yaml"),
            Path("nn_WFS/experiments/amplitude_range_sweep.yaml"),
        ]

        for m_path in manifest_paths:
            manifest, base_cfg = load_sweep_manifest(m_path)
            trials = expand_sweep_trials(manifest, base_cfg)
            self.assertGreater(len(trials), 0)

            for t in trials:
                # Check that slug formatting succeeded
                slug = format_override_slug(t.overrides)
                self.assertTrue(len(slug) > 0)
                # Check that duration estimation succeeds without error
                dur_s, wt_str = estimate_trial_duration(t.resolved_config, manifest)
                self.assertGreater(dur_s, 0.0)
                self.assertTrue(":" in wt_str)


if __name__ == '__main__':
    unittest.main()


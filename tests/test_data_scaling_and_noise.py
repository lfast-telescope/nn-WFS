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


class MockLoader(list):
    """Mock DataLoader list subclass supporting .dataset attribute."""
    def __init__(self, batches=None, dataset=None):
        super().__init__(batches if batches is not None else [])
        self.dataset = dataset


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

    # ── Task 4: GPU Noise Injection & Batched Seeding ─────────────────

    def test_detector_noise_gpu_support(self):
        """Verify that apply_detector_noise executes on CUDA when tensor is on GPU."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        I = torch.full((1, 32, 32), 0.05, dtype=torch.float32, device='cuda')
        noise_cfg = {'enabled': True, 'photons_per_frame': 1e5, 'read_noise_e': 1.0, 'seed': 42}

        # Train mode on CUDA
        noisy_train = apply_detector_noise(I, T=8, noise_cfg=noise_cfg, is_train=True)
        self.assertEqual(noisy_train.device.type, 'cuda')
        self.assertEqual(noisy_train.dtype, I.dtype)
        self.assertFalse(torch.all(noisy_train == I))

        # Eval mode deterministic on CUDA
        noisy_eval1 = apply_detector_noise(I, T=8, noise_cfg=noise_cfg, is_train=False, sample_id=77)
        noisy_eval2 = apply_detector_noise(I, T=8, noise_cfg=noise_cfg, is_train=False, sample_id=77)
        self.assertEqual(noisy_eval1.device.type, 'cuda')
        self.assertTrue(torch.all(noisy_eval1 == noisy_eval2))

        # Different sample_id produces different realization
        noisy_eval3 = apply_detector_noise(I, T=8, noise_cfg=noise_cfg, is_train=False, sample_id=78)
        self.assertFalse(torch.all(noisy_eval1 == noisy_eval3))

    def test_detector_noise_batched_gpu(self):
        """Verify batched [B, T, H, W] GPU tensor noise with per-sample deterministic seeding."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        B, T, H, W = 4, 8, 32, 32
        I = torch.full((B, T, H, W), 0.05, dtype=torch.float32, device=device)
        sample_ids = torch.tensor([101, 202, 303, 404], device=device)
        noise_cfg = {'enabled': True, 'photons_per_frame': 1e5, 'read_noise_e': 2.0, 'seed': 42}

        # Batched eval mode
        noisy1 = apply_detector_noise(I, T=T, noise_cfg=noise_cfg, is_train=False, sample_id=sample_ids)
        noisy2 = apply_detector_noise(I, T=T, noise_cfg=noise_cfg, is_train=False, sample_id=sample_ids)

        self.assertEqual(noisy1.shape, I.shape)
        self.assertEqual(noisy1.device.type, device)
        # 100% reproducible across calls with same sample_ids
        self.assertTrue(torch.all(noisy1 == noisy2))

        # Different samples in the batch have distinct noise realizations
        self.assertFalse(torch.all(noisy1[0] == noisy1[1]))
        self.assertFalse(torch.all(noisy1[1] == noisy1[2]))

    def test_cwfs_dataset_sample_id_and_clean_data(self):
        """Verify CWFSDataset yields clean I1, I2 and sample_id, without CPU noise bottleneck."""
        h5_path = self.test_dir / "mock_clean_dataset.h5"
        n_samples = 5
        n_modes = 14
        labels = np.ones((n_samples, n_modes), dtype=np.float32) * 1e-7
        psfs = np.full((n_samples, 2, 8, 32, 32), 0.05, dtype=np.float16)

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('labels', data=labels)
            f.create_dataset('psfs', data=psfs)

        noise_cfg = {'enabled': True, 'photons_per_frame': 1e4, 'seed': 42}

        # Stack mode (return_stacks=True)
        ds_stack = CWFSDataset(h5_path, list(range(n_samples)), return_stacks=True, noise_cfg=noise_cfg)
        item0 = ds_stack[0]
        self.assertIn('sample_id', item0)
        self.assertEqual(item0['sample_id'], 0)
        # PSFs must be clean float32 matching raw psfs (no CPU Poisson noise applied)
        self.assertTrue(np.allclose(item0['I1'].numpy(), psfs[0, 0].astype(np.float32)))
        self.assertTrue(np.allclose(item0['I2'].numpy(), psfs[0, 1].astype(np.float32)))

        # Pair mode (return_stacks=False)
        ds_pair = CWFSDataset(h5_path, list(range(n_samples)), return_stacks=False, noise_cfg=noise_cfg)
        item_p0 = ds_pair[0]
        self.assertIn('sample_id', item_p0)
        self.assertEqual(item_p0['sample_id'], 0)
        self.assertTrue(np.allclose(item_p0['I1'].numpy()[0], psfs[0, 0, 0].astype(np.float32)))

    def test_detector_noise_batch_size_one_and_list(self):
        """Verify that B=1 tensor, list, and scalar sample_ids produce identical deterministic results."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        B, T, H, W = 1, 8, 32, 32
        I = torch.full((B, T, H, W), 0.05, dtype=torch.float32, device=device)
        noise_cfg = {'enabled': True, 'photons_per_frame': 1e5, 'read_noise_e': 1.5, 'seed': 42}

        # 1D tensor of length 1
        s_tensor = torch.tensor([555], device=device)
        noisy_tensor = apply_detector_noise(I, T=T, noise_cfg=noise_cfg, is_train=False, sample_id=s_tensor)

        # Python list of length 1
        noisy_list = apply_detector_noise(I, T=T, noise_cfg=noise_cfg, is_train=False, sample_id=[555])

        # Scalar int
        noisy_scalar = apply_detector_noise(I, T=T, noise_cfg=noise_cfg, is_train=False, sample_id=555)

        self.assertTrue(torch.all(noisy_tensor == noisy_list))
        self.assertTrue(torch.all(noisy_tensor == noisy_scalar))

    def test_cwfs_dataset_sample_id_non_temporal(self):
        """Verify non-temporal dataset (shape: [N, 2, H, W]) works correctly in both lazy and preloaded modes."""
        h5_path = self.test_dir / "mock_non_temporal.h5"
        n_samples = 4
        n_modes = 14
        labels = np.ones((n_samples, n_modes), dtype=np.float32) * 1e-7
        psfs = np.full((n_samples, 2, 32, 32), 0.05, dtype=np.float16)

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('labels', data=labels)
            f.create_dataset('psfs', data=psfs)

        noise_cfg = {'enabled': True, 'photons_per_frame': 1e4, 'seed': 42}

        # Lazy mode
        ds_lazy = CWFSDataset(h5_path, list(range(n_samples)), preload=False, noise_cfg=noise_cfg)
        self.assertEqual(len(ds_lazy), n_samples)
        for idx in range(n_samples):
            item = ds_lazy[idx]
            self.assertEqual(item['sample_id'], idx)
            self.assertEqual(item['I1'].shape, (1, 32, 32))

        # Preload mode
        ds_preload = CWFSDataset(h5_path, list(range(n_samples)), preload=True, noise_cfg=noise_cfg)
        self.assertEqual(len(ds_preload), n_samples)
        for idx in range(n_samples):
            item = ds_preload[idx]
            self.assertEqual(item['sample_id'], idx)
            self.assertEqual(item['I1'].shape, (1, 32, 32))

    def test_evaluate_predict_tta_with_rodcnn_and_noise(self):
        """Verify evaluate._predict works with RODCNN + TTA + noise without argument collision."""
        from nn_WFS.evaluate import _predict
        from nn_WFS.models.cnn_cwfs import RODCNN

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RODCNN(base_ch=8, stage_blocks=1, n_outputs=14).to(device)

        # Mock loader yielding stack mode batches [B, T, H, W]
        B, T, H, W = 2, 8, 32, 32
        I1 = torch.full((B, T, H, W), 0.05, dtype=torch.float32)
        I2 = torch.full((B, T, H, W), 0.05, dtype=torch.float32)
        labels = torch.zeros((B, 14), dtype=torch.float32)
        sample_id = torch.tensor([10, 20], dtype=torch.long)

        batch = {'I1': I1, 'I2': I2, 'labels': labels, 'sample_id': sample_id}
        loader = MockLoader([batch])

        # Attach dataset attributes
        class MockDataset:
            T = 8
            noise_cfg = {'enabled': True, 'photons_per_frame': 1e5, 'seed': 42}
        loader.dataset = MockDataset()

        lm = torch.zeros(14)
        ls = torch.ones(14)

        # Run with TTA enabled and noise enabled
        pred, target = _predict(
            model=model,
            loader=loader,
            label_mean=lm,
            label_std=ls,
            device=device,
            tta=True,
            trained_modes=list(range(4, 18)),
            noise_cfg=loader.dataset.noise_cfg,
        )

        self.assertEqual(pred.shape, (B, 14))
        self.assertEqual(target.shape, (B, 14))

    def test_evaluate_predict_two_stream_and_r_stack_with_noise(self):
        """Verify evaluate._predict works with two_stream and r_stack input_modes."""
        from nn_WFS.evaluate import _predict
        from nn_WFS.models.cnn_cwfs import SIAMCNN

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Test two_stream
        model_two = SIAMCNN(base_ch=8, stage_blocks=1, n_outputs=14, input_mode='two_stream').to(device)
        B, T, H, W = 2, 8, 32, 32
        batch_two = {
            'I1': torch.full((B, T, H, W), 0.05, dtype=torch.float32),
            'I2': torch.full((B, T, H, W), 0.05, dtype=torch.float32),
            'labels': torch.zeros((B, 14), dtype=torch.float32),
            'sample_id': torch.tensor([1, 2], dtype=torch.long),
        }
        loader_two = MockLoader([batch_two])
        class MockDsTwo:
            T = 8
            noise_cfg = {'enabled': True, 'photons_per_frame': 1e5, 'seed': 42}
        loader_two.dataset = MockDsTwo()

        lm = torch.zeros(14)
        ls = torch.ones(14)

        pred_two, _ = _predict(
            model=model_two,
            loader=loader_two,
            label_mean=lm,
            label_std=ls,
            device=device,
            tta=False,
            noise_cfg=loader_two.dataset.noise_cfg,
        )
        self.assertEqual(pred_two.shape, (B, 14))

        # 2. Test r_stack
        model_r = SIAMCNN(base_ch=8, stage_blocks=1, n_cross_blocks=0, n_outputs=14, input_mode='r_stack').to(device)
        pred_r, _ = _predict(
            model=model_r,
            loader=loader_two,
            label_mean=lm,
            label_std=ls,
            device=device,
            tta=False,
            noise_cfg=loader_two.dataset.noise_cfg,
        )
        # In r_stack mode with B=2, TT=64, pred has B*T² rows
        self.assertEqual(pred_r.shape, (B * T * T, 14))


if __name__ == '__main__':
    unittest.main()



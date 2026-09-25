"""
Unit tests for Phase 6 Architecture Capacity Sweep:
base_ch and stage_blocks resolution, bounds validation, model instantiation,
parameter count scaling, CLI parsing, duration estimation scaling, and manifest expansion.
"""

import unittest
from pathlib import Path
import torch

from nn_WFS.train import _resolve_base_ch, _resolve_stage_blocks, _resolve_dropout, build_model, _parse_args
from nn_WFS.models.cnn_cwfs import RODCNN
from nn_WFS.sweep import load_sweep_manifest, expand_sweep_trials, estimate_trial_duration


class TestArchitectureResolution(unittest.TestCase):
    """Tests for _resolve_base_ch and _resolve_stage_blocks."""

    def test_default_values(self):
        """Verify default fallbacks (base_ch=32, stage_blocks=2)."""
        cfg = {'model': {}}
        self.assertEqual(_resolve_base_ch(cfg), 32)
        self.assertEqual(_resolve_stage_blocks(cfg), 2)
        self.assertEqual(_resolve_base_ch({}), 32)
        self.assertEqual(_resolve_stage_blocks({}), 2)

    def test_explicit_values(self):
        """Verify explicit integer values."""
        cfg = {'model': {'base_ch': 24, 'stage_blocks': 3}}
        self.assertEqual(_resolve_base_ch(cfg), 24)
        self.assertEqual(_resolve_stage_blocks(cfg), 3)

        cfg_str = {'model': {'base_ch': '48', 'stage_blocks': '2'}}
        self.assertEqual(_resolve_base_ch(cfg_str), 48)
        self.assertEqual(_resolve_stage_blocks(cfg_str), 2)

    def test_bounds_validation(self):
        """Verify invalid values raise ValueError."""
        # base_ch too small (<8)
        with self.assertRaises(ValueError):
            _resolve_base_ch({'model': {'base_ch': 4}})
        # base_ch odd
        with self.assertRaises(ValueError):
            _resolve_base_ch({'model': {'base_ch': 33}})
        # stage_blocks < 1
        with self.assertRaises(ValueError):
            _resolve_stage_blocks({'model': {'stage_blocks': 0}})
        # invalid type
        with self.assertRaises(ValueError):
            _resolve_base_ch({'model': {'base_ch': 'invalid'}})


class TestModelCapacityInstantiation(unittest.TestCase):
    """Verify RODCNN parameter counts and channel dimensions scale with architecture."""

    def test_parameter_scaling(self):
        # 24 x 2 (smallest: 2.94M)
        m_24_2 = RODCNN(base_ch=24, stage_blocks=2, n_outputs=33, dropout=0.0, stem_stride=2)
        p_24_2 = sum(p.numel() for p in m_24_2.parameters()) / 1e6
        self.assertAlmostEqual(p_24_2, 2.94, places=1)
        self.assertEqual(m_24_2.backbone.out_channels, 192)

        # 32 x 2 (baseline: 5.22M)
        m_32_2 = RODCNN(base_ch=32, stage_blocks=2, n_outputs=33, dropout=0.0, stem_stride=2)
        p_32_2 = sum(p.numel() for p in m_32_2.parameters()) / 1e6
        self.assertAlmostEqual(p_32_2, 5.22, places=1)
        self.assertEqual(m_32_2.backbone.out_channels, 256)

        # 48 x 3 (largest: 17.88M)
        m_48_3 = RODCNN(base_ch=48, stage_blocks=3, n_outputs=33, dropout=0.0, stem_stride=2)
        p_48_3 = sum(p.numel() for p in m_48_3.parameters()) / 1e6
        self.assertAlmostEqual(p_48_3, 17.88, places=1)
        self.assertEqual(m_48_3.backbone.out_channels, 384)

        # Verify strict capacity ordering
        self.assertLess(p_24_2, p_32_2)
        self.assertLess(p_32_2, p_48_3)

    def test_build_model_with_resolved_arch(self):
        cfg = {
            'model': {
                'type': 'rodcnn',
                'base_ch': 48,
                'stage_blocks': 3,
                'dropout': 0.0,
                'n_outputs': 33,
                'stem_stride': 2,
            }
        }
        model = build_model(cfg)
        self.assertEqual(model.__class__.__name__, 'RODCNN')
        self.assertEqual(model.backbone.out_channels, 384)
        self.assertEqual(model.head.net[2].p, 0.0)


class TestDurationEstimationScaling(unittest.TestCase):
    """Verify duration estimates scale with architecture capacity."""

    def test_arch_duration_scaling(self):
        manifest = {'name': 'test', 'slurm': {'time_per_epoch_s': 170.0, 'dataset_scale': 1.0}}
        cfg_small = {'training': {'epochs': 30}, 'ensemble': {'seeds': 2}, 'model': {'base_ch': 24, 'stage_blocks': 2}}
        cfg_base = {'training': {'epochs': 30}, 'ensemble': {'seeds': 2}, 'model': {'base_ch': 32, 'stage_blocks': 2}}
        cfg_large = {'training': {'epochs': 30}, 'ensemble': {'seeds': 2}, 'model': {'base_ch': 48, 'stage_blocks': 3}}

        dur_small, _ = estimate_trial_duration(cfg_small, manifest)
        dur_base, _ = estimate_trial_duration(cfg_base, manifest)
        dur_large, _ = estimate_trial_duration(cfg_large, manifest)

        self.assertLess(dur_small, dur_base)
        self.assertLess(dur_base, dur_large)


class TestArchitectureSweepManifest(unittest.TestCase):
    """Verify Phase 6 architecture_sweep.yaml expands into 6 trials with dropout=0.0."""

    def test_manifest_expansion(self):
        manifest_path = Path(__file__).resolve().parent.parent / "experiments" / "architecture_sweep.yaml"
        self.assertTrue(manifest_path.exists(), f"Missing manifest at {manifest_path}")

        manifest, base_cfg = load_sweep_manifest(manifest_path)
        self.assertEqual(manifest['name'], 'architecture_sweep')

        # Check Slurm settings
        slurm = manifest.get('slurm', {})
        self.assertEqual(slurm.get('account'), 'cbender')
        self.assertEqual(slurm.get('partition'), 'gpu_high_priority')
        self.assertEqual(slurm.get('qos'), 'user_qos_cbender')
        self.assertEqual(slurm.get('gres'), 'gpu:nvidia_a100_80gb_pcie_3g.40gb')

        # Check common overrides
        co = manifest.get('common_overrides', {})
        self.assertEqual(co.get('training.epochs'), 30)
        self.assertEqual(co.get('training.lr'), 1e-3)
        self.assertEqual(co.get('training.weight_decay'), 0.03)
        self.assertEqual(co.get('training.betas'), [0.9, 0.999])
        self.assertEqual(co.get('model.dropout'), 0.0)

        # Expand trials
        trials = expand_sweep_trials(manifest, base_cfg)
        self.assertEqual(len(trials), 6, "Expected exactly 6 trials for 3 base_ch x 2 stage_blocks")

        expected_combinations = [
            (24, 2),
            (24, 3),
            (32, 2),
            (32, 3),
            (48, 2),
            (48, 3),
        ]

        for t, (exp_b, exp_s) in zip(trials, expected_combinations):
            self.assertEqual(t.resolved_config['model']['base_ch'], exp_b)
            self.assertEqual(t.resolved_config['model']['stage_blocks'], exp_s)
            self.assertEqual(t.resolved_config['model']['dropout'], 0.0)
            self.assertEqual(t.resolved_config['training']['epochs'], 30)
            self.assertEqual(t.resolved_config['training']['lr'], 1e-3)
            self.assertEqual(t.resolved_config['training']['weight_decay'], 0.03)


if __name__ == '__main__':
    unittest.main()

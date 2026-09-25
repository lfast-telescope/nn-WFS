"""
Unit tests for Phase 5 Dropout Regularisation configuration, bounds validation,
model instantiation, CLI argument parsing, and sweep manifest expansion.
"""

import unittest
import torch
import torch.nn as nn
from pathlib import Path
import yaml

from nn_WFS.train import _resolve_dropout, _parse_args, build_model
from nn_WFS.models.cnn_cwfs import RODCNN
from nn_WFS.sweep import load_sweep_manifest, expand_sweep_trials


class TestDropoutResolution(unittest.TestCase):
    """Tests for _resolve_dropout helper and bounds validation."""

    def test_default_dropout(self):
        """Verify omission of model.dropout defaults to 0.1."""
        cfg = {'model': {}}
        self.assertEqual(_resolve_dropout(cfg), 0.1)
        self.assertEqual(_resolve_dropout({}), 0.1)

    def test_explicit_values(self):
        """Verify explicit float values are correctly parsed."""
        self.assertEqual(_resolve_dropout({'model': {'dropout': 0.0}}), 0.0)
        self.assertEqual(_resolve_dropout({'model': {'dropout': 0.05}}), 0.05)
        self.assertEqual(_resolve_dropout({'model': {'dropout': 0.1}}), 0.1)
        self.assertEqual(_resolve_dropout({'model': {'dropout': 0.2}}), 0.2)

    def test_string_and_none(self):
        """Verify string representations and None values."""
        self.assertEqual(_resolve_dropout({'model': {'dropout': '0.05'}}), 0.05)
        self.assertEqual(_resolve_dropout({'model': {'dropout': '0.0'}}), 0.0)
        self.assertEqual(_resolve_dropout({'model': {'dropout': None}}), 0.0)

    def test_out_of_bounds_validation(self):
        """Verify ValueError is raised when dropout is outside [0.0, 1.0)."""
        # Negative
        with self.assertRaises(ValueError):
            _resolve_dropout({'model': {'dropout': -0.05}})
        # Exactly 1.0
        with self.assertRaises(ValueError):
            _resolve_dropout({'model': {'dropout': 1.0}})
        # Exceeds 1.0
        with self.assertRaises(ValueError):
            _resolve_dropout({'model': {'dropout': 1.5}})
        # Invalid string
        with self.assertRaises(ValueError):
            _resolve_dropout({'model': {'dropout': 'invalid'}})


class TestModelDropoutInstantiation(unittest.TestCase):
    """Tests for model architecture dropout configuration in MLPHead."""

    def test_rodcnn_direct_instantiation(self):
        """Verify RODCNN passes dropout directly to MLPHead."""
        model_zero = RODCNN(base_ch=16, stage_blocks=1, n_outputs=10, dropout=0.0)
        # In MLPHead: layers = [Linear, GELU, Dropout, Linear]
        self.assertIsInstance(model_zero.head.net[2], nn.Dropout)
        self.assertEqual(model_zero.head.net[2].p, 0.0)

        model_20 = RODCNN(base_ch=16, stage_blocks=1, n_outputs=10, dropout=0.2)
        self.assertEqual(model_20.head.net[2].p, 0.2)

    def test_build_model_with_resolved_dropout(self):
        """Verify build_model applies resolved dropout from config."""
        cfg = {
            'model': {
                'type': 'rodcnn',
                'base_ch': 16,
                'stage_blocks': 1,
                'n_outputs': 10,
                'dropout': 0.05,
            }
        }
        model = build_model(cfg)
        self.assertEqual(model.__class__.__name__, 'RODCNN')
        self.assertEqual(model.head.net[2].p, 0.05)


class TestSweepManifestExpansion(unittest.TestCase):
    """Tests for Phase 5 dropout_sweep.yaml manifest expansion."""

    def test_manifest_expansion(self):
        manifest_path = Path(__file__).resolve().parent.parent / "experiments" / "dropout_sweep.yaml"
        self.assertTrue(manifest_path.exists(), f"Missing manifest at {manifest_path}")

        manifest, base_cfg = load_sweep_manifest(manifest_path)
        self.assertEqual(manifest['name'], 'dropout_sweep')

        # Check Slurm settings
        slurm = manifest.get('slurm', {})
        self.assertEqual(slurm.get('account'), 'cbender')
        self.assertEqual(slurm.get('partition'), 'gpu_high_priority')
        self.assertEqual(slurm.get('qos'), 'user_qos_cbender')
        self.assertEqual(slurm.get('gres'), 'gpu:nvidia_a100_80gb_pcie_3g.40gb')

        # Check common overrides (anchored at 30 epochs)
        co = manifest.get('common_overrides', {})
        self.assertEqual(co.get('training.epochs'), 30)
        self.assertEqual(co.get('training.lr'), 1e-3)
        self.assertEqual(co.get('training.weight_decay'), 0.03)
        self.assertEqual(co.get('training.betas'), [0.9, 0.999])

        # Expand trials
        trials = expand_sweep_trials(manifest, base_cfg)
        self.assertEqual(len(trials), 5, "Expected exactly 5 trials for dropout [0.0, 0.02, 0.05, 0.1, 0.2]")

        dropouts = [t.overrides.get('model.dropout') for t in trials]
        self.assertEqual(dropouts, [0.0, 0.02, 0.05, 0.1, 0.2])

        for t in trials:
            self.assertEqual(t.resolved_config['training']['epochs'], 30)
            self.assertEqual(t.resolved_config['training']['lr'], 1e-3)
            self.assertEqual(t.resolved_config['training']['weight_decay'], 0.03)
            self.assertIn(t.resolved_config['model']['dropout'], [0.0, 0.02, 0.05, 0.1, 0.2])


if __name__ == '__main__':
    unittest.main()

"""
Unit tests for AdamW betas configuration resolution, optimizer instantiation,
CLI argument parsing, and Phase 2 sweep manifest expansion.
"""

import unittest
import torch
import torch.nn as nn
from pathlib import Path
import yaml

from nn_WFS.train import _resolve_betas, _parse_args
from nn_WFS.sweep import load_sweep_manifest, expand_sweep_trials


class TestAdamWBetasResolution(unittest.TestCase):
    """Tests for _resolve_betas helper."""

    def test_default_betas(self):
        """Verify omission of betas defaults to (0.9, 0.999)."""
        cfg = {'training': {}}
        self.assertEqual(_resolve_betas(cfg), (0.9, 0.999))
        self.assertEqual(_resolve_betas({}), (0.9, 0.999))

    def test_list_and_tuple_betas(self):
        """Verify list and tuple values are correctly parsed."""
        cfg_list = {'training': {'betas': [0.85, 0.98]}}
        self.assertEqual(_resolve_betas(cfg_list), (0.85, 0.98))

        cfg_tuple = {'training': {'betas': (0.88, 0.995)}}
        self.assertEqual(_resolve_betas(cfg_tuple), (0.88, 0.995))

    def test_string_betas(self):
        """Verify string representations from YAML or CLI are correctly parsed."""
        cfg_str1 = {'training': {'betas': "[0.85, 0.99]"}}
        self.assertEqual(_resolve_betas(cfg_str1), (0.85, 0.99))

        cfg_str2 = {'training': {'betas': "0.85, 0.995"}}
        self.assertEqual(_resolve_betas(cfg_str2), (0.85, 0.995))

        cfg_str3 = {'training': {'betas': "(0.90, 0.98)"}}
        self.assertEqual(_resolve_betas(cfg_str3), (0.90, 0.98))

    def test_granular_beta1_beta2(self):
        """Verify granular beta1 and beta2 keys override defaults."""
        cfg = {'training': {'beta1': 0.85, 'beta2': 0.98}}
        self.assertEqual(_resolve_betas(cfg), (0.85, 0.98))

    def test_granular_override_precedence(self):
        """Verify granular beta1/beta2 keys take precedence over compound betas."""
        cfg = {'training': {'betas': [0.9, 0.999], 'beta2': 0.99}}
        self.assertEqual(_resolve_betas(cfg), (0.9, 0.99))

        cfg2 = {'training': {'betas': [0.9, 0.999], 'beta1': 0.85}}
        self.assertEqual(_resolve_betas(cfg2), (0.85, 0.999))

    def test_out_of_bounds_validation(self):
        """Verify that invalid beta values raise ValueError."""
        # beta1 >= 1.0
        with self.assertRaises(ValueError):
            _resolve_betas({'training': {'beta1': 1.0}})

        # beta1 < 0.0
        with self.assertRaises(ValueError):
            _resolve_betas({'training': {'beta1': -0.1}})

        # beta2 >= 1.0
        with self.assertRaises(ValueError):
            _resolve_betas({'training': {'beta2': 1.05}})

        # beta2 < 0.0
        with self.assertRaises(ValueError):
            _resolve_betas({'training': {'beta2': -0.05}})

        # Incorrect length
        with self.assertRaises(ValueError):
            _resolve_betas({'training': {'betas': [0.9]}})

        with self.assertRaises(ValueError):
            _resolve_betas({'training': {'betas': [0.9, 0.99, 0.999]}})


class TestOptimizerAdamWIntegration(unittest.TestCase):
    """Verify AdamW optimizer instantiation with resolved betas."""

    def test_adamw_param_groups(self):
        model = nn.Linear(4, 2)
        cfg = {
            'training': {
                'lr': 1e-3,
                'weight_decay': 0.03,
                'betas': [0.85, 0.995],
            }
        }
        betas = _resolve_betas(cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg['training']['lr'],
            betas=betas,
            weight_decay=cfg['training']['weight_decay'],
        )

        group = optimizer.param_groups[0]
        self.assertEqual(group['betas'], (0.85, 0.995))
        self.assertEqual(group['lr'], 1e-3)
        self.assertEqual(group['weight_decay'], 0.03)


class TestAdamWBetasSweepManifest(unittest.TestCase):
    """Verify adamw_betas_sweep.yaml expands accurately."""

    def test_sweep_expansion(self):
        manifest_path = Path(__file__).resolve().parent.parent / "experiments" / "adamw_betas_sweep.yaml"
        self.assertTrue(manifest_path.exists(), f"Manifest not found: {manifest_path}")

        manifest, base_cfg = load_sweep_manifest(manifest_path)
        trials = expand_sweep_trials(manifest, base_cfg)

        # 2 beta1 values x 4 beta2 values = 8 trials
        self.assertEqual(len(trials), 8)

        expected_pairs = [
            (0.85, 0.98),
            (0.85, 0.99),
            (0.85, 0.995),
            (0.85, 0.999),
            (0.90, 0.98),
            (0.90, 0.99),
            (0.90, 0.995),
            (0.90, 0.999),
        ]

        for t, (exp_b1, exp_b2) in zip(trials, expected_pairs):
            cfg = t.resolved_config
            betas = _resolve_betas(cfg)
            self.assertAlmostEqual(betas[0], exp_b1, places=3)
            self.assertAlmostEqual(betas[1], exp_b2, places=4)

            # Check anchor values
            self.assertEqual(cfg['training']['lr'], 0.001)
            self.assertEqual(cfg['training']['weight_decay'], 0.03)
            self.assertEqual(cfg['training']['epochs'], 30)
            self.assertTrue(cfg['training']['swa'])
            self.assertTrue(cfg['evaluation']['tta'])


if __name__ == '__main__':
    unittest.main()


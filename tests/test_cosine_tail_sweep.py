"""
Unit tests for Cosine Tail Sweep manifest and CLI configuration.
"""

import unittest
from pathlib import Path

from nn_WFS.train import _resolve_scheduler_config, _parse_args
from nn_WFS.sweep import load_sweep_manifest, expand_sweep_trials


class TestCosineTailSweepManifest(unittest.TestCase):
    """Tests for cosine_tail_sweep manifest loading and trial expansion."""

    def test_manifest_expansion(self):
        manifest_path = Path(__file__).resolve().parent.parent / "experiments" / "cosine_tail_sweep.yaml"
        self.assertTrue(manifest_path.exists(), f"Manifest not found: {manifest_path}")

        manifest, base_cfg = load_sweep_manifest(manifest_path)
        trials = expand_sweep_trials(manifest, base_cfg)

        # 3 min_lr x 3 tail_ratio = 9 trials
        self.assertEqual(len(trials), 9)

        expected_min_lrs = {1.0e-6, 2.0e-6, 1.0e-5}
        expected_tail_ratios = {0.0, 0.1, 0.2}

        observed_min_lrs = set()
        observed_tail_ratios = set()

        for t in trials:
            sched_cfg = _resolve_scheduler_config(t.resolved_config)
            self.assertEqual(sched_cfg['type'], 'cosine_tail')
            self.assertEqual(t.resolved_config['training']['epochs'], 50)
            observed_min_lrs.add(sched_cfg['min_lr'])
            observed_tail_ratios.add(round(sched_cfg['tail_ratio'], 4))

        self.assertEqual(observed_min_lrs, expected_min_lrs)
        self.assertEqual(observed_tail_ratios, expected_tail_ratios)

    def test_cli_parsing_tail_flags(self):
        """Test --scheduler, --tail_epochs, and --tail_ratio CLI flags."""
        args, _ = _parse_args([
            '--config', 'config/rodcnn.yaml',
            '--scheduler', 'cosine_tail',
            '--tail_epochs', '12',
            '--tail_ratio', '0.25',
            '--min_lr', '5e-6',
        ])
        self.assertEqual(args.scheduler, 'cosine_tail')
        self.assertEqual(args.tail_epochs, 12)
        self.assertEqual(args.tail_ratio, 0.25)
        self.assertEqual(args.min_lr, 5e-6)


if __name__ == '__main__':
    unittest.main()

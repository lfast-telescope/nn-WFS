import argparse
import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root and parent to sys.path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_GIT = _REPO.parent
for p in [str(_REPO), str(_GIT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from nn_WFS.dataset import subsample_indices
    from nn_WFS.train import _load_config, _apply_overrides, _parse_args
except ImportError:
    from dataset import subsample_indices
    from train import _load_config, _apply_overrides, _parse_args


class TestValidationSubsampling(unittest.TestCase):
    def setUp(self):
        self.indices = np.arange(100, 250, dtype=np.int64)  # 150 items

    def test_subsample_ratio_counts(self):
        # 80% of 150 = 120
        sub_80 = subsample_indices(self.indices, 0.8, seed=42)
        self.assertEqual(len(sub_80), 120)

        # 50% of 150 = 75
        sub_50 = subsample_indices(self.indices, 0.5, seed=42)
        self.assertEqual(len(sub_50), 75)

        # 10% of 150 = 15
        sub_10 = subsample_indices(self.indices, 0.1, seed=42)
        self.assertEqual(len(sub_10), 15)

    def test_subsample_ratio_1_0(self):
        # ratio = 1.0 returns entire array unmodified
        sub_full = subsample_indices(self.indices, 1.0, seed=42)
        self.assertEqual(len(sub_full), len(self.indices))
        np.testing.assert_array_equal(sub_full, self.indices)

    def test_subsample_preserves_sorting_and_uniqueness(self):
        sub = subsample_indices(self.indices, 0.8, seed=42)
        # All elements must be in the original array
        self.assertTrue(np.isin(sub, self.indices).all())
        # Must have no duplicates
        self.assertEqual(len(sub), len(np.unique(sub)))
        # Must be strictly ascending
        self.assertTrue(np.all(np.diff(sub) > 0))

    def test_subsample_determinism_and_seed(self):
        sub_a = subsample_indices(self.indices, 0.8, seed=42)
        sub_b = subsample_indices(self.indices, 0.8, seed=42)
        np.testing.assert_array_equal(sub_a, sub_b)

        sub_c = subsample_indices(self.indices, 0.8, seed=123)
        self.assertFalse(np.array_equal(sub_a, sub_c))

    def test_subsample_small_array_min_one(self):
        # Tiny array of 5 items with 0.05 ratio -> round(0.25) = 0, but max(1, ...) guarantees 1 sample
        tiny = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        sub = subsample_indices(tiny, 0.05, seed=42)
        self.assertEqual(len(sub), 1)
        self.assertIn(sub[0], tiny)

    def test_subsample_empty_array(self):
        empty = np.array([], dtype=np.int64)
        sub = subsample_indices(empty, 0.8, seed=42)
        self.assertEqual(len(sub), 0)

    def test_subsample_invalid_ratios_raise(self):
        with self.assertRaises(ValueError):
            subsample_indices(self.indices, 0.0)

        with self.assertRaises(ValueError):
            subsample_indices(self.indices, -0.5)

        with self.assertRaises(ValueError):
            subsample_indices(self.indices, 1.2)

    def test_yaml_config_parsing(self):
        cfg_path = _REPO / "config" / "rodcnn.yaml"
        cfg = _load_config(str(cfg_path))
        self.assertIn('val_sample_ratio', cfg['training'])
        val_ratio = float(cfg['training']['val_sample_ratio'])
        self.assertTrue(0.0 < val_ratio <= 1.0)

    def test_apply_overrides(self):
        cfg = {'training': {'val_sample_ratio': 0.8}}
        cfg = _apply_overrides(cfg, ['training.val_sample_ratio=0.5'])
        self.assertEqual(cfg['training']['val_sample_ratio'], 0.5)

    def test_cli_parsing(self):
        test_args = ['--config', 'config/rodcnn.yaml', '--val_sample_ratio', '0.6']
        sys_argv_bak = sys.argv
        try:
            sys.argv = ['train.py'] + test_args
            args, overrides = _parse_args()
            self.assertEqual(args.val_sample_ratio, 0.6)
        finally:
            sys.argv = sys_argv_bak


if __name__ == '__main__':
    unittest.main()


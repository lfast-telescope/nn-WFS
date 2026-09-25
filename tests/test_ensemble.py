import os
import sys
import unittest
from pathlib import Path

# Add project root and parent to sys.path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_GIT = _REPO.parent
for p in [str(_REPO), str(_GIT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

try:
    from nn_WFS.utils.ensemble import (
        generate_preset_seeds,
        resolve_ensemble_config,
        seed_everything,
        average_predictions,
    )
except ImportError:
    from utils.ensemble import (
        generate_preset_seeds,
        resolve_ensemble_config,
        seed_everything,
        average_predictions,
    )


class TestEnsembleUtils(unittest.TestCase):
    def test_generate_preset_seeds(self):
        self.assertEqual(generate_preset_seeds(0), [])
        self.assertEqual(generate_preset_seeds(1), [42])
        self.assertEqual(generate_preset_seeds(2), [42, 276])

        # Test n=5
        seeds_5 = generate_preset_seeds(5)
        self.assertEqual(len(seeds_5), 5)
        self.assertEqual(seeds_5[:2], [42, 276])
        # Verify subsequent seeds are positive, unique, and deterministic
        self.assertEqual(len(set(seeds_5)), 5)
        for s in seeds_5:
            self.assertIsInstance(s, int)
            self.assertGreater(s, 0)

        # Re-generating must be 100% deterministic
        seeds_5_again = generate_preset_seeds(5)
        self.assertEqual(seeds_5, seeds_5_again)

    def test_resolve_ensemble_config_defaults(self):
        # Omitted
        seeds, mode, is_ens = resolve_ensemble_config({})
        self.assertEqual(seeds, [42])
        self.assertEqual(mode, "best")
        self.assertFalse(is_ens)

        # None / null
        seeds, mode, is_ens = resolve_ensemble_config({"ensemble": None})
        self.assertEqual(seeds, [42])
        self.assertEqual(mode, "best")
        self.assertFalse(is_ens)

    def test_resolve_ensemble_config_int(self):
        # Single run integer
        seeds, mode, is_ens = resolve_ensemble_config({"ensemble": 1})
        self.assertEqual(seeds, [42])
        self.assertEqual(mode, "best")
        self.assertFalse(is_ens)

        # Multiple runs integer
        seeds, mode, is_ens = resolve_ensemble_config({"ensemble": 3})
        self.assertEqual(seeds[:2], [42, 276])
        self.assertEqual(len(seeds), 3)
        self.assertEqual(mode, "average")
        self.assertTrue(is_ens)

    def test_resolve_ensemble_config_list(self):
        seeds, mode, is_ens = resolve_ensemble_config({"ensemble": [42, 276, 117]})
        self.assertEqual(seeds, [42, 276, 117])
        self.assertEqual(mode, "average")
        self.assertTrue(is_ens)

        # Single element list
        seeds, mode, is_ens = resolve_ensemble_config({"ensemble": [123]})
        self.assertEqual(seeds, [123])
        self.assertEqual(mode, "best")
        self.assertFalse(is_ens)

    def test_resolve_ensemble_config_dict(self):
        # Dict with int seeds and explicit mode 'best'
        cfg = {"ensemble": {"seeds": 3, "mode": "best"}}
        seeds, mode, is_ens = resolve_ensemble_config(cfg)
        self.assertEqual(len(seeds), 3)
        self.assertEqual(mode, "best")
        self.assertTrue(is_ens)

        # Dict with list seeds and default mode
        cfg = {"ensemble": {"seeds": [100, 200]}}
        seeds, mode, is_ens = resolve_ensemble_config(cfg)
        self.assertEqual(seeds, [100, 200])
        self.assertEqual(mode, "average")
        self.assertTrue(is_ens)

        # Invalid mode
        with self.assertRaises(ValueError):
            resolve_ensemble_config({"ensemble": {"seeds": 2, "mode": "median"}})

    def test_resolve_ensemble_config_cli_string(self):
        # Simulating CLI overrides: --ensemble=3
        seeds, mode, is_ens = resolve_ensemble_config({"ensemble": "3"})
        self.assertEqual(len(seeds), 3)
        self.assertEqual(mode, "average")
        self.assertTrue(is_ens)

        # Simulating CLI overrides: --ensemble="[42, 276]"
        seeds, mode, is_ens = resolve_ensemble_config({"ensemble": "[42, 276]"})
        self.assertEqual(seeds, [42, 276])
        self.assertEqual(mode, "average")
        self.assertTrue(is_ens)

    def test_seed_everything(self):
        seed_everything(42)
        r1 = torch.randn(5)
        n1 = np.random.randn(5)

        seed_everything(42)
        r2 = torch.randn(5)
        n2 = np.random.randn(5)

        torch.testing.assert_close(r1, r2)
        np.testing.assert_allclose(n1, n2)

    def test_average_predictions(self):
        p1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        p2 = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        avg = average_predictions([p1, p2])
        expected = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        torch.testing.assert_close(avg, expected)

    def test_eval_ensemble_synthetic(self):
        import tempfile
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from evaluate import eval_ensemble
        from train import build_model

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {
                'model': {
                    'type': 'toy',
                    'img_size': 32,
                    'patch_size': 16,
                    'input_mode': 'pairs',
                    'trained_modes': [4, 5],
                    'n_outputs': 2,
                }
            }

            m1 = build_model(cfg)
            m2 = build_model(cfg)

            with torch.no_grad():
                nn.init.zeros_(m1.fc_output.weight)
                m1.fc_output.bias.fill_(1.0)
                nn.init.zeros_(m2.fc_output.weight)
                m2.fc_output.bias.fill_(3.0)

            # Save ckpt 1 (val_wfe = 20.0 nm, output = 1.0)
            ckpt1_path = os.path.join(tmpdir, "seed42.pt")
            torch.save({
                'model_state': m1.state_dict(),
                'val_wfe_rms': 20.0e-9,
                'epoch': 5,
                'config': cfg,
                'label_mean': np.array([0.0, 0.0], dtype=np.float32),
                'label_std': np.array([1.0, 1.0], dtype=np.float32),
            }, ckpt1_path)

            # Save ckpt 2 (val_wfe = 10.0 nm, output = 3.0) -> better model
            ckpt2_path = os.path.join(tmpdir, "seed276.pt")
            torch.save({
                'model_state': m2.state_dict(),
                'val_wfe_rms': 10.0e-9,
                'epoch': 8,
                'config': cfg,
                'label_mean': np.array([0.0, 0.0], dtype=np.float32),
                'label_std': np.array([1.0, 1.0], dtype=np.float32),
            }, ckpt2_path)


            # Create dummy test loader
            I1 = torch.zeros(4, 1, 32, 32)
            I2 = torch.zeros(4, 1, 32, 32)
            r = torch.zeros(4, 1, 32, 32)
            labels = torch.ones(4, 2) * 2.0  # target is 2.0

            class DummyDictDS(torch.utils.data.Dataset):
                def __len__(self): return 4
                def __getitem__(self, i):
                    return {'I1': I1[i], 'I2': I2[i], 'r': r[i], 'labels': labels[i]}

            loader = DataLoader(DummyDictDS(), batch_size=2)

            # Test mode="best": should choose ckpt2 (val_wfe = 10nm, output=3.0, target=2.0 -> error=1.0)
            metrics_best = eval_ensemble([ckpt1_path, ckpt2_path], loader, mode="best", labels_are_zscored=False)
            self.assertAlmostEqual(metrics_best['wfe_rms_nm'], np.sqrt(2.0) * 1e9, delta=100.0)

            # Test mode="average": output avg is (1.0 + 3.0)/2 = 2.0 -> exactly equals target! error = 0!
            metrics_avg = eval_ensemble([ckpt1_path, ckpt2_path], loader, mode="average", labels_are_zscored=False)
            self.assertAlmostEqual(metrics_avg['wfe_rms_nm'], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()


import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nn_WFS.utils.swa import ModelSWA
from nn_WFS.models.toy_model import SLPCWFS
from nn_WFS.evaluate import load_checkpoint
from nn_WFS.train import _resolve_swa


class SimpleToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2, bias=True)

    def forward(self, x):
        return self.fc(x)


class TestModelSWA(unittest.TestCase):
    def test_init(self):
        """Verify SWA initialization sets n_models=0 and captures model parameter templates."""
        model = SimpleToyModel()
        swa = ModelSWA(model)
        self.assertEqual(swa.n_models, 0)
        self.assertIn('fc.weight', swa.shadow_params)
        self.assertIn('fc.bias', swa.shadow_params)

    def test_running_mean_arithmetic(self):
        """Verify that SWA accumulates the exact arithmetic mean across update steps."""
        model = SimpleToyModel()
        swa = ModelSWA(model)

        w1 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        b1 = torch.tensor([0.1, 0.2])
        model.fc.weight.data.copy_(w1)
        model.fc.bias.data.copy_(b1)
        swa.update(model)
        self.assertEqual(swa.n_models, 1)
        self.assertTrue(torch.allclose(swa.shadow_params['fc.weight'], w1))
        self.assertTrue(torch.allclose(swa.shadow_params['fc.bias'], b1))

        w2 = torch.tensor([[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]])
        b2 = torch.tensor([0.3, 0.4])
        model.fc.weight.data.copy_(w2)
        model.fc.bias.data.copy_(b2)
        swa.update(model)
        self.assertEqual(swa.n_models, 2)
        expected_w = (w1 + w2) / 2.0
        expected_b = (b1 + b2) / 2.0
        self.assertTrue(torch.allclose(swa.shadow_params['fc.weight'], expected_w))
        self.assertTrue(torch.allclose(swa.shadow_params['fc.bias'], expected_b))

        w3 = torch.tensor([[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        b3 = torch.tensor([0.5, 0.6])
        model.fc.weight.data.copy_(w3)
        model.fc.bias.data.copy_(b3)
        swa.update(model)
        self.assertEqual(swa.n_models, 3)
        expected_w3 = (w1 + w2 + w3) / 3.0
        expected_b3 = (b1 + b2 + b3) / 3.0
        self.assertTrue(torch.allclose(swa.shadow_params['fc.weight'], expected_w3))
        self.assertTrue(torch.allclose(swa.shadow_params['fc.bias'], expected_b3))

    def test_apply_shadow_context_manager(self):
        """Verify apply_shadow swaps model weights in context and restores instantaneous weights on exit."""
        model = SimpleToyModel()
        swa = ModelSWA(model)

        w1 = torch.tensor([[2.0, 4.0, 6.0, 8.0], [10.0, 12.0, 14.0, 16.0]])
        model.fc.weight.data.copy_(w1)
        swa.update(model)

        w2 = torch.tensor([[6.0, 8.0, 10.0, 12.0], [14.0, 16.0, 18.0, 20.0]])
        model.fc.weight.data.copy_(w2)
        swa.update(model)

        expected_shadow = (w1 + w2) / 2.0

        # Before context: model has instantaneous w2
        self.assertTrue(torch.allclose(model.fc.weight, w2))

        # In context: model has shadow weights
        with swa.apply_shadow(model):
            self.assertTrue(torch.allclose(model.fc.weight, expected_shadow))

        # After context: model has restored instantaneous w2
        self.assertTrue(torch.allclose(model.fc.weight, w2))

    def test_apply_shadow_noop_when_empty(self):
        """Verify that apply_shadow does not alter model when n_models == 0."""
        model = SimpleToyModel()
        swa = ModelSWA(model)
        w_orig = model.fc.weight.clone()

        with swa.apply_shadow(model):
            self.assertTrue(torch.allclose(model.fc.weight, w_orig))

        self.assertTrue(torch.allclose(model.fc.weight, w_orig))

    def test_state_dict_serialization(self):
        """Verify state_dict serialization and load_state_dict round-trip."""
        model = SimpleToyModel()
        swa = ModelSWA(model)
        swa.update(model)
        swa.update(model)

        sd = swa.state_dict()
        self.assertEqual(sd['n_models'], 2)
        self.assertIn('fc.weight', sd['params'])

        new_swa = ModelSWA(model)
        new_swa.load_state_dict(sd)
        self.assertEqual(new_swa.n_models, 2)
        self.assertTrue(torch.allclose(new_swa.shadow_params['fc.weight'], swa.shadow_params['fc.weight']))

    def test_reset(self):
        """Verify reset() clears accumulated weights and resets counter."""
        model = SimpleToyModel()
        swa = ModelSWA(model)
        swa.update(model)
        self.assertEqual(swa.n_models, 1)

        swa.reset()
        self.assertEqual(swa.n_models, 0)

    def test_checkpoint_use_raw_toggle(self):
        """Verify that load_checkpoint loads smoothed weights by default and raw weights when use_raw=True."""
        model_cfg = {
            'type': 'toy',
            'img_size': 32,
            'patch_size': 8,
            'embed_dim': 16,
            'n_outputs': 2,
        }
        model = SLPCWFS(img_size=32, patch_size=8, embed_dim=16, n_outputs=2)
        raw_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Create smoothed weights with +1.0 offset
        smoothed_state = {k: v.clone() + 1.0 for k, v in model.state_dict().items()}

        ckpt = {
            'epoch': 10,
            'model_state': smoothed_state,
            'raw_model_state': raw_state,
            'swa_state': {'n_models': 5, 'params': smoothed_state, 'buffers': {}},
            'config': {'model': model_cfg},
            'label_mean': np.array([0.0, 0.0], dtype=np.float32),
            'label_std': np.array([1.0, 1.0], dtype=np.float32),
            'n_modes': 2,
            'trained_modes': [4, 5],
            'history': {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_swa_ckpt.pt"
            torch.save(ckpt, ckpt_path)

            # 1. Default (use_raw=False): loads model_state (smoothed)
            loaded_default, meta_default = load_checkpoint(ckpt_path, torch.device("cpu"), use_raw=False)
            self.assertTrue(torch.allclose(loaded_default.fc_output.weight, smoothed_state['fc_output.weight']))
            self.assertTrue(meta_default['has_swa'])
            self.assertFalse(meta_default['using_raw'])

            # 2. use_raw=True: loads raw_model_state (instantaneous)
            loaded_raw, meta_raw = load_checkpoint(ckpt_path, torch.device("cpu"), use_raw=True)
            self.assertTrue(torch.allclose(loaded_raw.fc_output.weight, raw_state['fc_output.weight']))
            self.assertTrue(meta_raw['has_swa'])
            self.assertTrue(meta_raw['using_raw'])

    def test_resolve_swa_config(self):
        """Verify _resolve_swa parses diverse config formats and defaults correctly."""
        # Default: disabled, tail_epochs=5
        enabled, tail = _resolve_swa({})
        self.assertFalse(enabled)
        self.assertEqual(tail, 5)

        # Explicit bool True
        enabled, tail = _resolve_swa({'training': {'swa': True, 'swa_tail_epochs': 8}})
        self.assertTrue(enabled)
        self.assertEqual(tail, 8)

        # String 'true'
        enabled, tail = _resolve_swa({'training': {'swa': 'true', 'swa_tail_epochs': '10'}})
        self.assertTrue(enabled)
        self.assertEqual(tail, 10)

        # String 'false'
        enabled, tail = _resolve_swa({'training': {'swa': 'false'}})
        self.assertFalse(enabled)


if __name__ == '__main__':
    unittest.main()

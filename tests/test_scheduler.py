"""
Unit tests for learning rate schedulers and configuration resolvers.
"""

import unittest
import torch
import torch.nn as nn
from copy import deepcopy

from nn_WFS.train import (
    _resolve_scheduler_config,
    build_lr_scheduler,
    WarmupReduceLROnPlateau,
)


class TestSchedulerConfigResolution(unittest.TestCase):
    """Tests for _resolve_scheduler_config helper."""

    def test_default_config_omitted(self):
        cfg = {'training': {'lr': 3e-4, 'epochs': 100}}
        res = _resolve_scheduler_config(cfg)
        self.assertEqual(res['type'], 'constant')
        self.assertEqual(res['min_lr'], 1.0e-6)
        self.assertEqual(res['patience'], 2)
        self.assertEqual(res['factor'], 0.5)
        self.assertEqual(res['warmup_steps'], 500)

    def test_string_scheduler_type(self):
        cfg = {
            'training': {
                'lr': 3e-4,
                'scheduler': 'cosine',
                'min_lr': 5.0e-5,
                'warmup_steps': 200,
            }
        }
        res = _resolve_scheduler_config(cfg)
        self.assertEqual(res['type'], 'cosine')
        self.assertEqual(res['min_lr'], 5.0e-5)
        self.assertEqual(res['warmup_steps'], 200)

    def test_structured_dict_config(self):
        cfg = {
            'training': {
                'lr': 6e-4,
                'warmup_steps': 1000,
                'scheduler': {
                    'type': 'plateau',
                    'min_lr': 1.0e-5,
                    'patience': 4,
                    'factor': 0.2,
                    'warmup_steps': 250,
                },
            }
        }
        res = _resolve_scheduler_config(cfg)
        self.assertEqual(res['type'], 'plateau')
        self.assertEqual(res['min_lr'], 1.0e-5)
        self.assertEqual(res['patience'], 4)
        self.assertEqual(res['factor'], 0.2)
        self.assertEqual(res['warmup_steps'], 250)

    def test_lr_scheduler_alias(self):
        cfg = {
            'training': {
                'lr': 1e-4,
                'lr_scheduler': 'cosine',
            }
        }
        res = _resolve_scheduler_config(cfg)
        self.assertEqual(res['type'], 'cosine')


class TestConstantScheduler(unittest.TestCase):
    """Tests for constant scheduler with linear warmup."""

    def test_warmup_curve_and_flat_phase(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=1.0e-3)
        sched_cfg = {
            'type': 'constant',
            'warmup_steps': 5,
            'min_lr': 1e-6,
        }
        sched = build_lr_scheduler(opt, sched_cfg, steps_per_epoch=10, total_epochs=2)

        # Initial LR before any batch update (used for batch 0)
        self.assertAlmostEqual(sched.get_last_lr()[0], 2.0e-4, places=7)

        # After each batch update (LR for subsequent batches)
        expected_after_step = [4.0e-4, 6.0e-4, 8.0e-4, 1.0e-3, 1.0e-3]
        for step_idx in range(5):
            opt.step()
            sched.step()
            self.assertAlmostEqual(sched.get_last_lr()[0], expected_after_step[step_idx], places=7)

        # Post-warmup flat phase
        for _ in range(10):
            opt.step()
            sched.step()
            self.assertAlmostEqual(sched.get_last_lr()[0], 1.0e-3, places=7)

    def test_zero_warmup_constant(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=3.0e-4)
        sched_cfg = {'type': 'constant', 'warmup_steps': 0}
        sched = build_lr_scheduler(opt, sched_cfg, steps_per_epoch=5, total_epochs=2)

        for _ in range(10):
            opt.step()
            sched.step()
            self.assertAlmostEqual(sched.get_last_lr()[0], 3.0e-4, places=7)


class TestCosineScheduler(unittest.TestCase):
    """Tests for cosine scheduler with warmup and min_lr floor."""

    def test_cosine_decay_curve(self):
        p = nn.Parameter(torch.zeros(1))
        base_lr = 1.0e-3
        min_lr = 1.0e-5
        warmup_steps = 10
        total_steps = 100
        opt = torch.optim.Adam([p], lr=base_lr)
        sched_cfg = {
            'type': 'cosine',
            'warmup_steps': warmup_steps,
            'min_lr': min_lr,
        }
        sched = build_lr_scheduler(opt, sched_cfg, steps_per_epoch=20, total_epochs=5)

        lrs = []
        for step in range(total_steps):
            opt.step()
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        # Step 9 (end of warmup): should reach base_lr
        self.assertAlmostEqual(lrs[9], base_lr, places=6)

        # Monotonic decrease during cosine decay phase
        for i in range(warmup_steps, total_steps - 1):
            self.assertLess(lrs[i + 1], lrs[i])

        # Final step reaches min_lr
        self.assertAlmostEqual(lrs[-1], min_lr, places=7)

        # Midpoint of decay (step 55): progress = 45/90 = 0.5 -> cos(pi/2) = 0 -> (base + min)/2
        mid_step_idx = 54  # 0-indexed step 54 corresponds to after 55 calls
        expected_mid = (base_lr + min_lr) / 2.0
        self.assertAlmostEqual(lrs[mid_step_idx], expected_mid, delta=0.02 * base_lr)

    def test_cosine_floor_beyond_total_steps(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=1.0e-3)
        sched_cfg = {'type': 'cosine', 'warmup_steps': 2, 'min_lr': 1.0e-6}
        sched = build_lr_scheduler(opt, sched_cfg, steps_per_epoch=5, total_epochs=2)  # 10 steps total

        for _ in range(15):
            opt.step()
            sched.step()

        # Should remain clamped at min_lr
        self.assertAlmostEqual(sched.get_last_lr()[0], 1.0e-6, places=7)


class TestPlateauScheduler(unittest.TestCase):
    """Tests for ReduceLROnPlateau with optional warmup."""

    def test_plateau_warmup_and_step_down(self):
        p = nn.Parameter(torch.zeros(1))
        base_lr = 1.0e-3
        min_lr = 1.0e-4
        factor = 0.5
        patience = 1
        opt = torch.optim.Adam([p], lr=base_lr)
        sched_cfg = {
            'type': 'plateau',
            'warmup_steps': 4,
            'factor': factor,
            'patience': patience,
            'min_lr': min_lr,
        }
        sched = build_lr_scheduler(opt, sched_cfg, steps_per_epoch=10, total_epochs=10)
        self.assertIsInstance(sched, WarmupReduceLROnPlateau)

        # Initial LR before any batch update (batch 0)
        self.assertAlmostEqual(sched.get_last_lr()[0], 2.5e-4, places=7)

        # After each batch update (LR for subsequent batches)
        warmup_after_step = [5.0e-4, 7.5e-4, 1.0e-3, 1.0e-3]
        for i in range(4):
            sched.step_batch()
            self.assertAlmostEqual(sched.get_last_lr()[0], warmup_after_step[i], places=7)

        # Validation steps: improving metric
        sched.step(50.0)
        self.assertAlmostEqual(sched.get_last_lr()[0], 1.0e-3, places=7)
        sched.step(45.0)  # new best
        self.assertAlmostEqual(sched.get_last_lr()[0], 1.0e-3, places=7)

        # Stagnant metric: bad epoch 1 (patience is 1, so bad epoch 1 does not decay yet)
        sched.step(46.0)
        self.assertAlmostEqual(sched.get_last_lr()[0], 1.0e-3, places=7)

        # Bad epoch 2: triggers decay by factor 0.5 -> 5.0e-4
        sched.step(47.0)
        self.assertAlmostEqual(sched.get_last_lr()[0], 5.0e-4, places=7)

        # Ensure future batch steps do NOT reset the decayed lr
        sched.step_batch()
        self.assertAlmostEqual(sched.get_last_lr()[0], 5.0e-4, places=7)

        # Another round of stagnation: bad 1, bad 2 -> decays to 2.5e-4
        sched.step(48.0)
        sched.step(49.0)
        self.assertAlmostEqual(sched.get_last_lr()[0], 2.5e-4, places=7)

        # Decay to min_lr floor
        sched.step(50.0)
        sched.step(51.0)
        self.assertAlmostEqual(sched.get_last_lr()[0], 1.25e-4, places=7)
        sched.step(52.0)
        sched.step(53.0)
        self.assertAlmostEqual(sched.get_last_lr()[0], min_lr, places=7)


class TestSchedulerSerialization(unittest.TestCase):
    """Tests for checkpoint state dict saving and loading."""

    def test_cosine_checkpoint_recovery(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=3.0e-4)
        sched_cfg = {'type': 'cosine', 'warmup_steps': 5, 'min_lr': 1.0e-6}
        sched = build_lr_scheduler(opt, sched_cfg, steps_per_epoch=10, total_epochs=5)

        for _ in range(12):
            opt.step()
            sched.step()

        opt_sd = opt.state_dict()
        sched_sd = sched.state_dict()
        current_lr = sched.get_last_lr()[0]

        # Re-create fresh optimizer and scheduler
        p2 = nn.Parameter(torch.zeros(1))
        opt2 = torch.optim.Adam([p2], lr=3.0e-4)
        sched2 = build_lr_scheduler(opt2, sched_cfg, steps_per_epoch=10, total_epochs=5)

        opt2.load_state_dict(opt_sd)
        sched2.load_state_dict(sched_sd)

        self.assertAlmostEqual(sched2.get_last_lr()[0], current_lr, places=7)

        # Step both forward and compare
        opt.step()
        sched.step()
        opt2.step()
        sched2.step()
        self.assertAlmostEqual(sched2.get_last_lr()[0], sched.get_last_lr()[0], places=7)

    def test_plateau_checkpoint_recovery(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=1.0e-3)
        sched_cfg = {'type': 'plateau', 'warmup_steps': 2, 'factor': 0.5, 'patience': 1, 'min_lr': 1e-5}
        sched = build_lr_scheduler(opt, sched_cfg, steps_per_epoch=5, total_epochs=5)

        for _ in range(2):
            sched.step_batch()
        sched.step(10.0)
        sched.step(12.0)
        sched.step(13.0)  # decayed to 5e-4

        ckpt = {
            'optim': opt.state_dict(),
            'sched': sched.state_dict(),
        }

        p2 = nn.Parameter(torch.zeros(1))
        opt2 = torch.optim.Adam([p2], lr=1.0e-3)
        sched2 = build_lr_scheduler(opt2, sched_cfg, steps_per_epoch=5, total_epochs=5)

        opt2.load_state_dict(ckpt['optim'])
        sched2.load_state_dict(ckpt['sched'])

        self.assertAlmostEqual(sched2.get_last_lr()[0], 5.0e-4, places=7)
        # Next bad epochs decay to 2.5e-4
        sched2.step(14.0)
        sched2.step(15.0)
        self.assertAlmostEqual(sched2.get_last_lr()[0], 2.5e-4, places=7)

    def test_invalid_scheduler_raises(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=1e-3)
        with self.assertRaises(ValueError):
            build_lr_scheduler(opt, {'type': 'unsupported_scheduler'}, 10, 5)


if __name__ == '__main__':
    unittest.main()

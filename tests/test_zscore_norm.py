import sys
import unittest
from pathlib import Path
import tempfile
import h5py
import numpy as np
import torch

# Add project root and parent to sys.path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_GIT = _REPO.parent
for p in [str(_REPO), str(_GIT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from nn_WFS.dataset import CWFSDataset, compute_label_stats
    from nn_WFS.train import (
        _load_config, _apply_overrides, _resolve_z_score_labels, _resolve_log_loss,
        _resolve_loss_type, LogMSELoss, RMSWFELoss
    )
except ImportError:
    from dataset import CWFSDataset, compute_label_stats
    from train import (
        _load_config, _apply_overrides, _resolve_z_score_labels, _resolve_log_loss,
        _resolve_loss_type, LogMSELoss, RMSWFELoss
    )


class TestZScoreNormConfig(unittest.TestCase):
    def test_default_is_false(self):
        # When missing from config completely, defaults to False
        cfg = {'data': {}, 'training': {}}
        self.assertFalse(_resolve_z_score_labels(cfg))
        self.assertFalse(_resolve_z_score_labels({}))

    def test_resolve_from_data_section(self):
        cfg_false = {'data': {'z_score_labels': False}}
        self.assertFalse(_resolve_z_score_labels(cfg_false))

        cfg_true = {'data': {'z_score_labels': True}}
        self.assertTrue(_resolve_z_score_labels(cfg_true))

    def test_resolve_from_training_section(self):
        cfg_false = {'training': {'z_score_labels': False}}
        self.assertFalse(_resolve_z_score_labels(cfg_false))

        cfg_true = {'training': {'z_score_labels': True}}
        self.assertTrue(_resolve_z_score_labels(cfg_true))

    def test_resolve_aliases(self):
        # normalize_labels alias
        self.assertTrue(_resolve_z_score_labels({'data': {'normalize_labels': True}}))
        self.assertFalse(_resolve_z_score_labels({'training': {'normalize_labels': False}}))

    def test_string_coercion(self):
        self.assertTrue(_resolve_z_score_labels({'data': {'z_score_labels': 'true'}}))
        self.assertTrue(_resolve_z_score_labels({'data': {'z_score_labels': 'yes'}}))
        self.assertFalse(_resolve_z_score_labels({'data': {'z_score_labels': 'false'}}))
        self.assertFalse(_resolve_z_score_labels({'data': {'z_score_labels': 'no'}}))

    def test_apply_overrides_bool_conversion(self):
        cfg = {'data': {'z_score_labels': True}}
        _apply_overrides(cfg, ['data.z_score_labels=false'])
        self.assertIs(cfg['data']['z_score_labels'], False)

        _apply_overrides(cfg, ['data.z_score_labels=true'])
        self.assertIs(cfg['data']['z_score_labels'], True)

    def test_rodcnn_yaml_contains_z_score_labels(self):
        config_path = _REPO / 'config' / 'rodcnn.yaml'
        if config_path.exists():
            cfg = _load_config(str(config_path))
            self.assertIn('z_score_labels', cfg['data'])
            self.assertIsInstance(cfg['data']['z_score_labels'], bool)


class TestZScoreNormDatasetIntegration(unittest.TestCase):
    def setUp(self):
        # Create a small temporary HDF5 file
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.h5_path = str(Path(self.tmp_dir.name) / "test_cwfs.h5")
        self.n_samples = 6
        self.n_modes = 14
        self.T = 2
        self.H = 32
        self.W = 32

        np.random.seed(42)
        psfs = np.random.randn(self.n_samples, 2, self.T, self.H, self.W).astype(np.float16)
        # Non-zero mean and non-unit variance labels in meters
        self.raw_labels = (np.random.randn(self.n_samples, self.n_modes) * 50e-9 + 10e-9).astype(np.float32)

        with h5py.File(self.h5_path, 'w') as f:
            f.create_dataset('psfs', data=psfs)
            f.create_dataset('labels', data=self.raw_labels)

        self.indices = np.arange(self.n_samples)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_dataset_unnormalized_when_label_stats_none(self):
        # When label_stats is None, labels should be returned unnormalized (raw physical units)
        ds = CWFSDataset(self.h5_path, self.indices, label_stats=None, return_stacks=True)
        item0 = ds[0]
        np.testing.assert_allclose(item0['labels'].numpy(), self.raw_labels[0], rtol=1e-5)

    def test_dataset_normalized_when_label_stats_provided(self):
        stats = compute_label_stats(self.h5_path, self.indices)
        ds = CWFSDataset(self.h5_path, self.indices, label_stats=stats, return_stacks=True)
        item0 = ds[0]
        expected_norm = (self.raw_labels[0] - stats['mean']) / (stats['std'] + 1e-8)
        np.testing.assert_allclose(item0['labels'].numpy(), expected_norm, rtol=1e-5)

    def test_identity_denormalization_when_disabled(self):
        # When disabled, label_mean=0 and label_std=1
        lm = torch.zeros(self.n_modes, dtype=torch.float32)
        ls = torch.ones(self.n_modes, dtype=torch.float32)

        raw_pred = torch.from_numpy(self.raw_labels[0])
        denorm_pred = raw_pred * ls + lm
        torch.testing.assert_close(denorm_pred, raw_pred)


class TestLogLossConfig(unittest.TestCase):
    def test_default_is_false(self):
        self.assertFalse(_resolve_log_loss({}))
        self.assertFalse(_resolve_log_loss({'training': {}, 'data': {}}))

    def test_resolve_from_training_section(self):
        self.assertTrue(_resolve_log_loss({'training': {'log_loss': True}}))
        self.assertFalse(_resolve_log_loss({'training': {'log_loss': False}}))

    def test_resolve_from_data_section(self):
        self.assertTrue(_resolve_log_loss({'data': {'log_loss': True}}))
        self.assertFalse(_resolve_log_loss({'data': {'log_loss': False}}))

    def test_resolve_aliases(self):
        self.assertTrue(_resolve_log_loss({'training': {'ln_loss': True}}))
        self.assertTrue(_resolve_log_loss({'training': {'log_mse': True}}))
        self.assertTrue(_resolve_log_loss({'data': {'ln_loss': True}}))

    def test_string_coercion(self):
        self.assertTrue(_resolve_log_loss({'training': {'log_loss': 'true'}}))
        self.assertTrue(_resolve_log_loss({'training': {'log_loss': 'yes'}}))
        self.assertFalse(_resolve_log_loss({'training': {'log_loss': 'false'}}))
        self.assertFalse(_resolve_log_loss({'training': {'log_loss': 'no'}}))

    def test_apply_overrides(self):
        cfg = {'training': {'log_loss': False}}
        _apply_overrides(cfg, ['training.log_loss=true'])
        self.assertIs(cfg['training']['log_loss'], True)
        self.assertTrue(_resolve_log_loss(cfg))

        _apply_overrides(cfg, ['training.log_loss=false'])
        self.assertIs(cfg['training']['log_loss'], False)
        self.assertFalse(_resolve_log_loss(cfg))

    def test_rodcnn_yaml_contains_log_loss(self):
        config_path = _REPO / 'config' / 'rodcnn.yaml'
        if config_path.exists():
            cfg = _load_config(str(config_path))
            self.assertIn('log_loss', cfg['training'])
            self.assertIsInstance(cfg['training']['log_loss'], bool)


class TestLogMSELoss(unittest.TestCase):
    def test_log_mse_mathematical_value(self):
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
        # error is 0.5 everywhere -> squared error is 0.25 -> MSE is 0.25
        # ln(0.25) = -1.38629436
        criterion = LogMSELoss()
        loss = criterion(pred, target)
        expected = np.log(0.25)
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_log_mse_small_values_clamped(self):
        # Exact match -> MSE = 0.0 -> clamped to eps
        pred = torch.zeros(2, 2)
        target = torch.zeros(2, 2)
        criterion = LogMSELoss(eps=1e-30)
        loss = criterion(pred, target)
        expected = np.log(1e-30)
        self.assertAlmostEqual(loss.item(), expected, places=3)
        self.assertTrue(torch.isfinite(loss))

    def test_log_mse_backward_gradient(self):
        x = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
        target = torch.tensor([1.0, 1.0])
        criterion = LogMSELoss()
        loss = criterion(x, target)
        loss.backward()
        # MSE = ((2-1)^2 + (3-1)^2)/2 = (1 + 4)/2 = 2.5
        # ln(MSE) gradient wrt x: (x - target) / (len * MSE) = (x - target) / 2.5
        expected_grad = (torch.tensor([2.0, 3.0]) - target) / 2.5
        torch.testing.assert_close(x.grad, expected_grad)


class TestRMSWFELoss(unittest.TestCase):
    def test_rms_wfe_mathematical_value(self):
        pred = torch.tensor([[3.0, 4.0]])
        target = torch.tensor([[0.0, 0.0]])
        # diff = [3, 4], sum of sq = 9 + 16 = 25, sqrt = 5.0
        criterion = RMSWFELoss(eps=1e-8)
        loss = criterion(pred, target)
        self.assertAlmostEqual(loss.item(), 5.0, places=5)

    def test_rms_wfe_batch_mean(self):
        pred = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        target = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        # sample 1 WFE = 5.0, sample 2 WFE = 1.0 -> mean = 3.0
        criterion = RMSWFELoss(eps=1e-8)
        loss = criterion(pred, target)
        self.assertAlmostEqual(loss.item(), 3.0, places=5)

    def test_rms_wfe_backward_gradient(self):
        x = torch.nn.Parameter(torch.tensor([[3.0, 4.0]]))
        target = torch.tensor([[0.0, 0.0]])
        criterion = RMSWFELoss(eps=1e-8)
        loss = criterion(x, target)
        loss.backward()
        # grad = (x - target) / sqrt(sum(x^2)) = [3/5, 4/5] = [0.6, 0.8]
        expected_grad = torch.tensor([[0.6, 0.8]])
        torch.testing.assert_close(x.grad, expected_grad)

    def test_nanometer_dataset_scaling(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = str(Path(tmp) / "test_scale.h5")
            psfs = np.random.randn(2, 2, 2, 16, 16).astype(np.float16)
            raw_labels = np.array([[50e-9, -20e-9], [100e-9, -50e-9]], dtype=np.float32)
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('psfs', data=psfs)
                f.create_dataset('labels', data=raw_labels)

            ds = CWFSDataset(h5_path, [0, 1], label_stats=None, label_scale=1e9, return_stacks=True)
            item0 = ds[0]
            # 50e-9 * 1e9 = 50.0 nm, -20e-9 * 1e9 = -20.0 nm
            np.testing.assert_allclose(item0['labels'].numpy(), [50.0, -20.0], rtol=1e-5)


class TestLossTypeConfig(unittest.TestCase):
    def test_resolve_explicit_wfe_rms(self):
        self.assertEqual(_resolve_loss_type({'training': {'loss_type': 'wfe_rms'}}), 'wfe_rms')
        self.assertEqual(_resolve_loss_type({'training': {'loss_type': 'rms_wfe'}}), 'wfe_rms')
        self.assertEqual(_resolve_loss_type({'training': {'loss_type': 'wfe'}}), 'wfe_rms')

    def test_resolve_defaults_to_wfe_rms_when_z_score_false(self):
        cfg = {'data': {'z_score_labels': False}}
        self.assertEqual(_resolve_loss_type(cfg), 'wfe_rms')

    def test_resolve_defaults_to_mse_when_z_score_true(self):
        cfg = {'data': {'z_score_labels': True}}
        self.assertEqual(_resolve_loss_type(cfg), 'mse')

    def test_resolve_log_loss_fallback(self):
        cfg = {'training': {'log_loss': True}}
        self.assertEqual(_resolve_loss_type(cfg), 'log_mse')


if __name__ == '__main__':
    unittest.main()

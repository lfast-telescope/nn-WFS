import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

# Add project root and parent to sys.path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_GIT = _REPO.parent
for p in [str(_REPO), str(_GIT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import yaml
import evaluate

from nn_WFS.train import (
    CheckpointManager,
    read_seed_summary,
    write_seed_summary,
    _check_seed_completed,
    _parse_args,
    _apply_overrides,
)


class TestCheckpointManagerScan(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_resume(self):
        mgr = CheckpointManager(str(self.temp_dir))
        self.assertIsNone(mgr.load_resume(torch.device('cpu')))

        mgr.save_resume({'epoch': 5, 'model_state': {}})
        loaded = mgr.load_resume(torch.device('cpu'))
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['epoch'], 5)

        # Overwriting resume.pt should replace, not accumulate, files
        mgr.save_resume({'epoch': 6, 'model_state': {}})
        self.assertEqual(list(self.temp_dir.glob('*.pt')), [mgr.resume_path])
        loaded = mgr.load_resume(torch.device('cpu'))
        self.assertEqual(loaded['epoch'], 6)

        mgr.clear_resume()
        self.assertIsNone(mgr.load_resume(torch.device('cpu')))

    def test_save_final_replaces_previous(self):
        mgr = CheckpointManager(str(self.temp_dir))
        self.assertIsNone(mgr.final_path())

        path1 = mgr.save_final({'epoch': 12, 'model_state': {}}, metric=38.1e-9)
        self.assertTrue(Path(path1).exists())
        self.assertEqual(mgr.final_path(), path1)

        # A better final checkpoint should replace the earlier one on disk
        path2 = mgr.save_final({'epoch': 15, 'model_state': {}}, metric=30.0e-9)
        self.assertFalse(Path(path1).exists())
        self.assertTrue(Path(path2).exists())
        self.assertEqual(mgr.final_path(), path2)
        self.assertEqual(len(list(self.temp_dir.glob('final_wfe*nm.pt'))), 1)

    def test_empty_dir(self):
        mgr = CheckpointManager(str(self.temp_dir))
        self.assertIsNone(mgr.load_resume(torch.device('cpu')))
        self.assertIsNone(mgr.final_path())


class TestSeedSummaryIO(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_write_seed_summary(self):
        self.assertIsNone(read_seed_summary(self.temp_dir))

        data = {
            'seed': 42,
            'status': 'COMPLETED',
            'best_val_wfe': 3.32e-8,
            'best_val_wfe_nm': 33.2,
            'best_epoch': 28,
            'best_ckpt_path': str(self.temp_dir / "final_wfe33.2nm.pt"),
            'completed_epochs': 30,
            'target_epochs': 30,
            'elapsed_s': 7357.1,
        }
        write_seed_summary(self.temp_dir, data)
        loaded = read_seed_summary(self.temp_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['seed'], 42)
        self.assertEqual(loaded['status'], 'COMPLETED')
        self.assertEqual(loaded['completed_epochs'], 30)
        self.assertAlmostEqual(loaded['best_val_wfe_nm'], 33.2, places=1)


class TestCheckSeedCompleted(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(tempfile.mkdtemp())
        self.seed_dir = self.base_dir / "seed_42"
        self.seed_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_completed_via_seed_summary(self):
        ckpt = self.seed_dir / "final_wfe30.0nm.pt"
        torch.save({'epoch': 30}, ckpt)

        data = {
            'seed': 42,
            'status': 'COMPLETED',
            'best_val_wfe': 3.0e-8,
            'best_val_wfe_nm': 30.0,
            'best_epoch': 30,
            'best_ckpt_path': str(ckpt),
            'completed_epochs': 30,
            'target_epochs': 30,
            'elapsed_s': 500.0,
        }
        write_seed_summary(self.seed_dir, data)

        res = _check_seed_completed(self.seed_dir, 42, target_epochs=30)
        self.assertIsNotNone(res)
        self.assertEqual(res['seed'], 42)
        self.assertEqual(res['status'], 'COMPLETED')

        # If target_epochs is higher than completed, returns None
        res_higher = _check_seed_completed(self.seed_dir, 42, target_epochs=50)
        self.assertIsNone(res_higher)

        # If checkpoint file was deleted, returns None
        ckpt.unlink()
        res_missing = _check_seed_completed(self.seed_dir, 42, target_epochs=30)
        self.assertIsNone(res_missing)

    def test_completed_via_parent_ensemble_summary(self):
        ckpt = self.seed_dir / "final_wfe30.0nm.pt"
        torch.save({'epoch': 30}, ckpt)

        ens_summary_path = self.base_dir / "ensemble_summary.yaml"
        ens_data = {
            'runs': [
                {
                    'seed': 42,
                    'best_val_wfe_nm': 30.0,
                    'best_epoch': 30,
                    'checkpoint': str(ckpt),
                }
            ]
        }
        with open(ens_summary_path, 'w') as f:
            yaml.safe_dump(ens_data, f)

        res = _check_seed_completed(self.seed_dir, 42, target_epochs=30)
        self.assertIsNotNone(res)
        self.assertEqual(res['seed'], 42)
        self.assertEqual(res['status'], 'COMPLETED')
        self.assertEqual(res['best_ckpt_path'], str(ckpt))

        # Check that seed_summary.yaml was generated
        summary = read_seed_summary(self.seed_dir)
        self.assertIsNotNone(summary)
        self.assertEqual(summary['seed'], 42)

    def test_completed_via_log_and_checkpoint_fallback(self):
        ckpt = self.seed_dir / "final_wfe33.2nm.pt"
        torch.save({'epoch': 28}, ckpt)

        trial_dir = self.base_dir / "trial_001"
        logs_dir = trial_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "sparse_log.txt"
        with open(log_file, 'w') as f:
            f.write("[+ 7357.1s] [TRAIN] epoch s42_e30/30: loss=0.001\n")

        # Point seed_dir inside trial_dir/checkpoints/seed_42
        ckpt_dir = trial_dir / "checkpoints" / "seed_42"
        ckpt_dir.mkdir(parents=True)
        ckpt_in_trial = ckpt_dir / "final_wfe33.2nm.pt"
        torch.save({'epoch': 28}, ckpt_in_trial)

        res = _check_seed_completed(ckpt_dir, 42, target_epochs=30, trial_dir=trial_dir)
        self.assertIsNotNone(res)
        self.assertEqual(res['seed'], 42)
        self.assertEqual(res['status'], 'COMPLETED')
        self.assertEqual(res['best_epoch'], 28)
        self.assertAlmostEqual(res['best_val_wfe_nm'], 33.2, places=1)
        self.assertAlmostEqual(res['elapsed_s'], 7357.1, places=1)

        # Also verified seed_summary.yaml was created
        summary = read_seed_summary(ckpt_dir)
        self.assertIsNotNone(summary)
        self.assertEqual(summary['seed'], 42)


class TestCLIFlagsAndOverrides(unittest.TestCase):
    def test_cli_resume_flags_parsing(self):
        sys_argv_backup = sys.argv[:]
        try:
            sys.argv = ['train.py', '--config', 'dummy.yaml', '--resume_seeds', '--no_resume_epochs']
            args, overrides = _parse_args()
            self.assertTrue(args.resume_seeds)
            self.assertFalse(args.resume_epochs)

            cfg = {'training': {}}
            if args.resume_seeds is not None:
                overrides.append(f'training.resume_seeds={str(args.resume_seeds).lower()}')
            if args.resume_epochs is not None:
                overrides.append(f'training.resume_epochs={str(args.resume_epochs).lower()}')

            cfg = _apply_overrides(cfg, overrides)
            self.assertTrue(cfg['training']['resume_seeds'])
            self.assertFalse(cfg['training']['resume_epochs'])
        finally:
            sys.argv = sys_argv_backup


from unittest.mock import patch, MagicMock


class TestTrainSeedSkipping(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ckpt_dir = self.temp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('evaluate.load_checkpoint')
    @patch('evaluate._predict')
    @patch('evaluate._metrics_table')
    @patch('nn_WFS.train.CWFSDataset')
    @patch('nn_WFS.train.train_val_test_split')
    @patch('nn_WFS.train.get_n_modes')
    @patch('nn_WFS.train._train_single_seed')
    def test_train_skips_completed_seed_when_resume_enabled(
        self,
        mock_train_single,
        mock_get_n_modes,
        mock_split,
        mock_ds,
        mock_metrics_table,
        mock_predict,
        mock_load_ckpt,
    ):
        mock_get_n_modes.return_value = 10
        mock_split.return_value = ([0], [0], [0])
        mock_predict.return_value = (torch.zeros(1, 10), torch.zeros(1, 10))
        mock_metrics_table.return_value = {'wfe_rms_nm': 30.0, 'strehl': 0.95, 'mode_rms_nm': [10.0, 10.0, 10.0]}
        mock_load_ckpt.return_value = (MagicMock(), None)

        # Mark seed 42 as completed
        s42_dir = self.ckpt_dir / "seed_42"
        s42_dir.mkdir(parents=True)
        ckpt_42 = s42_dir / "final_wfe30.0nm.pt"
        torch.save({'epoch': 30}, ckpt_42)
        write_seed_summary(s42_dir, {
            'seed': 42,
            'status': 'COMPLETED',
            'best_val_wfe': 3.0e-8,
            'best_val_wfe_nm': 30.0,
            'best_epoch': 30,
            'best_ckpt_path': str(ckpt_42),
            'completed_epochs': 30,
            'target_epochs': 30,
            'elapsed_s': 100.0,
        })

        mock_train_single.return_value = {
            'seed': 276,
            'best_val_wfe': 3.5e-8,
            'best_epoch': 25,
            'best_ckpt_path': str(ckpt_42),
            'ckpt_dir': str(self.ckpt_dir / "seed_276"),
            'elapsed_s': 200.0,
        }

        cfg = {
            'data': {'hdf5_path': 'dummy.h5'},
            'model': {'type': 'cnn', 'trained_modes': [4, 5, 6]},
            'ensemble': {
                'seeds': [42, 276],
                'mode': 'best',
            },
            'training': {
                'epochs': 30,
                'resume_seeds': True,
                'resume_epochs': True,
            },
            'logging': {
                'checkpoint_dir': str(self.ckpt_dir),
                'sparse_record': False,
            },
        }

        from nn_WFS.train import train
        train(cfg)

        # Seed 42 should be skipped; seed 276 should be trained
        self.assertEqual(mock_train_single.call_count, 1)
        call_kwargs = mock_train_single.call_args[1]
        self.assertEqual(call_kwargs['run_seed'], 276)
        self.assertTrue(call_kwargs['resume_checkpoint'])

        # Check ensemble_summary.yaml
        ens_summary = self.ckpt_dir / "ensemble_summary.yaml"
        self.assertTrue(ens_summary.exists())
        with open(ens_summary) as f:
            data = yaml.safe_load(f)
        runs = data['runs']
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]['seed'], 42)
        self.assertEqual(runs[1]['seed'], 276)

    @patch('evaluate.load_checkpoint')
    @patch('evaluate._predict')
    @patch('evaluate._metrics_table')
    @patch('nn_WFS.train.CWFSDataset')
    @patch('nn_WFS.train.train_val_test_split')
    @patch('nn_WFS.train.get_n_modes')
    @patch('nn_WFS.train._train_single_seed')
    def test_train_retrains_all_when_resume_disabled(
        self,
        mock_train_single,
        mock_get_n_modes,
        mock_split,
        mock_ds,
        mock_metrics_table,
        mock_predict,
        mock_load_ckpt,
    ):
        mock_get_n_modes.return_value = 10
        mock_split.return_value = ([0], [0], [0])
        mock_predict.return_value = (torch.zeros(1, 10), torch.zeros(1, 10))
        mock_metrics_table.return_value = {'wfe_rms_nm': 30.0, 'strehl': 0.95, 'mode_rms_nm': [10.0, 10.0, 10.0]}
        mock_load_ckpt.return_value = (MagicMock(), None)

        s42_dir = self.ckpt_dir / "seed_42"
        s42_dir.mkdir(parents=True)
        ckpt_42 = s42_dir / "final_wfe30.0nm.pt"
        torch.save({'epoch': 30}, ckpt_42)
        write_seed_summary(s42_dir, {
            'seed': 42,
            'status': 'COMPLETED',
            'best_val_wfe': 3.0e-8,
            'best_val_wfe_nm': 30.0,
            'best_epoch': 30,
            'best_ckpt_path': str(ckpt_42),
            'completed_epochs': 30,
            'target_epochs': 30,
            'elapsed_s': 100.0,
        })

        mock_train_single.side_effect = [
            {
                'seed': 42,
                'best_val_wfe': 3.0e-8,
                'best_epoch': 30,
                'best_ckpt_path': str(ckpt_42),
                'ckpt_dir': str(s42_dir),
                'elapsed_s': 100.0,
            },
            {
                'seed': 276,
                'best_val_wfe': 3.5e-8,
                'best_epoch': 25,
                'best_ckpt_path': str(ckpt_42),
                'ckpt_dir': str(self.ckpt_dir / "seed_276"),
                'elapsed_s': 200.0,
            }
        ]

        cfg = {
            'data': {'hdf5_path': 'dummy.h5'},
            'model': {'type': 'cnn', 'trained_modes': [4, 5, 6]},
            'ensemble': {
                'seeds': [42, 276],
                'mode': 'best',
            },
            'training': {
                'epochs': 30,
                'resume_seeds': False,
                'resume_epochs': False,
            },
            'logging': {
                'checkpoint_dir': str(self.ckpt_dir),
                'sparse_record': False,
            },
        }

        from nn_WFS.train import train
        train(cfg)

        # When resume_seeds=False, both seeds must be trained
        self.assertEqual(mock_train_single.call_count, 2)
        call_kwargs1 = mock_train_single.call_args_list[0][1]
        self.assertEqual(call_kwargs1['run_seed'], 42)
        self.assertFalse(call_kwargs1['resume_checkpoint'])

        call_kwargs2 = mock_train_single.call_args_list[1][1]
        self.assertEqual(call_kwargs2['run_seed'], 276)
        self.assertFalse(call_kwargs2['resume_checkpoint'])


if __name__ == '__main__':
    unittest.main()

import unittest
import tempfile
import os
import shutil
from pathlib import Path

from nn_WFS.sweep import (
    parse_slurm_duration_to_seconds,
    format_seconds_to_slurm_time,
    estimate_trial_duration,
    select_active_job_trial,
    generate_slurm_script,
    resolve_max_slurm_workers,
    package_trials_into_workers,
    generate_packaged_slurm_script,
    TrialSpec,
)


class TestSlurmDispatch(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.manifest = {
            'name': 'test_sweep',
            'slurm': {
                'account': 'cbender',
                'partition': 'standard',
                'cpus_per_task': 4,
                'mem': '20G',
                'gres': 'gpu:1',
                'dataset_scale': 1.0,
                'time_per_epoch_s': 170.0,
            }
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_slurm_duration(self):
        self.assertEqual(parse_slurm_duration_to_seconds("45:20"), 2720.0)
        self.assertEqual(parse_slurm_duration_to_seconds("06:30:15"), 23415.0)
        self.assertEqual(parse_slurm_duration_to_seconds("1-02:15:00"), 94500.0)
        self.assertEqual(parse_slurm_duration_to_seconds("2-00:00:00"), 172800.0)
        self.assertIsNone(parse_slurm_duration_to_seconds(""))
        self.assertIsNone(parse_slurm_duration_to_seconds("INVALID"))

    def test_estimate_trial_duration_epochs_scaling(self):
        cfg_30 = {
            'training': {'epochs': 30},
            'ensemble': {'seeds': [42, 276]},
            'data': {'batch_size': 8},
            'model': {'k_pairs_train': 8}
        }
        cfg_60 = {
            'training': {'epochs': 60},
            'ensemble': {'seeds': [42, 276]},
            'data': {'batch_size': 8},
            'model': {'k_pairs_train': 8}
        }

        dur_30, wt_30 = estimate_trial_duration(cfg_30, self.manifest, repo_root=self.root)
        dur_60, wt_60 = estimate_trial_duration(cfg_60, self.manifest, repo_root=self.root)

        self.assertGreater(dur_60, dur_30)
        self.assertIn(":", wt_30)
        self.assertIn(":", wt_60)
        # Compute raw difference without fixed 3600s buffer: 60 epochs should have roughly double compute
        raw_30 = dur_30 - 3600.0
        raw_60 = dur_60 - 3600.0
        self.assertAlmostEqual(raw_60 / raw_30, 2.0, places=2)

    def test_estimate_trial_duration_dataset_scaling(self):
        cfg = {
            'training': {'epochs': 30},
            'ensemble': {'seeds': [42, 276]},
            'data': {'batch_size': 8},
            'model': {'k_pairs_train': 8}
        }
        manifest_1x = dict(self.manifest)
        manifest_1x['slurm'] = dict(self.manifest['slurm'])
        manifest_1x['slurm']['dataset_scale'] = 1.0

        manifest_2x = dict(self.manifest)
        manifest_2x['slurm'] = dict(self.manifest['slurm'])
        manifest_2x['slurm']['dataset_scale'] = 2.0

        dur_1x, _ = estimate_trial_duration(cfg, manifest_1x, repo_root=self.root)
        dur_2x, _ = estimate_trial_duration(cfg, manifest_2x, repo_root=self.root)

        raw_1x = dur_1x - 3600.0
        raw_2x = dur_2x - 3600.0
        self.assertAlmostEqual(raw_2x / raw_1x, 2.0, places=2)

    def test_select_active_job_trial_longest_feasible(self):
        trials = [
            TrialSpec(
                trial_id=1, trial_name="t1_30ep", overrides={},
                resolved_config={'training': {'epochs': 30}, 'ensemble': {'seeds': 2}},
                trial_dir=self.root / "t1"
            ),
            TrialSpec(
                trial_id=2, trial_name="t2_50ep", overrides={},
                resolved_config={'training': {'epochs': 50}, 'ensemble': {'seeds': 2}},
                trial_dir=self.root / "t2"
            ),
            TrialSpec(
                trial_id=3, trial_name="t3_75ep", overrides={},
                resolved_config={'training': {'epochs': 75}, 'ensemble': {'seeds': 2}},
                trial_dir=self.root / "t3"
            ),
            TrialSpec(
                trial_id=4, trial_name="t4_100ep", overrides={},
                resolved_config={'training': {'epochs': 100}, 'ensemble': {'seeds': 2}},
                trial_dir=self.root / "t4"
            ),
        ]

        # Calculate exact duration for 50 epochs with safety buffer
        dur_50, _ = estimate_trial_duration(trials[1].resolved_config, self.manifest, repo_root=self.root)
        dur_75, _ = estimate_trial_duration(trials[2].resolved_config, self.manifest, repo_root=self.root)

        # Give active job enough time for 50 epochs + buffer, but NOT enough for 75 epochs
        active_remaining_s = dur_50 + 2000.0  # safety_buffer_s default is 1800s
        self.assertLess(active_remaining_s - 1800.0, dur_75)

        selected, rem = select_active_job_trial(
            trials, active_remaining_s, self.manifest, safety_buffer_s=1800.0, repo_root=self.root
        )
        self.assertIsNotNone(selected)
        # Should pick trial 2 (50 epochs) as the longest feasible
        self.assertEqual(selected.trial_id, 2)
        self.assertEqual(len(rem), 3)
        self.assertEqual([t.trial_id for t in rem], [1, 3, 4])

    def test_select_active_job_trial_none_if_too_short(self):
        trials = [
            TrialSpec(
                trial_id=1, trial_name="t1_30ep", overrides={},
                resolved_config={'training': {'epochs': 30}, 'ensemble': {'seeds': 2}},
                trial_dir=self.root / "t1"
            ),
        ]
        # Only 1 hour remaining -> less than 30 epochs duration (~4.8h)
        selected, rem = select_active_job_trial(
            trials, remaining_s=3600.0, manifest=self.manifest, safety_buffer_s=1800.0, repo_root=self.root
        )
        self.assertIsNone(selected)
        self.assertEqual(len(rem), 1)

    def test_generate_slurm_script_headers_and_hooks(self):
        trial = TrialSpec(
            trial_id=1, trial_name="trial_001_test", overrides={},
            resolved_config={'training': {'epochs': 30}, 'ensemble': {'seeds': 2}},
            trial_dir=self.root / "trial_001"
        )
        trial.trial_dir.mkdir(parents=True, exist_ok=True)
        (trial.trial_dir / "logs").mkdir(parents=True, exist_ok=True)
        cfg_path = trial.trial_dir / "config.yaml"
        cfg_path.touch()

        sbatch_path = generate_slurm_script(trial, cfg_path, self.manifest, self.root)
        self.assertTrue(sbatch_path.exists())

        with open(sbatch_path) as f:
            content = f.read()

        # Check Slurm headers
        self.assertIn("#SBATCH --account=cbender", content)
        self.assertIn("#SBATCH --partition=standard", content)
        self.assertIn("#SBATCH --cpus-per-task=4", content)
        self.assertIn("#SBATCH --mem=20G", content)
        self.assertIn("#SBATCH --gres=gpu:1", content)
        self.assertIn("#SBATCH --time=", content)

        # Check module loading
        self.assertIn("module load python/3.14", content)

        # Check intermediate compare hook
        self.assertIn("nn_WFS.compare", content)

        # Test omitting memory directives per UA HPC best practice
        manifest_no_mem = {
            'name': 'test_sweep',
            'slurm': {
                'account': 'cbender',
                'partition': 'standard',
                'cpus_per_task': 4,
                'gres': 'gpu:1',
            }
        }
        sbatch_no_mem = generate_slurm_script(trial, cfg_path, manifest_no_mem, self.root)
        with open(sbatch_no_mem) as f:
            content_no_mem = f.read()
        self.assertNotIn("#SBATCH --mem=", content_no_mem)
        self.assertNotIn("#SBATCH --mem-per-cpu=", content_no_mem)

        # Test mem_per_cpu directive
        manifest_mpc = {
            'name': 'test_sweep',
            'slurm': {
                'account': 'cbender',
                'partition': 'standard',
                'cpus_per_task': 4,
                'mem_per_cpu': '5G',
                'gres': 'gpu:1',
            }
        }
        sbatch_mpc = generate_slurm_script(trial, cfg_path, manifest_mpc, self.root)
        with open(sbatch_mpc) as f:
            content_mpc = f.read()
        self.assertIn("#SBATCH --mem-per-cpu=5G", content_mpc)
        self.assertNotIn("#SBATCH --mem=", content_mpc)

        # Test new defaults: gpu_high_priority partition, specific A100 GRES, and user_qos_<account>
        manifest_defaults = {
            'name': 'test_sweep',
            'slurm': {}
        }
        sbatch_def = generate_slurm_script(trial, cfg_path, manifest_defaults, self.root)
        with open(sbatch_def) as f:
            content_def = f.read()
        self.assertIn("#SBATCH --partition=gpu_high_priority", content_def)
        self.assertIn("#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb", content_def)
        self.assertIn("#SBATCH --account=cbender", content_def)
        self.assertIn("#SBATCH --qos=user_qos_cbender", content_def)

    def test_resolve_max_slurm_workers(self):
        # Default limits
        self.assertEqual(resolve_max_slurm_workers({}, "gpu_high_priority"), 4)
        self.assertEqual(resolve_max_slurm_workers({}, "gpu_standard"), 4)

        # Active job in same partition family: reduces by 1
        self.assertEqual(
            resolve_max_slurm_workers({}, "gpu_standard", active_partition="gpu_standard", has_active_job=True),
            3
        )
        # Active job in standard does not reduce high_priority
        self.assertEqual(
            resolve_max_slurm_workers({}, "gpu_high_priority", active_partition="gpu_standard", has_active_job=True),
            4
        )

        # Manifest overrides
        manifest_mw = {'slurm': {'max_workers': 2}}
        self.assertEqual(resolve_max_slurm_workers(manifest_mw, "gpu_high_priority"), 2)

        manifest_mc = {'slurm': {'max_concurrent': 3}}
        self.assertEqual(resolve_max_slurm_workers(manifest_mc, "gpu_high_priority"), 3)

        # CLI override takes highest precedence
        self.assertEqual(resolve_max_slurm_workers(manifest_mw, "gpu_high_priority", cli_max_workers=1), 1)

    def test_package_trials_into_workers_lpt(self):
        trials = [
            TrialSpec(trial_id=1, trial_name="t1", overrides={},
                      resolved_config={'training': {'epochs': 30}, 'ensemble': {'seeds': 2}},
                      trial_dir=self.root / "t1"),
            TrialSpec(trial_id=2, trial_name="t2", overrides={},
                      resolved_config={'training': {'epochs': 40}, 'ensemble': {'seeds': 2}},
                      trial_dir=self.root / "t2"),
            TrialSpec(trial_id=3, trial_name="t3", overrides={},
                      resolved_config={'training': {'epochs': 50}, 'ensemble': {'seeds': 2}},
                      trial_dir=self.root / "t3"),
            TrialSpec(trial_id=4, trial_name="t4", overrides={},
                      resolved_config={'training': {'epochs': 60}, 'ensemble': {'seeds': 2}},
                      trial_dir=self.root / "t4"),
            TrialSpec(trial_id=5, trial_name="t5", overrides={},
                      resolved_config={'training': {'epochs': 70}, 'ensemble': {'seeds': 2}},
                      trial_dir=self.root / "t5"),
        ]

        # Packaging 5 trials across 4 workers
        packages = package_trials_into_workers(trials, max_workers=4, manifest=self.manifest, repo_root=self.root)
        self.assertEqual(len(packages), 4, "Expected exactly 4 worker packages for 5 trials with max_workers=4")
        total_packaged = sum(len(p) for p in packages)
        self.assertEqual(total_packaged, 5)

        # One worker should receive 2 trials, and three workers receive 1 trial
        counts = sorted([len(p) for p in packages])
        self.assertEqual(counts, [1, 1, 1, 2])

        # When trials <= max_workers, exactly len(trials) workers created
        packages_small = package_trials_into_workers(trials[:2], max_workers=4, manifest=self.manifest, repo_root=self.root)
        self.assertEqual(len(packages_small), 2)

    def test_generate_packaged_slurm_script(self):
        t1 = TrialSpec(trial_id=1, trial_name="t1", overrides={},
                       resolved_config={'training': {'epochs': 30}, 'ensemble': {'seeds': 2}},
                       trial_dir=self.root / "t1")
        t2 = TrialSpec(trial_id=2, trial_name="t2", overrides={},
                       resolved_config={'training': {'epochs': 40}, 'ensemble': {'seeds': 2}},
                       trial_dir=self.root / "t2")
        t1.trial_dir.mkdir(parents=True, exist_ok=True)
        t2.trial_dir.mkdir(parents=True, exist_ok=True)

        script_path = generate_packaged_slurm_script(
            worker_idx=1,
            total_workers=2,
            worker_trials=[t1, t2],
            manifest=self.manifest,
            repo_root=self.root,
            output_root=self.root,
        )

        self.assertTrue(script_path.exists())
        self.assertEqual(script_path.name, "worker_01_of_02.sbatch")

        with open(script_path) as f:
            content = f.read()

        # Check job name and output logs
        self.assertIn("#SBATCH --job-name=swp_test_sweep_w01", content)
        self.assertIn("slurm_worker_01_%j.out", content)
        self.assertIn("#SBATCH --time=", content)

        # Check that both trial configs appear in the sequential execution
        self.assertIn(str(t1.trial_dir / "config_resolved.yaml"), content)
        self.assertIn(str(t2.trial_dir / "config_resolved.yaml"), content)

        # Check intermediate compare calls
        self.assertIn("nn_WFS.compare --experiment_dir", content)


if __name__ == "__main__":
    unittest.main()


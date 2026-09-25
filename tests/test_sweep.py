"""
test_sweep.py — Unit tests for the sweep orchestration engine.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_GIT = _REPO.parent
for p in [str(_REPO), str(_GIT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from nn_WFS.sweep import (
    apply_dot_override,
    get_dot_value,
    slugify,
    format_override_slug,
    expand_sweep_trials,
    load_sweep_manifest,
    prepare_trial_dir,
    generate_slurm_script,
    write_trial_summary,
    read_trial_summary,
    TrialSpec,
)


class TestSweepDotHelpers(unittest.TestCase):
    def test_apply_and_get_dot_override(self):
        cfg = {"model": {"base_ch": 32}, "training": {"lr": 1e-4}}
        apply_dot_override(cfg, "training.lr", 3e-4)
        self.assertEqual(get_dot_value(cfg, "training.lr"), 3e-4)

        apply_dot_override(cfg, "training.scheduler.type", "cosine")
        self.assertEqual(get_dot_value(cfg, "training.scheduler.type"), "cosine")

        # Nested non-existent paths
        apply_dot_override(cfg, "data.augmentation.prob", 0.5)
        self.assertEqual(get_dot_value(cfg, "data.augmentation.prob"), 0.5)
        self.assertIsNone(get_dot_value(cfg, "data.non_existent"))

    def test_slugify(self):
        self.assertEqual(slugify("3.0e-04"), "3.0e-04")
        self.assertEqual(slugify("training.lr=3e-4 / test"), "training.lr_3e-4_test")

    def test_format_override_slug(self):
        overrides = {"training.lr": 0.0003, "data.batch_size": 4}
        slug = format_override_slug(overrides)
        self.assertIn("lr_3.0e-04", slug)
        self.assertIn("batch_size_4", slug)


class TestSweepExpansion(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.base_cfg_path = self.root / "base.yaml"
        with open(self.base_cfg_path, "w") as f:
            yaml.safe_dump({
                "model": {"type": "rodcnn", "base_ch": 32},
                "data": {"batch_size": 4, "split_seed": 42},
                "training": {"lr": 3e-4, "epochs": 50},
                "logging": {"checkpoint_dir": "checkpoints/default"},
            }, f)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_matrix_expansion_cartesian_product(self):
        manifest = {
            "name": "test_matrix_sweep",
            "base_config": str(self.base_cfg_path),
            "common_overrides": {
                "training.epochs": 5,
            },
            "matrix": {
                "training.lr": [1e-4, 3e-4],
                "data.batch_size": [4, 8],
            },
            "output_root": str(self.root / "results"),
        }
        with open(self.base_cfg_path) as f:
            base_cfg = yaml.safe_load(f)

        trials = expand_sweep_trials(manifest, base_cfg, repo_root=self.root)
        # 2 x 2 = 4 combinations
        self.assertEqual(len(trials), 4)

        # Check common overrides applied to all
        for t in trials:
            self.assertEqual(t.resolved_config["training"]["epochs"], 5)
            self.assertEqual(t.resolved_config["model"]["type"], "rodcnn")
            self.assertIn(t.resolved_config["training"]["lr"], [1e-4, 3e-4])
            self.assertIn(t.resolved_config["data"]["batch_size"], [4, 8])

        # Check unique trial directories
        dirs = [str(t.trial_dir) for t in trials]
        self.assertEqual(len(dirs), len(set(dirs)))

    def test_explicit_trials_expansion(self):
        manifest = {
            "name": "test_trials_sweep",
            "base_config": str(self.base_cfg_path),
            "trials": [
                {
                    "name": "trial_a",
                    "overrides": {"training.lr": 1e-4, "data.batch_size": 2},
                },
                {
                    "name": "trial_b",
                    "overrides": {"training.lr": 5e-4, "data.batch_size": 16},
                },
            ],
            "output_root": str(self.root / "results"),
        }
        with open(self.base_cfg_path) as f:
            base_cfg = yaml.safe_load(f)

        trials = expand_sweep_trials(manifest, base_cfg, repo_root=self.root)
        self.assertEqual(len(trials), 2)
        self.assertIn("trial_a", trials[0].trial_name)
        self.assertIn("trial_b", trials[1].trial_name)
        self.assertEqual(trials[0].resolved_config["training"]["lr"], 1e-4)
        self.assertEqual(trials[1].resolved_config["training"]["lr"], 5e-4)


class TestSlurmAndSummaryPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.trial_dir = self.root / "trial_001"
        self.trial = TrialSpec(
            trial_id=1,
            trial_name="trial_001_test",
            overrides={"training.lr": 1e-4},
            resolved_config={"training": {"lr": 1e-4}},
            trial_dir=self.trial_dir,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_prepare_and_summary_lifecycle(self):
        cfg_path = prepare_trial_dir(self.trial, {})
        self.assertTrue(cfg_path.exists())
        self.assertTrue((self.trial_dir / "checkpoints").exists())
        self.assertTrue((self.trial_dir / "logs").exists())

        # Write summary
        self.trial.status = "COMPLETED"
        self.trial.elapsed_s = 42.5
        self.trial.best_val_wfe_nm = 34.2
        self.trial.best_epoch = 8
        write_trial_summary(self.trial)

        read_back = read_trial_summary(self.trial_dir)
        self.assertIsNotNone(read_back)
        self.assertEqual(read_back["status"], "COMPLETED")
        self.assertAlmostEqual(read_back["best_val_wfe_nm"], 34.2)
        self.assertEqual(read_back["best_epoch"], 8)

    def test_slurm_script_generation(self):
        cfg_path = prepare_trial_dir(self.trial, {})
        manifest = {
            "name": "slurm_test",
            "slurm": {
                "partition": "gpu_standard",
                "time": "01:30:00",
                "gres": "gpu:1",
                "mem": "16G",
            }
        }
        sb_path = generate_slurm_script(self.trial, cfg_path, manifest, self.root)
        self.assertTrue(sb_path.exists())
        with open(sb_path) as f:
            content = f.read()

        self.assertIn("#SBATCH --partition=gpu_standard", content)
        self.assertIn("#SBATCH --time=01:30:00", content)
        self.assertIn("#SBATCH --gres=gpu:1", content)
        self.assertIn("#SBATCH --mem=16G", content)
        self.assertIn(f"$PYTHON_BIN -m nn_WFS.train --config {cfg_path}", content)

    def test_subprocess_streaming_no_line_buffering_warning(self):
        import subprocess
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proc = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdout.write('line1\\n'); sys.stdout.flush()"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            out, _ = proc.communicate()
            buffering_warnings = [w for w in caught if "line buffering" in str(w.message)]
            self.assertEqual(len(buffering_warnings), 0)
            self.assertEqual(out, "line1\n")


if __name__ == "__main__":
    unittest.main()

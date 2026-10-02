"""
test_hpc_datagen.py — Unit tests for HPC data generation configuration,
staging logic, and Slurm array script generation.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import yaml

from scripts.submit_datagen_array import generate_slurm_array_script


class TestHPCDataGen(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.cfg_file = self.root / "data_generation.yaml"

        self.mock_cfg = {
            "optics": {
                "OD": 0.762,
                "ID": 0.152,
                "focal_ratio": 3.33,
                "delta_z1": 0.369e-3,
                "delta_z2": 0.738e-3,
            },
            "tolerancing": {
                "enabled": True,
                "delta_z1_pct": 0.10,
                "delta_z2_pct": 0.10,
                "asymmetry_um": 40.0,
                "theta_deg": 2.0,
            },
            "storage": {
                "staging_dir": str(self.root / "tmp_staging"),
                "target_root": str(self.root / "rental"),
                "sub_dir": "shards",
                "cleanup_staging": True,
            },
            "hpc": {
                "mode": "array",
                "partition": "standard",
                "account": "cbender",
                "qos": "user_default",
                "time_limit": "01:30:00",
                "cpus_per_task": 4,
                "mem_gb": 16,
                "n_tasks": 10,
                "examples_per_task": 500,
            },
            "simulation": {
                "n_channels": 5,
            },
        }

        with open(self.cfg_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.mock_cfg, f)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_parsing(self):
        with open(self.cfg_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        self.assertEqual(loaded["simulation"]["n_channels"], 5)
        self.assertEqual(loaded["hpc"]["n_tasks"], 10)
        self.assertTrue(loaded["tolerancing"]["enabled"])
        self.assertEqual(loaded["optics"]["delta_z1"], 0.369e-3)
        self.assertEqual(loaded["optics"]["delta_z2"], 0.738e-3)

    def test_generate_slurm_array_script(self):
        out_script = self.root / "submit.sh"
        script_text = generate_slurm_array_script(
            self.cfg_file,
            self.mock_cfg,
            n_tasks=10,
            output_script=out_script,
        )

        self.assertTrue(out_script.exists())
        self.assertIn("#SBATCH --array=0-9", script_text)
        self.assertIn("#SBATCH --account=cbender", script_text)
        self.assertIn("#SBATCH --partition=standard", script_text)
        self.assertIn("#SBATCH --cpus-per-task=4", script_text)
        self.assertIn("#SBATCH --mem=16G", script_text)
        self.assertIn("--n_tasks 10", script_text)
        self.assertIn("--task_id ${SLURM_ARRAY_TASK_ID}", script_text)

    def test_slurm_array_custom_tasks(self):
        out_script = self.root / "submit_custom.sh"
        script_text = generate_slurm_array_script(
            self.cfg_file,
            self.mock_cfg,
            n_tasks=25,
            output_script=out_script,
        )
        self.assertIn("#SBATCH --array=0-24", script_text)
        self.assertIn("--n_tasks 25", script_text)

    def test_staging_directories_resolution(self):
        staging = Path(self.mock_cfg["storage"]["staging_dir"])
        target = Path(self.mock_cfg["storage"]["target_root"]) / self.mock_cfg["storage"]["sub_dir"]
        self.assertNotEqual(staging, target)
        self.assertTrue(str(target).endswith("rental/shards"))


if __name__ == "__main__":
    unittest.main()

"""
test_compare.py — Unit tests for the sweep comparison and metric aggregation engine.
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_GIT = _REPO.parent
for p in [str(_REPO), str(_GIT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from nn_WFS.compare import (
    parse_sparse_recorder_log,
    extract_trial_record,
    render_terminal_table,
    generate_markdown_report,
    export_csv_and_json,
    TrialComparisonRecord,
    EpochRecord,
    DistributionStats,
)


class TestCompareLogParser(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.tmp_dir.name) / "test_log.txt"

        log_content = """
==============================================================================
HPC RUN RECORD: train_rodcnn_test
==============================================================================
Timestamp            | Elapsed   | Event / Details
------------------------------------------------------------------------------
[2026-09-12 10:00:00] [+   100.0s] [TRAIN] epoch s42_e1/10: seed=42, train_loss=0.3500, train_wfe_nm=95.0, train_strehl=0.30, val_loss=0.2000, val_wfe_nm=75.0, val_strehl=0.45, epoch_time_s=100.0, lr=3.00e-04
[2026-09-12 10:01:40] [+   200.0s] [TRAIN] epoch s42_e2/10: seed=42, train_loss=0.1500, train_wfe_nm=55.0, train_strehl=0.60, val_loss=0.0900, val_wfe_nm=48.0, val_strehl=0.68, epoch_time_s=100.0, lr=3.00e-04
[2026-09-12 10:03:20] [+   300.0s] [TRAIN] epoch s42_e3/10: seed=42, train_loss=0.0800, train_wfe_nm=42.0, train_strehl=0.75, val_loss=0.0500, val_wfe_nm=41.5, val_strehl=0.76, epoch_time_s=100.0, lr=3.00e-04
[2026-09-12 10:05:00] [+   400.0s] [TRAIN] epoch s42_e4/10: seed=42, train_loss=0.0500, train_wfe_nm=36.0, train_strehl=0.82, val_loss=0.0450, val_wfe_nm=42.0, val_strehl=0.75, epoch_time_s=100.0, lr=3.00e-04
"""
        with open(self.log_file, "w") as f:
            f.write(log_content)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_parse_epochs_and_overfitting_gap(self):
        epochs, _ = parse_sparse_recorder_log(self.log_file, target_wfe_nm=45.0)
        self.assertEqual(len(epochs), 4)

        # Check Epoch 1: val_wfe (75.0) - train_wfe (95.0) = -20.0 nm
        self.assertAlmostEqual(epochs[0].val_wfe_nm, 75.0)
        self.assertAlmostEqual(epochs[0].train_wfe_nm, 95.0)
        self.assertAlmostEqual(epochs[0].overfitting_gap_nm, -20.0)

        # Check Epoch 4: val_wfe (42.0) - train_wfe (36.0) = +6.0 nm (overfitting gap)
        self.assertAlmostEqual(epochs[3].overfitting_gap_nm, 6.0)

    def test_convergence_velocity_detection(self):
        trial_dir = Path(self.tmp_dir.name)
        logs_dir = trial_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(self.log_file, logs_dir / "recorder.txt")

        # Threshold = 45.0 nm: should reach target at Epoch 3 (41.5 nm)
        record = extract_trial_record(trial_dir, target_wfe_nm=45.0)
        self.assertTrue(record.reached_target)
        self.assertEqual(record.epoch_reached_target, 3)
        self.assertAlmostEqual(record.time_reached_target_s, 300.0)

        # Best validation was Epoch 3 (41.5 nm)
        self.assertEqual(record.best_epoch, 3)
        self.assertAlmostEqual(record.best_val_wfe_nm, 41.5)
        # Gap at best epoch: 41.5 - 42.0 = -0.5 nm
        self.assertAlmostEqual(record.overfitting_gap_best_nm, -0.5)
        # Gap at final epoch (Epoch 4): 42.0 - 36.0 = 6.0 nm
        self.assertAlmostEqual(record.overfitting_gap_final_nm, 6.0)

        # Strict Threshold = 30.0 nm: should NOT reach target
        record_strict = extract_trial_record(trial_dir, target_wfe_nm=30.0)
        self.assertFalse(record_strict.reached_target)
        self.assertIsNone(record_strict.epoch_reached_target)
        self.assertIsNone(record_strict.time_reached_target_s)


class TestDistributionStatsAndReporting(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp_dir.name)

        # Synthesize distribution stats
        self.dist_stats = DistributionStats(
            mean_true_nm=50.0,
            std_true_nm=25.0,
            mean_pred_nm=48.0,
            std_pred_nm=24.0,
            mean_err_nm=-2.0,   # 2 nm bias
            std_err_nm=10.0,    # 10 nm scatter
            total_wfe_rms_nm=10.2,
            strehl=0.98,
        )

        self.records = [
            TrialComparisonRecord(
                trial_id=1,
                trial_name="trial_001_lr_3e-4",
                status="COMPLETED",
                trial_dir=str(self.out_dir / "trial_001"),
                overrides={"training.lr": 3e-4},
                total_elapsed_s=320.0,
                best_val_wfe_nm=38.5,
                best_epoch=7,
                best_checkpoint=str(self.out_dir / "ckpt1.pt"),
                reached_target=True,
                epoch_reached_target=5,
                time_reached_target_s=210.0,
                overfitting_gap_best_nm=1.2,
                overfitting_gap_final_nm=3.5,
                dist_stats=self.dist_stats,
            ),
            TrialComparisonRecord(
                trial_id=2,
                trial_name="trial_002_lr_1e-4",
                status="COMPLETED",
                trial_dir=str(self.out_dir / "trial_002"),
                overrides={"training.lr": 1e-4},
                total_elapsed_s=315.0,
                best_val_wfe_nm=42.1,
                best_epoch=9,
                best_checkpoint=str(self.out_dir / "ckpt2.pt"),
                reached_target=True,
                epoch_reached_target=8,
                time_reached_target_s=280.0,
                overfitting_gap_best_nm=0.5,
                overfitting_gap_final_nm=1.1,
            ),
        ]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_render_terminal_table(self):
        output = render_terminal_table(self.records, target_wfe_nm=45.0)
        self.assertIn("SWEEP COMPARISON SUMMARY", output)
        self.assertIn("trial_001_lr_3e-4", output)
        self.assertIn("trial_002_lr_1e-4", output)
        self.assertIn("38.50", output)

    def test_markdown_report_and_exports(self):
        report_path = generate_markdown_report(self.records, self.out_dir, "test_sweep", target_wfe_nm=45.0)
        self.assertTrue(report_path.exists())

        with open(report_path) as f:
            content = f.read()

        self.assertIn("# Experiment Sweep Report: test_sweep", content)
        self.assertIn("Top Performing Configuration: `trial_001_lr_3e-4`", content)
        self.assertIn("Convergence Velocity", content)
        self.assertIn("Label, Estimation & Error Distribution Analysis", content)
        self.assertIn("-2.00 nm", content)  # Mean error (bias)

        export_csv_and_json(self.records, self.out_dir)
        csv_file = self.out_dir / "comparison_summary.csv"
        json_file = self.out_dir / "comparison_summary.json"
        self.assertTrue(csv_file.exists())
        self.assertTrue(json_file.exists())


if __name__ == "__main__":
    unittest.main()

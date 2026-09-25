import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nn_WFS.utils.sparse_recorder import SparseRecorder, get_hpc_nodename, get_hpc_job_id


class TestSparseRecorder(unittest.TestCase):
    def test_get_hpc_nodename_slurm(self):
        with patch.dict(os.environ, {"SLURMD_NODENAME": "compute-node-42", "HOSTNAME": "local"}):
            self.assertEqual(get_hpc_nodename(), "compute-node-42")

    def test_get_hpc_nodename_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            name = get_hpc_nodename()
            self.assertTrue(len(name) > 0)
            self.assertNotIn(" ", name)

    def test_get_hpc_job_id(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123456"}):
            self.assertEqual(get_hpc_job_id(), "123456")

    def test_sparse_recorder_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            task_name = "test_task"
            with patch.dict(os.environ, {"SLURMD_NODENAME": "test_node_01", "SLURM_JOB_ID": "999"}):
                recorder = SparseRecorder(
                    task_name=task_name,
                    output_dir=tmp_dir,
                    config={"batch_size": 16, "lr": 1e-4},
                    catch_signals=False,
                )

                # Check filename contains node name
                self.assertIn("test_node_01", recorder.filename)
                self.assertIn("test_task", recorder.filename)
                self.assertTrue(recorder.filepath.exists())

                # Log step and message
                recorder.log_message("Testing status message")
                recorder.record_step(
                    step=1,
                    metrics={"loss": 0.05, "wfe_rms_nm": 22.3, "strehl": 0.89},
                    phase="train",
                    step_name="epoch",
                    lr=1e-4,
                )
                recorder.record_checkpoint(
                    epoch=1,
                    metric_name="val_wfe_rms_nm",
                    metric_val=22.3,
                    checkpoint_path="checkpoints/best.pt",
                )
                recorder.record_table(
                    title="Per-Mode Evaluation",
                    headers=["Mode", "RMS (nm)"],
                    rows=[["Z4 (defocus)", "12.4"], ["Z5 (astig)", "8.1"]],
                )
                recorder.close(status="COMPLETED")

                # Verify contents
                content = recorder.filepath.read_text(encoding="utf-8")
                self.assertIn("HPC RUN RECORD: test_task", content)
                self.assertIn("Node Name         : test_node_01", content)
                self.assertIn("SLURM / Job ID    : 999", content)
                self.assertIn("Testing status message", content)
                self.assertIn("wfe_rms_nm=22.3000", content)
                self.assertIn("NEW BEST CHECKPOINT", content)
                self.assertIn("Per-Mode Evaluation", content)
                self.assertIn("Run Status        : COMPLETED", content)
                self.assertFalse((recorder.run_dir / "config.yaml").exists())

    def test_context_manager_exception(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filepath = None
            try:
                with SparseRecorder(task_name="fail_task", output_dir=tmp_dir, catch_signals=False) as rec:
                    filepath = rec.filepath
                    rec.log_message("Starting run before crash")
                    raise RuntimeError("Simulated crash")
            except RuntimeError:
                pass

            self.assertIsNotNone(filepath)
            self.assertTrue(filepath.exists())
            content = filepath.read_text(encoding="utf-8")
            self.assertIn("[ERROR / EXCEPTION] RuntimeError: Simulated crash", content)
            self.assertIn("Run Status        : FAILED (RuntimeError)", content)


if __name__ == "__main__":
    unittest.main()


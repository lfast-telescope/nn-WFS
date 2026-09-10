"""
sparse_recorder.py — Sparse progress and metric recorder for HPC cluster runs.

Writes and immediately flushes timestamped logs, step metrics, evaluation tables,
and checkpoint events to a text file in `nn_WFS/tmp/` whose filename contains the
compute node name (e.g. `node042_train_20260901_120000.txt`).

Guarantees:
1. Every write is immediately flushed and synced to OS storage (`fsync`), ensuring
   no data loss if an HPC node closes, times out, or is preempted.
2. Intercepts POSIX termination signals (SIGTERM, SIGINT, SIGUSR1) sent by cluster
   schedulers (e.g. SLURM) to log a termination banner before exit.
"""

from __future__ import annotations

import math
import os
import platform
import re
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import yaml

_DEFAULT_TMP_DIR = Path(__file__).resolve().parent.parent / "tmp"


def get_hpc_nodename() -> str:
    """
    Resolve the compute node name from environment variables or system info.

    Checks in order:
    1. SLURMD_NODENAME (SLURM compute node name)
    2. SLURM_NODENAME
    3. PBS_NODENAME (PBS/Torque)
    4. LSB_MCPU_HOSTS (LSF)
    5. HOSTNAME (standard environment variable)
    6. socket.gethostname()
    7. platform.node()
    """
    candidates = [
        os.environ.get("SLURMD_NODENAME"),
        os.environ.get("SLURM_NODENAME"),
        os.environ.get("PBS_NODENAME"),
        os.environ.get("LSB_MCPU_HOSTS"),
        os.environ.get("HOSTNAME"),
        socket.gethostname(),
        platform.node(),
    ]
    for name in candidates:
        if name and name.strip():
            # Sanitize for safe filenames (alphanumeric, dots, hyphens, underscores)
            cleaned = re.sub(r"[^\w\.-]", "_", name.strip())
            if cleaned:
                return cleaned
    return "unknown_node"


def get_hpc_job_id() -> Optional[str]:
    """Return the batch scheduler Job ID if running under SLURM / PBS / LSF."""
    return (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("PBS_JOBID")
        or os.environ.get("LSB_JOBID")
    )


class SparseRecorder:
    """
    Sparse progress and metric recorder for long-running HPC workloads.

    Parameters
    ----------
    task_name : str
        Identifier for the task (e.g. 'train_rodcnn', 'eval_synthetic', 'make_data').
    output_dir : str or Path, optional
        Target directory for log files. Defaults to `/home/u7/warrenbfoster/git/nn_WFS/tmp`.
    filename_prefix : str, optional
        Custom prefix. If None, uses `{node_name}_{task_name}`.
    config : dict, optional
        Configuration dict or metadata to include in the header.
    catch_signals : bool, default True
        Whether to register signal handlers for SIGTERM, SIGINT, SIGUSR1 to log
        a preemption/termination warning upon receiving scheduler kill signals.
    """

    def __init__(
        self,
        task_name: str = "test",
        output_dir: Optional[Union[str, Path]] = None,
        filename_prefix: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        catch_signals: bool = True,
    ):
        self.task_name = task_name
        self.node_name = get_hpc_nodename()
        self.job_id = get_hpc_job_id()
        self.start_time = time.time()
        self.start_datetime = datetime.now()

        # Set up output directory (ensures nn_WFS/tmp/ exists)
        self.output_dir = Path(output_dir) if output_dir is not None else _DEFAULT_TMP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Construct run-specific directory inside output_dir (nn_WFS/tmp/<run_dir>/)
        timestamp_str = self.start_datetime.strftime("%Y%m%d_%H%M%S")
        prefix = filename_prefix or f"{self.node_name}_{self.task_name}"
        job_tag = f"_job{self.job_id}" if self.job_id else ""
        self.run_folder_name = f"{prefix}{job_tag}_{timestamp_str}"
        self.run_dir = self.output_dir / self.run_folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.filename = f"{prefix}{job_tag}_{timestamp_str}.txt"
        self.filepath = self.run_dir / self.filename

        self._file = open(self.filepath, "w", encoding="utf-8")
        self._closed = False
        self._zernike_basis_cache: Optional[Dict[str, Any]] = None

        # Write initial metadata header and save config.yaml
        self._write_header(config)

        # Set up signal handlers for graceful HPC preemption notice
        self._prev_handlers = {}
        if catch_signals:
            self._register_signals()

    def _register_signals(self) -> None:
        """Register signal handlers for common HPC termination signals."""
        signals_to_catch = [signal.SIGTERM, signal.SIGINT]
        if hasattr(signal, "SIGUSR1"):
            signals_to_catch.append(signal.SIGUSR1)
        if hasattr(signal, "SIGUSR2"):
            signals_to_catch.append(signal.SIGUSR2)

        for sig in signals_to_catch:
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._signal_handler)
            except (ValueError, AttributeError, RuntimeError):
                # Signals might not be interceptable if not in main thread
                pass

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle termination signal by flushing an alert line."""
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        elapsed = time.time() - self.start_time
        msg = f"[HPC PREEMPTION/ABORT] Received signal {sig_name} ({signum}) after {elapsed:.1f}s. Node is shutting down."
        self.log_message(msg)
        self._flush()

        prev = self._prev_handlers.get(signum)
        if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
            prev(signum, frame)
        else:
            # Exit cleanly with standard signal exit code
            sys.exit(128 + signum)

    def _flush(self) -> None:
        """Flush buffer and synchronize with OS filesystem."""
        if self._file and not self._file.closed:
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except (AttributeError, OSError):
                pass

    def _write_header(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Write structured metadata header to the top of the file."""
        lines = [
            "=" * 78,
            f"HPC RUN RECORD: {self.task_name}",
            "=" * 78,
            f"Node Name         : {self.node_name}",
            f"SLURM / Job ID    : {self.job_id or 'N/A'}",
            f"Process ID (PID)  : {os.getpid()}",
            f"Start Time (Local): {self.start_datetime.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Start Time (UTC)  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Run Directory     : {self.run_dir.resolve()}",
            f"Log File Path     : {self.filepath.resolve()}",
            f"Working Directory : {os.getcwd()}",
            f"Python Executable : {sys.executable}",
            f"Command Line      : {' '.join(sys.argv)}",
        ]

        # Add hardware / PyTorch device info if available
        try:
            import torch
            if torch.cuda.is_available():
                dev_name = torch.cuda.get_device_name(0)
                dev_count = torch.cuda.device_count()
                lines.append(f"Compute Device    : CUDA ({dev_count}x {dev_name})")
            else:
                lines.append("Compute Device    : CPU")
        except ImportError:
            lines.append("Compute Device    : (PyTorch not imported)")

        if config:
            lines.append("-" * 78)
            try:
                config_path = self.run_dir / "config.yaml"
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
                lines.append(f"Config File Saved : {config_path.resolve()}")
            except Exception as e:
                lines.append(f"Config Save Note  : Failed to write config.yaml ({e})")
            lines.append("Configuration Summary:")
            for k, v in config.items():
                lines.append(f"  {k}: {v}")

        lines.extend([
            "=" * 78,
            f"{'Timestamp':<20} | {'Elapsed':<9} | {'Event / Details'}",
            "-" * 78,
        ])
        self._file.write("\n".join(lines) + "\n")
        self._flush()

    def log_message(self, message: str) -> None:
        """
        Log an arbitrary timestamped message with immediate disk synchronization.

        Parameters
        ----------
        message : str
        """
        if self._closed or self._file.closed:
            return
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time
        line = f"[{now_str}] [+ {elapsed:>7.1f}s] {message}"
        self._file.write(line + "\n")
        self._flush()

    def record_step(
        self,
        step: Union[int, str],
        metrics: Dict[str, Any],
        phase: str = "train",
        step_name: str = "step",
        **extra_info: Any,
    ) -> None:
        """
        Record a step / epoch progress row with key metrics.

        Parameters
        ----------
        step : int or str
            Current step or epoch index (e.g. 1, 15, "15/50").
        metrics : dict
            Dict of metric name -> value (e.g. {'loss': 0.042, 'wfe_rms_nm': 18.5, 'strehl': 0.91}).
        phase : str, default 'train'
            Phase identifier ('train', 'val', 'test', 'eval', 'gen').
        step_name : str, default 'step'
            Label for the step ('step', 'epoch', 'example').
        **extra_info
            Additional key-value pairs (e.g. lr=3e-4, eta="00:15:30").
        """
        if self._closed or self._file.closed:
            return
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time

        # Format metrics string
        m_parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                m_parts.append(f"{k}={v:.4f}")
            elif isinstance(v, (int, str)):
                m_parts.append(f"{k}={v}")
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                mean_val = sum(v) / len(v)
                m_parts.append(f"{k}_mean={mean_val:.2f}")

        for k, v in extra_info.items():
            if isinstance(v, float):
                m_parts.append(f"{k}={v:.2e}" if (v < 1e-3 and v > 0) else f"{k}={v:.4f}")
            else:
                m_parts.append(f"{k}={v}")

        metrics_str = ", ".join(m_parts)
        line = f"[{now_str}] [+ {elapsed:>7.1f}s] [{phase.upper()}] {step_name} {step}: {metrics_str}"
        self._file.write(line + "\n")
        self._flush()

    def record_checkpoint(
        self,
        epoch: int,
        metric_name: str,
        metric_val: float,
        checkpoint_path: str,
    ) -> None:
        """
        Record a newly saved model checkpoint.

        Parameters
        ----------
        epoch : int
            The epoch at which the checkpoint was created.
        metric_name : str
            The metric that improved (e.g. 'val_wfe_rms_nm').
        metric_val : float
            The metric value achieved.
        checkpoint_path : str
            Path to the saved `.pt` checkpoint file.
        """
        msg = (
            f"*** NEW BEST CHECKPOINT (Epoch {epoch}) *** | "
            f"{metric_name}={metric_val:.2f} nm | Saved to: {checkpoint_path}"
        )
        self.log_message(msg)

    def record_table(
        self,
        title: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
    ) -> None:
        """
        Write a structured ASCII table (e.g. per-mode evaluation results).

        Parameters
        ----------
        title : str
            Table header / title.
        headers : list of str
            Column names.
        rows : list of rows
            Table data rows.
        """
        if self._closed or self._file.closed:
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        str_rows = []
        for r in rows:
            s_row = [str(c) for c in r]
            str_rows.append(s_row)
            for i, c in enumerate(s_row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(c))

        sep_line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        header_line = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"

        lines = [
            f"\n[{now_str}] [+ {elapsed:>7.1f}s] --- {title} ---",
            sep_line,
            header_line,
            sep_line,
        ]
        for s_row in str_rows:
            row_line = "| " + " | ".join(f"{c:<{w}}" for c, w in zip(s_row, col_widths)) + " |"
            lines.append(row_line)
        lines.append(sep_line)
        lines.append("")

    def save_pupil_reconstruction(
        self,
        epoch: int,
        c_true: Union[np.ndarray, Sequence[float]],
        c_pred: Union[np.ndarray, Sequence[float]],
        trained_modes: Sequence[int],
        pupil_grid_res: int = 256,
        prefix: str = "val",
    ) -> Optional[str]:
        """
        Synthesize and save a 3-panel figure showing the true pupil wavefront,
        predicted pupil wavefront, and residual wavefront error map as a .jpg.

        Standardizes the colormap limits [vmin, vmax] = [-V, +V] across all panels.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        c_true : array-like
            Ground truth Zernike coefficients (length == len(trained_modes)).
        c_pred : array-like
            Predicted Zernike coefficients (length == len(trained_modes)).
        trained_modes : list of int
            Noll mode indices corresponding to the coefficients (e.g. [4, 5, ..., 15]).
        pupil_grid_res : int, default 256
            Resolution of the pupil grid in pixels.
        prefix : str, default 'val'
            Prefix for the saved file name.

        Returns
        -------
        str or None
            Absolute path to the saved .jpg figure, or None if plotting failed.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Initialize / cache HCIPy Zernike basis on circular pupil
            if self._zernike_basis_cache is None or self._zernike_basis_cache.get("res") != pupil_grid_res:
                try:
                    import hcipy
                    grid = hcipy.make_pupil_grid(pupil_grid_res, 1.0)
                    aperture = np.array(hcipy.make_circular_aperture(1.0)(grid).shaped, dtype=bool)
                    max_mode = int(max(trained_modes))
                    basis = hcipy.make_zernike_basis(max_mode, 1.0, grid, starting_mode=1)
                    basis_dict = {
                        int(m): np.array(basis[int(m) - 1].shaped, dtype=np.float64)
                        for m in trained_modes
                    }
                    self._zernike_basis_cache = {
                        "res": pupil_grid_res,
                        "aperture": aperture,
                        "basis": basis_dict,
                    }
                except Exception as e:
                    self.log_message(f"[WARNING] HCIPy basis creation failed: {e}")
                    return None

            aperture = self._zernike_basis_cache["aperture"]
            basis_dict = self._zernike_basis_cache["basis"]

            # Convert coefficients to nanometers if they are in meters (< 1e-2)
            c_t = np.asarray(c_true, dtype=np.float64).flatten()
            c_p = np.asarray(c_pred, dtype=np.float64).flatten()
            scale = 1e9 if np.max(np.abs(c_t)) < 1e-2 else 1.0
            c_t_nm = c_t * scale
            c_p_nm = c_p * scale

            # Synthesize pupil wavefronts
            phi_true = np.zeros((pupil_grid_res, pupil_grid_res), dtype=np.float64)
            phi_pred = np.zeros((pupil_grid_res, pupil_grid_res), dtype=np.float64)

            for idx, mode in enumerate(trained_modes):
                mode_map = basis_dict[int(mode)]
                phi_true += c_t_nm[idx] * mode_map
                phi_pred += c_p_nm[idx] * mode_map

            phi_diff = phi_true - phi_pred

            # Mask outside circular pupil
            phi_true_masked = np.where(aperture, phi_true, np.nan)
            phi_pred_masked = np.where(aperture, phi_pred, np.nan)
            phi_diff_masked = np.where(aperture, phi_diff, np.nan)

            # Compute statistics strictly over the pupil aperture
            ap_true = phi_true[aperture]
            ap_pred = phi_pred[aperture]
            ap_diff = phi_diff[aperture]

            rms_true = float(np.sqrt(np.mean(ap_true ** 2)))
            rms_pred = float(np.sqrt(np.mean(ap_pred ** 2)))
            rms_diff = float(np.sqrt(np.mean(ap_diff ** 2)))

            pv_true = float(np.ptp(ap_true))
            pv_pred = float(np.ptp(ap_pred))
            pv_diff = float(np.ptp(ap_diff))

            # Standardize [vmin, vmax] symmetrically across all 3 maps
            v_max = max(
                float(np.percentile(np.abs(ap_true), 99.5)),
                float(np.percentile(np.abs(ap_pred), 99.5)),
                5.0,
            )
            v_max = float(math.ceil(v_max / 5.0) * 5.0)
            v_min = -v_max

            # Render 3-panel figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
            cmap = "RdBu_r"

            im0 = axes[0].imshow(phi_true_masked, origin="lower", cmap=cmap, vmin=v_min, vmax=v_max)
            axes[0].set_title(f"Ground Truth Pupil\nRMS: {rms_true:.1f} nm | PV: {pv_true:.1f} nm", fontsize=11, fontweight="semibold")
            axes[0].axis("off")
            cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            cbar0.set_label("OPD (nm)", fontsize=9)

            im1 = axes[1].imshow(phi_pred_masked, origin="lower", cmap=cmap, vmin=v_min, vmax=v_max)
            axes[1].set_title(f"Predicted Pupil (Model)\nRMS: {rms_pred:.1f} nm | PV: {pv_pred:.1f} nm", fontsize=11, fontweight="semibold")
            axes[1].axis("off")
            cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            cbar1.set_label("OPD (nm)", fontsize=9)

            im2 = axes[2].imshow(phi_diff_masked, origin="lower", cmap=cmap, vmin=v_min, vmax=v_max)
            axes[2].set_title(f"Residual Error (True - Pred)\nRMS: {rms_diff:.1f} nm | PV: {pv_diff:.1f} nm", fontsize=11, fontweight="semibold")
            axes[2].axis("off")
            cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            cbar2.set_label("OPD (nm)", fontsize=9)

            modes_str = f"Z{trained_modes[0]}–Z{trained_modes[-1]} ({len(trained_modes)} modes)"
            fig.suptitle(
                f"Epoch {epoch:03d} Validation Pupil Reconstruction  [{modes_str}]  "
                f"(Standardized Scale: [{v_min:.0f}, +{v_max:.0f}] nm)",
                fontsize=13,
                fontweight="bold",
                y=1.02,
            )

            fig_filename = f"epoch{epoch:03d}_{prefix}_pupil_reconstruction.jpg"
            fig_path = self.run_dir / fig_filename
            fig.savefig(fig_path, format="jpg", dpi=150, bbox_inches="tight")
            plt.close(fig)

            self.log_message(
                f"[PUPIL MAP] Epoch {epoch:03d}: Saved {fig_filename} "
                f"(True RMS={rms_true:.1f}nm, Pred RMS={rms_pred:.1f}nm, Residual RMS={rms_diff:.1f}nm, scale=[{v_min:.0f}, +{v_max:.0f}] nm)"
            )
            return str(fig_path)

        except Exception as e:
            self.log_message(f"[WARNING] save_pupil_reconstruction failed for epoch {epoch}: {e}")
            return None

    def close(self, status: str = "COMPLETED") -> None:
        """
        Write completion summary and close file handle.

        Parameters
        ----------
        status : str, default 'COMPLETED'
            Final status flag ('COMPLETED', 'INTERRUPTED', 'FAILED', etc.).
        """
        if self._closed or self._file.closed:
            return
        total_time = time.time() - self.start_time
        lines = [
            "-" * 78,
            f"Run Status        : {status}",
            f"Total Duration    : {total_time:.1f} s ({total_time / 60:.2f} min)",
            f"End Time (Local)  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 78,
        ]
        self._file.write("\n".join(lines) + "\n")
        self._flush()
        self._file.close()
        self._closed = True

    def __enter__(self) -> SparseRecorder:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self.log_message(f"[ERROR / EXCEPTION] {exc_type.__name__}: {exc_val}")
            self.close(status=f"FAILED ({exc_type.__name__})")
        else:
            self.close(status="COMPLETED")

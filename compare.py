"""
compare.py — Metric aggregation, statistical analysis, and comparative reporting
engine for nn_WFS experiment sweeps.

Extracts and reports:
1. Convergence Velocity: Epoch and wall-clock time to reach target WFE threshold.
2. Distribution Moments: (mean, std) for True labels, Estimations, and Errors.
3. Overfitting Gap: (val_wfe - train_wfe) at best epoch and final epoch.
4. Per-mode RMS error breakdown and Strehl ratio.
5. Markdown, CSV, JSON, and optional visualization outputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

# Add repo root to sys.path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from evaluate import load_checkpoint, _predict, _metrics_table, NOLL_MODE_NAMES, make_test_loader
from dataset import CWFSDataset, train_val_test_split, get_n_modes


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_wfe_nm: float
    train_strehl: float
    val_loss: float
    val_wfe_nm: float
    val_strehl: float
    epoch_time_s: float
    cumulative_time_s: float
    lr: float
    overfitting_gap_nm: float  # val_wfe_nm - train_wfe_nm


@dataclass
class DistributionStats:
    mean_true_nm: float = 0.0
    std_true_nm: float = 0.0
    mean_pred_nm: float = 0.0
    std_pred_nm: float = 0.0
    mean_err_nm: float = 0.0   # Systematic bias: mean(pred - true)
    std_err_nm: float = 0.0    # Random scatter / jitter: std(pred - true)
    total_wfe_rms_nm: float = 0.0
    strehl: float = 0.0
    per_mode_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class TrialComparisonRecord:
    trial_id: int
    trial_name: str
    status: str
    trial_dir: str
    overrides: dict
    total_elapsed_s: float = 0.0
    best_val_wfe_nm: float = float('inf')
    best_epoch: int = -1
    best_checkpoint: Optional[str] = None
    
    # Multi-Seed Statistics
    val_wfe_mean_nm: float = float('inf')
    val_wfe_std_nm: float = 0.0
    num_seeds: int = 1
    ensemble_test_wfe_nm: Optional[float] = None
    ensemble_mode: Optional[str] = None

    # Convergence Velocity
    target_threshold_nm: float = 45.0
    reached_target: bool = False
    epoch_reached_target: Optional[int] = None
    time_reached_target_s: Optional[float] = None
    
    # Overfitting Gap
    overfitting_gap_best_nm: float = 0.0
    overfitting_gap_final_nm: float = 0.0

    # Epoch History
    epochs: List[EpochRecord] = field(default_factory=list)

    # Deep Distribution Statistics (if evaluated)
    dist_stats: Optional[DistributionStats] = None

    def __post_init__(self):
        if self.best_val_wfe_nm is None:
            self.best_val_wfe_nm = float('inf')
        if self.val_wfe_mean_nm is None:
            self.val_wfe_mean_nm = float('inf')
        if self.val_wfe_mean_nm == float('inf') and self.best_val_wfe_nm != float('inf'):
            self.val_wfe_mean_nm = self.best_val_wfe_nm
        if self.best_val_wfe_nm == float('inf') and self.val_wfe_mean_nm != float('inf'):
            self.best_val_wfe_nm = self.val_wfe_mean_nm


# ─────────────────────────────────────────────────────────────────────────────
# Parsing & Metric Extraction
# ─────────────────────────────────────────────────────────────────────────────

def parse_sparse_recorder_log(log_path: Path, target_wfe_nm: float = 45.0) -> Tuple[List[EpochRecord], dict]:
    """
    Parse a SparseRecorder text log to extract epoch-by-epoch metrics.
    """
    epochs: List[EpochRecord] = []
    cum_time = 0.0
    meta = {}

    if not log_path.exists():
        return epochs, meta

    # Regex for [TRAIN] epoch lines:
    # [TRAIN] epoch s42_e1/50: seed=42, train_loss=0.3346, train_wfe_nm=107.6855, train_strehl=0.2202, val_loss=0.1715, val_wfe_nm=81.6230, val_strehl=0.4192, epoch_time_s=779.1035, lr=3.00e-04
    pattern = re.compile(
        r"\[TRAIN\]\s+epoch\s+(?:s\d+_)?e(\d+)/\d+:\s*"
        r".*?train_loss=([\d\.\-e]+),\s*"
        r"train_wfe_nm=([\d\.\-e]+),\s*"
        r"train_strehl=([\d\.\-e]+),\s*"
        r"val_loss=([\d\.\-e]+),\s*"
        r"val_wfe_nm=([\d\.\-e]+),\s*"
        r"val_strehl=([\d\.\-e]+),\s*"
        r"epoch_time_s=([\d\.\-e]+),\s*"
        r"lr=([\d\.\-e]+)"
    )

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ep = int(m.group(1))
                tr_loss = float(m.group(2))
                tr_wfe = float(m.group(3))
                tr_strehl = float(m.group(4))
                va_loss = float(m.group(5))
                va_wfe = float(m.group(6))
                va_strehl = float(m.group(7))
                ep_time = float(m.group(8))
                lr = float(m.group(9))
                cum_time += ep_time

                overfit_gap = va_wfe - tr_wfe

                epochs.append(
                    EpochRecord(
                        epoch=ep,
                        train_loss=tr_loss,
                        train_wfe_nm=tr_wfe,
                        train_strehl=tr_strehl,
                        val_loss=va_loss,
                        val_wfe_nm=va_wfe,
                        val_strehl=va_strehl,
                        epoch_time_s=ep_time,
                        cumulative_time_s=cum_time,
                        lr=lr,
                        overfitting_gap_nm=overfit_gap,
                    )
                )

    return epochs, meta


def extract_trial_record(trial_dir: Path, target_wfe_nm: float = 45.0) -> TrialComparisonRecord:
    """
    Extract comparison metrics from a trial directory.
    """
    summary_file = trial_dir / "trial_summary.yaml"
    summary_data = {}
    if summary_file.exists():
        with open(summary_file) as f:
            summary_data = yaml.safe_load(f) or {}

    trial_id = summary_data.get('trial_id', 0)
    trial_name = summary_data.get('trial_name', trial_dir.name)
    status = summary_data.get('status', 'UNKNOWN')
    elapsed_s = summary_data.get('elapsed_s', 0.0)
    # Failed/crashed trials write these as explicit `null`, so `.get(..., default)`
    # does not apply; coalesce None -> inf explicitly.
    best_val_wfe_nm = summary_data.get('best_val_wfe_nm')
    if best_val_wfe_nm is None:
        best_val_wfe_nm = float('inf')
    best_epoch = summary_data.get('best_epoch', -1)
    best_checkpoint = summary_data.get('best_checkpoint')
    overrides = summary_data.get('overrides', {})

    # If trial_summary was missing trial_id, deduce from name
    if trial_id == 0:
        m = re.search(r"trial_(\d+)", trial_dir.name)
        if m:
            trial_id = int(m.group(1))

    # Parse logs
    logs_dir = trial_dir / "logs"
    epochs: List[EpochRecord] = []
    if logs_dir.exists():
        txt_logs = list(logs_dir.rglob("*.txt"))
        best_parsed_epochs: List[EpochRecord] = []
        for log_f in txt_logs:
            parsed_epochs, _ = parse_sparse_recorder_log(log_f, target_wfe_nm=target_wfe_nm)
            if len(parsed_epochs) > len(best_parsed_epochs):
                best_parsed_epochs = parsed_epochs
        epochs = best_parsed_epochs

    # Calculate convergence velocity
    reached_target = False
    epoch_target = None
    time_target = None
    overfit_best = 0.0
    overfit_final = 0.0

    if epochs:
        if best_val_wfe_nm == float('inf'):
            best_rec = min(epochs, key=lambda e: e.val_wfe_nm)
            best_val_wfe_nm = best_rec.val_wfe_nm
            best_epoch = best_rec.epoch

        # Find first epoch reaching threshold
        for e in epochs:
            if e.val_wfe_nm <= target_wfe_nm:
                reached_target = True
                epoch_target = e.epoch
                time_target = e.cumulative_time_s
                break

        # Overfitting gap at best epoch
        for e in epochs:
            if e.epoch == best_epoch:
                overfit_best = e.overfitting_gap_nm
                break
        # Final epoch gap
        overfit_final = epochs[-1].overfitting_gap_nm

    # If best_checkpoint is missing, locate in checkpoints dir
    ckpt_dir = trial_dir / "checkpoints"
    if not best_checkpoint:
        ckpt_dir = trial_dir / "checkpoints"
        if ckpt_dir.exists():
            pts = list(ckpt_dir.rglob("final_wfe*nm.pt"))
            if pts:
                best_checkpoint = str(sorted(pts, key=lambda p: p.stat().st_mtime)[-1])

    # Check ensemble_summary.yaml for multi-seed statistics and ensemble test metrics
    ens_summary = ckpt_dir / "ensemble_summary.yaml"
    # Same explicit-null issue as best_val_wfe_nm above.
    val_wfe_mean_nm = summary_data.get('val_wfe_mean_nm')
    if val_wfe_mean_nm is None:
        val_wfe_mean_nm = float('inf')
    val_wfe_std_nm = summary_data.get('val_wfe_std_nm', 0.0)
    num_seeds = summary_data.get('num_seeds', 1)
    ensemble_test_wfe_nm = None
    ensemble_mode = None

    if ens_summary.exists():
        try:
            with open(ens_summary) as f:
                edata = yaml.safe_load(f) or {}
            runs = edata.get('runs', [])
            if runs:
                seed_wfes = [r.get('best_val_wfe_nm') for r in runs if r.get('best_val_wfe_nm') is not None]
                if seed_wfes:
                    val_wfe_mean_nm = float(np.mean(seed_wfes))
                    val_wfe_std_nm = float(np.std(seed_wfes))
                    num_seeds = len(seed_wfes)
            if 'final_test_metrics' in edata:
                ensemble_test_wfe_nm = edata['final_test_metrics'].get('wfe_rms_nm')
            ensemble_mode = edata.get('ensemble_mode')
        except Exception:
            pass

    if val_wfe_mean_nm == float('inf') and best_val_wfe_nm != float('inf'):
        val_wfe_mean_nm = best_val_wfe_nm
        val_wfe_std_nm = 0.0
    elif best_val_wfe_nm == float('inf') and val_wfe_mean_nm != float('inf'):
        best_val_wfe_nm = val_wfe_mean_nm

    return TrialComparisonRecord(
        trial_id=trial_id,
        trial_name=trial_name,
        status=status,
        trial_dir=str(trial_dir),
        overrides=overrides,
        total_elapsed_s=elapsed_s,
        best_val_wfe_nm=best_val_wfe_nm if best_val_wfe_nm is not None else float('inf'),
        best_epoch=best_epoch,
        best_checkpoint=best_checkpoint,
        val_wfe_mean_nm=val_wfe_mean_nm,
        val_wfe_std_nm=val_wfe_std_nm,
        num_seeds=num_seeds,
        ensemble_test_wfe_nm=ensemble_test_wfe_nm,
        ensemble_mode=ensemble_mode,
        target_threshold_nm=target_wfe_nm,
        reached_target=reached_target,
        epoch_reached_target=epoch_target,
        time_reached_target_s=time_target,
        overfitting_gap_best_nm=overfit_best,
        overfitting_gap_final_nm=overfit_final,
        epochs=epochs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deep Distribution Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_trial_distribution(
    trial: TrialComparisonRecord,
    data_loader: Any,
    device: torch.device,
    trained_modes: Optional[list] = None,
    use_raw: bool = False,
    tta: bool = False,
) -> DistributionStats:
    """
    Run checkpoint evaluation to compute True, Estimation, and Error distribution moments.
    """
    if not trial.best_checkpoint or not Path(trial.best_checkpoint).exists():
        return DistributionStats()

    model, meta = load_checkpoint(trial.best_checkpoint, device, use_raw=use_raw)
    lm = torch.from_numpy(meta['label_mean'])
    ls = torch.from_numpy(meta['label_std'])
    z_score_labels = meta.get('z_score_labels', True)

    pred, target = _predict(
        model=model,
        loader=data_loader,
        label_mean=lm,
        label_std=ls,
        device=device,
        labels_are_zscored=z_score_labels,
        tta=tta,
        trained_modes=trained_modes,
    )

    # In nanometers
    pred_nm = (pred * 1e9).numpy()      # [N, M]
    target_nm = (target * 1e9).numpy()  # [N, M]
    err_nm = pred_nm - target_nm        # [N, M]

    mean_true = float(np.mean(target_nm))
    std_true = float(np.std(target_nm))
    mean_pred = float(np.mean(pred_nm))
    std_pred = float(np.std(pred_nm))
    mean_err = float(np.mean(err_nm))   # Bias
    std_err = float(np.std(err_nm))     # Random scatter

    total_wfe_rms = float(np.sqrt(np.mean(np.sum(err_nm**2, axis=1))))
    # Strehl proxy from total WFE in meters
    wfe_m = total_wfe_rms * 1e-9
    strehl = float(np.exp(-(2.0 * np.pi * wfe_m / 632.8e-9)**2)) if wfe_m < 500e-9 else 0.0

    per_mode = {}
    if trained_modes is None:
        trained_modes = list(range(4, 4 + pred_nm.shape[1]))

    for i, mode_num in enumerate(trained_modes):
        if i >= pred_nm.shape[1]:
            break
        mode_name = NOLL_MODE_NAMES.get(mode_num, f"Z{mode_num}")
        m_true = float(np.mean(target_nm[:, i]))
        s_true = float(np.std(target_nm[:, i]))
        m_pred = float(np.mean(pred_nm[:, i]))
        s_pred = float(np.std(pred_nm[:, i]))
        m_err = float(np.mean(err_nm[:, i]))
        s_err = float(np.std(err_nm[:, i]))
        rms_err = float(np.sqrt(np.mean(err_nm[:, i]**2)))

        per_mode[mode_name] = {
            'mean_true_nm': m_true,
            'std_true_nm': s_true,
            'mean_pred_nm': m_pred,
            'std_pred_nm': s_pred,
            'mean_err_nm': m_err,
            'std_err_nm': s_err,
            'rms_err_nm': rms_err,
        }

    return DistributionStats(
        mean_true_nm=mean_true,
        std_true_nm=std_true,
        mean_pred_nm=mean_pred,
        std_pred_nm=std_pred,
        mean_err_nm=mean_err,
        std_err_nm=std_err,
        total_wfe_rms_nm=total_wfe_rms,
        strehl=strehl,
        per_mode_stats=per_mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def render_terminal_table(records: List[TrialComparisonRecord], target_wfe_nm: float) -> str:
    """Format comparative summary table for console output."""
    lines = []
    lines.append(f"\n{'='*110}")
    lines.append(f"\n{'='*128}")
    lines.append(f"  SWEEP COMPARISON SUMMARY (Target WFE Threshold: {target_wfe_nm:.1f} nm)")
    lines.append(f"{'='*110}")
    lines.append(f"{'='*128}")

    headers = [
        ("Rank", 4),
        ("Trial Name", 32),
        ("Best Val (nm)", 13),
        ("Trial Name", 30),
        ("Val WFE (Mean±Std)", 20),
        ("Best Seed", 11),
        ("Ens Test", 10),
        ("Best Ep", 7),
        (f"Ep<={target_wfe_nm:g}nm", 10),
        ("Time to Tgt", 11),
        ("Overfit Best", 12),
        ("Overfit Fin", 12),
        ("Status", 10),
    ]

    header_str = "  " + " ".join(f"{h:<{w}s}" for h, w in headers)
    div_str = "  " + " ".join(f"{'-'*w}" for _, w in headers)
    lines.append(header_str)
    lines.append(div_str)

    sorted_records = sorted(records, key=lambda r: r.best_val_wfe_nm)
    sorted_records = sorted(records, key=lambda r: r.val_wfe_mean_nm)

    for rank, r in enumerate(sorted_records, 1):
        best_str = f"{r.best_val_wfe_nm:.2f}" if r.best_val_wfe_nm < float('inf') else "N/A"
        if r.val_wfe_mean_nm < float('inf'):
            if r.num_seeds > 1:
                val_perf_str = f"{r.val_wfe_mean_nm:.2f} ± {r.val_wfe_std_nm:.2f} nm"
            else:
                val_perf_str = f"{r.val_wfe_mean_nm:.2f} nm"
        else:
            val_perf_str = "N/A"

        best_seed_str = f"{r.best_val_wfe_nm:.2f} nm" if r.best_val_wfe_nm < float('inf') else "N/A"
        ens_test_str = f"{r.ensemble_test_wfe_nm:.2f} nm" if r.ensemble_test_wfe_nm is not None else "N/A"
        ep_str = str(r.best_epoch) if r.best_epoch >= 0 else "N/A"
        tgt_ep_str = f"Ep {r.epoch_reached_target}" if r.reached_target else "No"
        tgt_time_str = f"{r.time_reached_target_s:.0f}s" if r.time_reached_target_s is not None else "N/A"
        of_best_str = f"{r.overfitting_gap_best_nm:+.2f} nm"
        of_fin_str = f"{r.overfitting_gap_final_nm:+.2f} nm"

        t_name = r.trial_name
        if len(t_name) > 32:
            t_name = t_name[:29] + "..."
        if len(t_name) > 30:
            t_name = t_name[:27] + "..."

        row = [
            (str(rank), 4),
            (t_name, 32),
            (best_str, 13),
            (t_name, 30),
            (val_perf_str, 20),
            (best_seed_str, 11),
            (ens_test_str, 10),
            (ep_str, 7),
            (tgt_ep_str, 10),
            (tgt_time_str, 11),
            (of_best_str, 12),
            (of_fin_str, 12),
            (r.status, 10),
        ]
        lines.append("  " + " ".join(f"{val:<{w}s}" for val, w in row))

    lines.append(f"{'='*110}\n")
    lines.append(f"{'='*128}\n")
    return "\n".join(lines)


def generate_markdown_report(
    records: List[TrialComparisonRecord],
    output_dir: Path,
    manifest_name: str,
    target_wfe_nm: float,
) -> Path:
    """Generate comprehensive report.md artifact."""
    sorted_records = sorted(records, key=lambda r: r.best_val_wfe_nm)
    sorted_records = sorted(records, key=lambda r: r.val_wfe_mean_nm)
    best_trial = sorted_records[0] if sorted_records else None

    lines = []
    lines.append(f"# Experiment Sweep Report: {manifest_name}")
    lines.append("")
    lines.append(f"- **Generated At**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Total Trials**: {len(records)}")
    lines.append(f"- **Convergence Threshold Target**: {target_wfe_nm:.1f} nm")
    lines.append("")

    if best_trial and best_trial.val_wfe_mean_nm < float('inf'):
        lines.append("> [!TIP]")
        lines.append(f"> **Top Performing Configuration: `{best_trial.trial_name}`**")
        lines.append(f"> - **Best Val WFE**: {best_trial.best_val_wfe_nm:.2f} nm (achieved at Epoch {best_trial.best_epoch})")
        if best_trial.num_seeds > 1:
            lines.append(f"> - **Validation WFE ({best_trial.num_seeds} seeds)**: **{best_trial.val_wfe_mean_nm:.2f} ± {best_trial.val_wfe_std_nm:.2f} nm**")
            lines.append(f"> - **Best Individual Seed**: {best_trial.best_val_wfe_nm:.2f} nm (achieved at Epoch {best_trial.best_epoch})")
        else:
            lines.append(f"> - **Best Val WFE**: {best_trial.val_wfe_mean_nm:.2f} nm (achieved at Epoch {best_trial.best_epoch})")
        if best_trial.ensemble_test_wfe_nm is not None:
            mode_lbl = f" ({best_trial.ensemble_mode} ensemble)" if best_trial.ensemble_mode else ""
            lines.append(f"> - **Ensemble Test WFE**: {best_trial.ensemble_test_wfe_nm:.2f} nm{mode_lbl}")
        if best_trial.reached_target:
            lines.append(f"> - **Convergence Velocity**: Reached $\\le {target_wfe_nm:.1f}$ nm at **Epoch {best_trial.epoch_reached_target}** ({best_trial.time_reached_target_s:.1f}s)")
        lines.append(f"> - **Overfitting Gap**: {best_trial.overfitting_gap_best_nm:+.2f} nm at best epoch ({best_trial.overfitting_gap_final_nm:+.2f} nm at final epoch)")
        lines.append("")

    lines.append("## 1. Leaderboard & Performance Summary")
    lines.append("")
    lines.append(
        f"| Rank | Trial | Best Val WFE (nm) | Best Epoch | Ep $\\le {target_wfe_nm:g}$nm | Time to Target | Overfit Gap (Best) | Overfit Gap (Final) | Status |"
        f"| Rank | Trial | Val WFE (Mean ± Std) | Best Seed (nm) | Ensemble Test (nm) | Best Epoch | Ep $\\le {target_wfe_nm:g}$nm | Time to Target | Overfit Gap | Status |"
    )
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for rank, r in enumerate(sorted_records, 1):
        best_str = f"**{r.best_val_wfe_nm:.2f}**" if rank == 1 else f"{r.best_val_wfe_nm:.2f}"
        if r.val_wfe_mean_nm < float('inf'):
            if r.num_seeds > 1:
                perf_str = f"**{r.val_wfe_mean_nm:.2f} ± {r.val_wfe_std_nm:.2f} nm**" if rank == 1 else f"{r.val_wfe_mean_nm:.2f} ± {r.val_wfe_std_nm:.2f} nm"
            else:
                perf_str = f"**{r.val_wfe_mean_nm:.2f} nm**" if rank == 1 else f"{r.val_wfe_mean_nm:.2f} nm"
        else:
            perf_str = "N/A"

        best_seed_str = f"{r.best_val_wfe_nm:.2f}" if r.best_val_wfe_nm < float('inf') else "N/A"
        ens_str = f"{r.ensemble_test_wfe_nm:.2f} nm" if r.ensemble_test_wfe_nm is not None else "N/A"
        tgt_ep = f"Epoch {r.epoch_reached_target}" if r.reached_target else "Did not reach"
        tgt_time = f"{r.time_reached_target_s:.1f}s" if r.time_reached_target_s is not None else "N/A"
        lines.append(
            f"| {rank} | `{r.trial_name}` | {best_str} | {r.best_epoch} | {tgt_ep} | {tgt_time} | {r.overfitting_gap_best_nm:+.2f} nm | {r.overfitting_gap_final_nm:+.2f} nm | {r.status} |"
            f"| {rank} | `{r.trial_name}` | {perf_str} | {best_seed_str} | {ens_str} | {r.best_epoch} | {tgt_ep} | {tgt_time} | {r.overfitting_gap_best_nm:+.2f} nm | {r.status} |"
        )
    lines.append("")

    lines.append("## 2. Parameter Variation Details")
    lines.append("")
    lines.append("| Trial | Overrides | Output Directory |")
    lines.append("| :--- | :--- | :--- |")
    for r in records:
        diff_str = ", ".join(f"`{k}={v}`" for k, v in r.overrides.items())
        lines.append(f"| `{r.trial_name}` | {diff_str} | `{r.trial_dir}` |")
    lines.append("")

    # If distribution stats exist on any trial, add section
    has_dist = any(r.dist_stats is not None for r in records)
    if has_dist:
        lines.append("## 3. Label, Estimation & Error Distribution Analysis")
        lines.append("")
        lines.append(
            "| Trial | True Label $(\\mu \\pm \\sigma)$ | Estimation $(\\mu \\pm \\sigma)$ | Bias $\\mu(e)$ | Scatter $\\sigma(e)$ | Total WFE RMS | Strehl |"
        )
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in sorted_records:
            if r.dist_stats:
                ds = r.dist_stats
                true_str = f"{ds.mean_true_nm:.1f} ± {ds.std_true_nm:.1f} nm"
                pred_str = f"{ds.mean_pred_nm:.1f} ± {ds.std_pred_nm:.1f} nm"
                bias_str = f"{ds.mean_err_nm:+.2f} nm"
                scat_str = f"{ds.std_err_nm:.2f} nm"
                wfe_str = f"{ds.total_wfe_rms_nm:.2f} nm"
                s_str = f"{ds.strehl:.4f}"
                lines.append(f"| `{r.trial_name}` | {true_str} | {pred_str} | {bias_str} | {scat_str} | {wfe_str} | {s_str} |")
        lines.append("")

    report_path = output_dir / "report.md"
    tmp_report_path = report_path.with_suffix(".tmp")
    with open(tmp_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    tmp_report_path.replace(report_path)

    return report_path


def export_csv_and_json(records: List[TrialComparisonRecord], output_dir: Path) -> None:
    """Export machine-readable comparison artifacts."""
    csv_path = output_dir / "comparison_summary.csv"
    json_path = output_dir / "comparison_summary.json"
    tmp_csv_path = csv_path.with_suffix(".tmp")
    tmp_json_path = json_path.with_suffix(".tmp")

    # Export CSV
    with open(tmp_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id", "trial_name", "status", "best_val_wfe_nm", "best_epoch",
            "trial_id", "trial_name", "status", "val_wfe_mean_nm", "val_wfe_std_nm",
            "num_seeds", "best_val_wfe_nm", "ensemble_test_wfe_nm", "best_epoch",
            "reached_target", "epoch_reached_target", "time_reached_target_s",
            "overfitting_gap_best_nm", "overfitting_gap_final_nm", "total_elapsed_s",
            "best_checkpoint", "overrides",
        ])
        for r in records:
            writer.writerow([
                r.trial_id, r.trial_name, r.status, r.best_val_wfe_nm, r.best_epoch,
                r.trial_id, r.trial_name, r.status, r.val_wfe_mean_nm, r.val_wfe_std_nm,
                r.num_seeds, r.best_val_wfe_nm, r.ensemble_test_wfe_nm, r.best_epoch,
                r.reached_target, r.epoch_reached_target, r.time_reached_target_s,
                r.overfitting_gap_best_nm, r.overfitting_gap_final_nm, r.total_elapsed_s,
                r.best_checkpoint, json.dumps(r.overrides),
            ])
    tmp_csv_path.replace(csv_path)

    # Export JSON
    json_data = []
    for r in records:
        d = asdict(r)
        # Convert Path objects to str if any
        json_data.append(d)

    with open(tmp_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    tmp_json_path.replace(json_path)


def plot_learning_curves(records: List[TrialComparisonRecord], output_dir: Path) -> Optional[Path]:
    """Generate comparative validation WFE vs epoch overlay plot if matplotlib is installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    trials_with_epochs = [r for r in records if r.epochs]
    if not trials_with_epochs:
        return None

    plt.figure(figsize=(10, 6), dpi=150)
    for r in trials_with_epochs:
        # Group validation WFE by epoch across seeds to prevent wrap-around diagonal lines
        epoch_to_vals = defaultdict(list)
        for e in r.epochs:
            epoch_to_vals[e.epoch].append(e.val_wfe_nm)

        sorted_eps = sorted(epoch_to_vals.keys())
        mean_wfes = [float(np.mean(epoch_to_vals[ep])) for ep in sorted_eps]

        line = plt.plot(sorted_eps, mean_wfes, marker='o', markersize=3, label=r.trial_name)[0]

        # If trial ran multiple seeds, display the inter-seed standard deviation as a shaded envelope
        std_wfes = [float(np.std(epoch_to_vals[ep])) for ep in sorted_eps]
        if any(s > 1e-4 for s in std_wfes):
            plt.fill_between(
                sorted_eps,
                [m - s for m, s in zip(mean_wfes, std_wfes)],
                [m + s for m, s in zip(mean_wfes, std_wfes)],
                color=line.get_color(),
                alpha=0.15,
            )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation WFE RMS (nm)", fontsize=12)
    plt.title("Convergence Velocity: Validation WFE vs Epoch", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()

    plot_path = output_dir / "val_wfe_vs_epoch.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# Main Comparison Runner
# ─────────────────────────────────────────────────────────────────────────────

def compare_experiment_suite(
    experiment_dir: str | Path,
    target_wfe_nm: float = 45.0,
    eval_split: Optional[str] = None,
    device: Optional[torch.device] = None,
    plot: bool = True,
    use_raw: bool = False,
    tta: bool = False,
) -> Tuple[List[TrialComparisonRecord], Path]:
    """
    Scan an experiment directory, harvest metrics, evaluate checkpoints (optional),
    and render report.
    """
    exp_dir = Path(experiment_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Find trial subdirectories
    trial_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")])
    if not trial_dirs:
        raise ValueError(f"No trial subdirectories (starting with 'trial_') found in {exp_dir}")

    records = [extract_trial_record(td, target_wfe_nm=target_wfe_nm) for td in trial_dirs]

    # Rank by best_val_wfe_nm ascending
    records.sort(key=lambda r: r.best_val_wfe_nm)

    # Optional deep distribution evaluation
    if eval_split in ("val", "test"):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load first trial's resolved config to get dataset path
        cfg_0 = exp_dir / trial_dirs[0].name / "config_resolved.yaml"
        if cfg_0.exists():
            with open(cfg_0) as f:
                c0 = yaml.safe_load(f)
            hdf5_path = c0.get('data', {}).get('hdf5_path')
            if hdf5_path and Path(hdf5_path).exists():
                print(f"\n[Deep Eval] Evaluating checkpoints on {eval_split} split from {hdf5_path}...")
                n_modes = get_n_modes(hdf5_path)
                trained_modes = c0.get('model', {}).get('trained_modes') or list(range(4, n_modes + 1))
                split_ratios = c0.get('data', {}).get('split_ratios', [0.8, 0.1, 0.1])
                split_seed = c0.get('data', {}).get('split_seed', 42)
                tr_idx, val_idx, test_idx = train_val_test_split(hdf5_path, split_ratios, split_seed)
                eval_idx = val_idx if eval_split == "val" else test_idx

                # Create raw unnormalized loader
                is_rodcnn = c0.get('model', {}).get('type', '').lower() == 'rodcnn'
                eval_ds = CWFSDataset(hdf5_path, eval_idx, label_stats=None, return_stacks=is_rodcnn)
                eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=4, shuffle=False)

                for r in records:
                    if r.best_checkpoint:
                        print(f"  Evaluating {r.trial_name}...")
                        r.dist_stats = evaluate_trial_distribution(
                            trial=r,
                            data_loader=eval_loader,
                            device=device,
                            trained_modes=trained_modes,
                            use_raw=use_raw,
                            tta=tta,
                        )

    # Print terminal table
    print(render_terminal_table(records, target_wfe_nm=target_wfe_nm))

    # Generate Markdown report & exports
    manifest_name = exp_dir.name
    report_path = generate_markdown_report(records, exp_dir, manifest_name, target_wfe_nm)
    export_csv_and_json(records, exp_dir)
    plot_path = plot_learning_curves(records, exp_dir) if plot else None

    print(f"Comparison report saved to: {report_path}")
    print(f"CSV / JSON summaries saved to: {exp_dir}")
    if plot_path:
        print(f"Learning curves plot saved to: {plot_path}")

    return records, report_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare experiment sweeps for nn_WFS")
    parser.add_argument("--experiment_dir", required=True, help="Directory containing trial subdirectories")
    parser.add_argument("--target_wfe", type=float, default=45.0, help="Target threshold in nm for convergence velocity")
    parser.add_argument("--eval", choices=["val", "test"], default=None,
                        help="Run deep evaluation on validation or test split to compute (mean, std) distributions")
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
        help="Generate learning curves plot (default: True)",
    )
    parser.add_argument(
        "--no-plot",
        "--no_plot",
        dest="plot",
        action="store_false",
        help="Disable generating learning curves plot",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Enable D4 Test-Time Augmentation during deep evaluation",
    )
    parser.add_argument(
        "--no_tta",
        "--no-tta",
        dest="tta",
        action="store_false",
        help="Disable Test-Time Augmentation (default)",
    )
    parser.add_argument(
        "--raw_weights",
        action="store_true",
        default=False,
        help="Evaluate using raw model weights even if SWA weights are present in checkpoint",
    )

    args = parser.parse_args()
    compare_experiment_suite(
        experiment_dir=args.experiment_dir,
        target_wfe_nm=args.target_wfe,
        eval_split=args.eval,
        plot=args.plot,
        use_raw=args.raw_weights,
        tta=args.tta,
    )


if __name__ == "__main__":
    main()

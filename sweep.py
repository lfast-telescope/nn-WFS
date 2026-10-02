"""
sweep.py — Experiment sweep orchestration engine for nn_WFS.

Supports parameter grid sweeps and trial matrices with:
1. Sequential in-job execution (--mode sequential)
2. Slurm batch cluster job generation and submission (--mode slurm)
3. Automatic trial resumption and isolated per-trial outputs
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# Add repo root to sys.path
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ─────────────────────────────────────────────────────────────────────────────
# Config Dot-Path Helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_dot_override(cfg: dict, dot_path: str, value: Any) -> None:
    """
    Set a nested value in a dict using a dot-delimited path (e.g. 'training.lr').
    Creates intermediate dictionaries if they do not exist.
    """
    keys = dot_path.lstrip('-').split('.')
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def get_dot_value(cfg: dict, dot_path: str, default: Any = None) -> Any:
    """Retrieve a nested value using a dot-delimited path."""
    keys = dot_path.split('.')
    node = cfg
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            return default
    return node


def slugify(text: str) -> str:
    """Convert text into a safe filename / directory slug."""
    text = str(text)
    text = re.sub(r"[^\w\.-]", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def format_override_slug(overrides: dict) -> str:
    """Create a short, descriptive slug from trial overrides."""
    parts = []
    for k, v in overrides.items():
        short_k = k.split('.')[-1]
        if isinstance(v, float):
            # Compact scientific or float format
            v_str = f"{v:.1e}" if (v != 0.0 and (abs(v) < 0.001 or abs(v) >= 10000)) else f"{v:g}"
        elif isinstance(v, (list, tuple)):
            items = []
            for item in v:
                if isinstance(item, float):
                    items.append(f"{item:.1e}" if (item != 0.0 and (abs(item) < 0.001 or abs(item) >= 10000)) else f"{item:g}")
                else:
                    items.append(str(item))
            v_str = "_".join(items)
        elif v is None:
            v_str = "none"
        else:
            v_str = str(v)
        parts.append(f"{short_k}_{slugify(v_str)}")
    return "_".join(parts) if parts else "default"


# ─────────────────────────────────────────────────────────────────────────────
# Trial Specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrialSpec:
    trial_id: int
    trial_name: str
    overrides: dict
    resolved_config: dict
    trial_dir: Path
    status: str = "PENDING"
    elapsed_s: float = 0.0
    val_wfe_mean_nm: Optional[float] = None
    val_wfe_std_nm: Optional[float] = None
    best_val_wfe_nm: Optional[float] = None
    best_epoch: Optional[int] = None
    best_checkpoint: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Manifest Loading & Grid Expansion
# ─────────────────────────────────────────────────────────────────────────────

def load_sweep_manifest(manifest_path: str | Path) -> Tuple[dict, dict]:
    """
    Load a sweep manifest YAML and resolve its base configuration.

    Returns
    -------
    (manifest_dict, base_config_dict)
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Sweep manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}

    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid sweep manifest: root must be a mapping, got {type(manifest)}")

    name = manifest.get('name')
    if not name:
        raise ValueError(f"Sweep manifest must define a 'name' string: {manifest_path}")

    base_config_ref = manifest.get('base_config')
    if not base_config_ref:
        raise ValueError(f"Sweep manifest must specify 'base_config': {manifest_path}")

    # Resolve base_config: check relative to manifest, then relative to repo root, then absolute
    candidates = [
        manifest_path.parent / base_config_ref,
        _REPO / base_config_ref,
        _HERE / base_config_ref,
        Path(base_config_ref),
    ]
    resolved_base_path = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            resolved_base_path = cand.resolve()
            break

    if resolved_base_path is None:
        raise FileNotFoundError(
            f"Could not locate base_config '{base_config_ref}' referenced in {manifest_path}. "
            f"Checked: {[str(c) for c in candidates]}"
        )

    with open(resolved_base_path) as f:
        base_cfg = yaml.safe_load(f) or {}

    manifest['_manifest_path'] = str(manifest_path)
    manifest['_base_config_path'] = str(resolved_base_path)
    return manifest, base_cfg


def expand_sweep_trials(manifest: dict, base_cfg: dict, repo_root: Optional[Path] = None) -> List[TrialSpec]:
    """
    Expand a sweep manifest into a list of concrete TrialSpec objects.
    """
    if repo_root is None:
        repo_root = _REPO

    name = manifest['name']
    common_overrides = manifest.get('common_overrides') or {}
    output_root_ref = manifest.get('output_root', f"experiments/results/{name}")
    output_root = (repo_root / output_root_ref).resolve() if not Path(output_root_ref).is_absolute() else Path(output_root_ref)

    matrix = manifest.get('matrix')
    explicit_trials = manifest.get('trials')

    trials_spec_list: List[TrialSpec] = []

    if matrix and explicit_trials:
        raise ValueError("Sweep manifest cannot specify both 'matrix' and 'trials'. Choose one.")

    if matrix:
        if not isinstance(matrix, dict):
            raise ValueError(f"'matrix' must be a dict mapping dot-paths to lists of values, got {type(matrix)}")

        param_keys = list(matrix.keys())
        param_values = []
        for k in param_keys:
            vals = matrix[k]
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            param_values.append(vals)

        combinations = list(itertools.product(*param_values))
        for idx, combo in enumerate(combinations, 1):
            trial_overrides = copy.deepcopy(common_overrides)
            param_delta = {}
            for k, val in zip(param_keys, combo):
                trial_overrides[k] = val
                param_delta[k] = val

            slug = format_override_slug(param_delta)
            trial_name = f"trial_{idx:03d}_{slug}"
            trial_dir = output_root / trial_name

            resolved_cfg = copy.deepcopy(base_cfg)
            for k, val in trial_overrides.items():
                apply_dot_override(resolved_cfg, k, val)

            # Route logging & checkpoints inside trial_dir
            apply_dot_override(resolved_cfg, 'logging.checkpoint_dir', str(trial_dir / "checkpoints"))
            apply_dot_override(resolved_cfg, 'logging.sparse_dir', str(trial_dir / "logs"))

            trials_spec_list.append(
                TrialSpec(
                    trial_id=idx,
                    trial_name=trial_name,
                    overrides=trial_overrides,
                    resolved_config=resolved_cfg,
                    trial_dir=trial_dir,
                )
            )

    elif explicit_trials:
        if not isinstance(explicit_trials, list):
            raise ValueError(f"'trials' must be a list of dicts, got {type(explicit_trials)}")

        for idx, t_dict in enumerate(explicit_trials, 1):
            t_name = t_dict.get('name')
            t_overrides = copy.deepcopy(common_overrides)
            specific_overrides = t_dict.get('overrides') or {}
            t_overrides.update(specific_overrides)

            if not t_name:
                slug = format_override_slug(specific_overrides)
                t_name = f"trial_{idx:03d}_{slug}"
            else:
                t_name = f"trial_{idx:03d}_{slugify(t_name)}"

            trial_dir = output_root / t_name

            resolved_cfg = copy.deepcopy(base_cfg)
            for k, val in t_overrides.items():
                apply_dot_override(resolved_cfg, k, val)

            apply_dot_override(resolved_cfg, 'logging.checkpoint_dir', str(trial_dir / "checkpoints"))
            apply_dot_override(resolved_cfg, 'logging.sparse_dir', str(trial_dir / "logs"))

            trials_spec_list.append(
                TrialSpec(
                    trial_id=idx,
                    trial_name=t_name,
                    overrides=t_overrides,
                    resolved_config=resolved_cfg,
                    trial_dir=trial_dir,
                )
            )
    else:
        # Single baseline trial with common overrides only
        trial_name = "trial_001_baseline"
        trial_dir = output_root / trial_name
        resolved_cfg = copy.deepcopy(base_cfg)
        for k, val in common_overrides.items():
            apply_dot_override(resolved_cfg, k, val)
        apply_dot_override(resolved_cfg, 'logging.checkpoint_dir', str(trial_dir / "checkpoints"))
        apply_dot_override(resolved_cfg, 'logging.sparse_dir', str(trial_dir / "logs"))

        trials_spec_list.append(
            TrialSpec(
                trial_id=1,
                trial_name=trial_name,
                overrides=common_overrides,
                resolved_config=resolved_cfg,
                trial_dir=trial_dir,
            )
        )

    return trials_spec_list


# ─────────────────────────────────────────────────────────────────────────────
# Execution Backends
# ─────────────────────────────────────────────────────────────────────────────

def prepare_trial_dir(trial: TrialSpec, manifest: dict) -> Path:
    """Create directory structure and serialize config_resolved.yaml."""
    trial.trial_dir.mkdir(parents=True, exist_ok=True)
    (trial.trial_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (trial.trial_dir / "logs").mkdir(parents=True, exist_ok=True)

    config_path = trial.trial_dir / "config_resolved.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(trial.resolved_config, f, sort_keys=False)

    return config_path


def read_trial_summary(trial_dir: Path) -> Optional[dict]:
    """Read trial_summary.yaml if present."""
    summary_path = trial_dir / "trial_summary.yaml"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                return yaml.safe_load(f)
        except Exception:
            return None
    return None


def write_trial_summary(trial: TrialSpec, extra: Optional[dict] = None) -> None:
    """Save trial_summary.yaml with status and metrics."""
    data = {
        'trial_id': trial.trial_id,
        'trial_name': trial.trial_name,
        'status': trial.status,
        'elapsed_s': trial.elapsed_s,
        'val_wfe_mean_nm': trial.val_wfe_mean_nm,
        'val_wfe_std_nm': trial.val_wfe_std_nm,
        'best_val_wfe_nm': trial.best_val_wfe_nm,
        'best_epoch': trial.best_epoch,
        'best_checkpoint': trial.best_checkpoint,
        'overrides': trial.overrides,
        'trial_dir': str(trial.trial_dir),
    }
    if extra:
        data.update(extra)

    with open(trial.trial_dir / "trial_summary.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def inspect_trial_checkpoint(trial: TrialSpec) -> None:
    """Scan trial checkpoints to find best val WFE and best checkpoint path."""
    ckpt_dir = trial.trial_dir / "checkpoints"
    if not ckpt_dir.exists():
        return

    # Check for ensemble_summary.yaml first
    ens_summary = ckpt_dir / "ensemble_summary.yaml"
    if ens_summary.exists():
        try:
            with open(ens_summary) as f:
                edata = yaml.safe_load(f)
            if 'final_test_metrics' in edata:
                trial.best_val_wfe_nm = edata['final_test_metrics'].get('wfe_rms_nm')
            # Look at runs list
            runs = edata.get('runs', [])
            if runs:
                seed_wfes = [r.get('best_val_wfe_nm') for r in runs if r.get('best_val_wfe_nm') is not None]
                if seed_wfes:
                    trial.val_wfe_mean_nm = float(sum(seed_wfes) / len(seed_wfes))
                    var = sum((x - trial.val_wfe_mean_nm) ** 2 for x in seed_wfes) / len(seed_wfes)
                    trial.val_wfe_std_nm = float(var ** 0.5)

                best_run = min(runs, key=lambda x: x.get('best_val_wfe_nm', float('inf')))
                trial.best_val_wfe_nm = best_run.get('best_val_wfe_nm')
                trial.best_epoch = best_run.get('best_epoch')
                trial.best_checkpoint = best_run.get('checkpoint')
                return
        except Exception:
            pass

    # Scan .pt files
    pts = list(ckpt_dir.rglob("final_wfe*nm.pt"))
    if not pts:
        return

    best_wfe = float('inf')
    best_pt = None
    best_ep = -1

    for pt in pts:
        # Match pattern: final_wfe12.3nm.pt
        m = re.search(r"wfe([\d\.]+)nm\.pt", pt.name)
        if m:
            wfe = float(m.group(1))
            if wfe < best_wfe:
                best_wfe = wfe
                best_pt = pt

    if best_pt:
        try:
            import torch
            ckpt_data = torch.load(best_pt, map_location='cpu', weights_only=False)
            best_ep = int(ckpt_data.get('epoch', -1))
        except Exception:
            best_ep = -1
        trial.best_val_wfe_nm = best_wfe
        trial.best_epoch = best_ep
        trial.best_checkpoint = str(best_pt)


def run_trial_subprocess(trial: TrialSpec, config_path: Path, python_exe: str = sys.executable) -> bool:
    """
    Execute a single trial using an isolated Python subprocess.
    """
    stdout_log_path = trial.trial_dir / "logs" / "stdout.log"
    stderr_log_path = trial.trial_dir / "logs" / "stderr.log"

    cmd = [
        python_exe,
        "-m", "nn_WFS.train",
        "--config", str(config_path),
    ]

    t0 = time.time()
    trial.status = "RUNNING"
    print(f"\n{'='*70}")
    print(f"  [RUNNING] Trial {trial.trial_id:03d}: {trial.trial_name}")
    print(f"  Config   : {config_path}")
    print(f"  Outputs  : {trial.trial_dir}")
    print(f"{'='*70}\n", flush=True)

    with open(stdout_log_path, "w", encoding="utf-8") as out_f, open(stderr_log_path, "w", encoding="utf-8") as err_f:
        # Stream stdout to both console and file in real-time
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_REPO),
            env=os.environ.copy(),
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        # Read output stream
        assert proc.stdout is not None
        assert proc.stderr is not None

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            out_f.write(line)
            out_f.flush()

        stderr_output = proc.stderr.read()
        if stderr_output:
            sys.stderr.write(stderr_output)
            sys.stderr.flush()
            err_f.write(stderr_output)

        proc.wait()

    trial.elapsed_s = time.time() - t0

    if proc.returncode == 0:
        trial.status = "COMPLETED"
        inspect_trial_checkpoint(trial)
        wfe_str = f"{trial.best_val_wfe_nm:.2f} nm" if trial.best_val_wfe_nm is not None else "N/A"
        print(f"\n[SUCCESS] Trial {trial.trial_id:03d} finished in {trial.elapsed_s:.1f}s. Best Val WFE: {wfe_str}")
        write_trial_summary(trial)
        return True
    else:
        trial.status = f"FAILED (exit {proc.returncode})"
        print(f"\n[ERROR] Trial {trial.trial_id:03d} failed with exit code {proc.returncode} in {trial.elapsed_s:.1f}s")
        write_trial_summary(trial, extra={'returncode': proc.returncode})
        return False


def parse_slurm_duration_to_seconds(time_str: str) -> Optional[float]:
    """Parse Slurm duration strings ('D-HH:MM:SS', 'HH:MM:SS', 'MM:SS') to total seconds."""
    if not time_str:
        return None
    time_str = time_str.strip()
    if not time_str or time_str == "INVALID":
        return None
    try:
        days = 0
        if '-' in time_str:
            d_part, time_str = time_str.split('-', 1)
            days = int(d_part)
        parts = [int(p) for p in time_str.split(':')]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        elif len(parts) == 1:
            h, m, s = 0, 0, parts[0]
        else:
            return None
        return float(days * 86400 + h * 3600 + m * 60 + s)
    except Exception:
        return None


def estimate_trial_duration(trial_cfg: dict, manifest: dict, repo_root: Path = _REPO) -> Tuple[float, str]:
    """
    Estimate expected trial execution time in seconds and format into a Slurm walltime string.
    Dynamically scales with:
      1. Training epochs (training.epochs)
      2. Ensemble seed count (ensemble.seeds)
      3. Dataset size / sample count expansion (from data.hdf5_path)
      4. Mini-batch size and k_pairs_train subsampling
    
    Returns
    -------
    (estimated_duration_seconds, slurm_walltime_string)
    """
    tc = trial_cfg.get('training', {})
    dc = trial_cfg.get('data', {})
    mc = trial_cfg.get('model', {})
    ec = trial_cfg.get('ensemble', {})
    slurm_cfg = manifest.get('slurm') or {}

    epochs = int(tc.get('epochs', 30))
    
    seeds_val = ec.get('seeds', 2)
    if isinstance(seeds_val, list):
        n_seeds = len(seeds_val)
    elif isinstance(seeds_val, int):
        n_seeds = max(1, seeds_val)
    else:
        n_seeds = 1

    # Dataset size scaling relative to chunk 1 baseline (43,300 examples ~ 20.3 GB)
    # Baseline for rodcnn: 1 epoch of 4330 batches takes ~135s train + ~35s val = ~170s.
    dataset_scale = 1.0
    if 'dataset_scale' in slurm_cfg:
        dataset_scale = float(slurm_cfg['dataset_scale'])
    else:
        hdf5_rel = dc.get('hdf5_path', 'data/cwfs_consolidated_chunk1.h5')
        hdf5_path = (repo_root / hdf5_rel) if not Path(hdf5_rel).is_absolute() else Path(hdf5_rel)
        if hdf5_path.exists():
            try:
                import h5py
                with h5py.File(hdf5_path, 'r') as hf:
                    if 'labels' in hf:
                        n_examples = hf['labels'].shape[0]
                        dataset_scale = max(0.1, n_examples / 43300.0)
            except Exception:
                try:
                    size_gb = hdf5_path.stat().st_size / (1024 ** 3)
                    dataset_scale = max(0.1, size_gb / 20.3)
                except Exception:
                    dataset_scale = 1.0

    # Scale with training subsampling (train_sample_count or train_sample_ratio)
    train_count = dc.get('train_sample_count')
    train_ratio = dc.get('train_sample_ratio')
    if train_count is not None:
        try:
            train_scale = max(0.02, min(1.0, float(train_count) / 34638.0))
            dataset_scale *= train_scale
        except (ValueError, TypeError):
            pass
    elif train_ratio is not None:
        try:
            train_scale = max(0.02, min(1.0, float(train_ratio)))
            dataset_scale *= train_scale
        except (ValueError, TypeError):
            pass

    # Scale with amplitude range filtering (typically 20% to 60% of dataset)
    if dc.get('amplitude_range_nm') is not None or dc.get('amplitude_min_nm') is not None or dc.get('amplitude_max_nm') is not None:
        dataset_scale *= 0.5

    batch_size = max(1, int(dc.get('batch_size', 8)))
    bs_factor = 8.0 / batch_size

    k_pairs = int(mc.get('k_pairs_train', 8))
    k_factor = 0.2 + 0.8 * (k_pairs / 8.0)

    base_ch = float(mc.get('base_ch', 32))
    stage_blocks = float(mc.get('stage_blocks', 2))
    arch_factor = ((base_ch / 32.0) ** 0.5) * ((stage_blocks / 2.0) ** 0.5)

    # Base time per epoch per seed (seconds)
    base_time_per_epoch = float(slurm_cfg.get('time_per_epoch_s', 170.0))
    time_per_epoch = base_time_per_epoch * dataset_scale * bs_factor * k_factor * arch_factor

    # Total raw compute time
    raw_compute_s = epochs * n_seeds * time_per_epoch

    # Safety buffer: 35% margin for I/O, SWA, TTA, plus 1 hour startup/shutdown buffer
    buffer_factor = float(slurm_cfg.get('buffer_factor', 1.35))
    fixed_buffer_s = float(slurm_cfg.get('fixed_buffer_s', 3600.0))
    estimated_total_s = max(7200.0, raw_compute_s * buffer_factor + fixed_buffer_s)

    # Format into Slurm HH:MM:SS or D-HH:MM:SS
    walltime_str = format_seconds_to_slurm_time(estimated_total_s)

    return estimated_total_s, walltime_str


def format_seconds_to_slurm_time(total_seconds: float) -> str:
    """Format total seconds into Slurm walltime string ('HH:MM:SS' or 'D-HH:MM:SS')."""
    total_sec = int(math.ceil(max(0.0, total_seconds)))
    days = total_sec // 86400
    rem = total_sec % 86400
    hours = rem // 3600
    minutes = (rem % 3600) // 60
    seconds = rem % 60

    if days > 0:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_active_job_info(fallback_hours: Optional[float] = None) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Detect if currently executing inside an active running Slurm allocation,
    determine remaining walltime, and identify partition name.
    Returns (job_id, remaining_seconds, partition_name).
    If not in an active job or job is not running, returns (None, 0.0, None).
    """
    if fallback_hours is not None and fallback_hours > 0:
        job_id = os.environ.get("SLURM_JOB_ID", "manual_active")
        partition = os.environ.get("SLURM_JOB_PARTITION")
        return job_id, fallback_hours * 3600.0, partition

    job_id = os.environ.get("SLURM_JOB_ID")

    squeue_bin = shutil.which("squeue")
    hostname_bin = shutil.which("hostname")

    # 1. If on login node, we do not treat this as an active compute allocation
    if hostname_bin:
        try:
            node = subprocess.check_output([hostname_bin, "-s"], text=True, timeout=2).strip().lower()
            if any(kw in node for kw in ["login", "puma", "ocelote", "elgato"]):
                return None, 0.0, None
        except Exception:
            pass

    # 2. Query remaining time and partition for known SLURM_JOB_ID
    if job_id and squeue_bin:
        try:
            out = subprocess.check_output(
                [squeue_bin, "-j", str(job_id), "-h", "-o", "%T %L %P"],
                text=True, errors="replace", stderr=subprocess.DEVNULL, timeout=5
            ).strip()
            parts = out.split()
            if len(parts) >= 2 and parts[0] == "RUNNING":
                rem_s = parse_slurm_duration_to_seconds(parts[1])
                part = parts[2] if len(parts) >= 3 else None
                if rem_s is not None and rem_s > 0:
                    return str(job_id), rem_s, part
        except Exception:
            pass

    # 3. Fallback: auto-discover active job for current user on this node
    if squeue_bin and hostname_bin:
        try:
            node = subprocess.check_output([hostname_bin, "-s"], text=True, timeout=2).strip()
            user = os.environ.get("USER", "")
            sq_out = subprocess.check_output(
                [squeue_bin, "-u", user, "-w", node, "-h", "-o", "%i %T %L %P"],
                text=True, errors="replace", stderr=subprocess.DEVNULL, timeout=5
            ).strip()
            for line in sq_out.splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "RUNNING":
                    discovered_id = parts[0]
                    rem_s = parse_slurm_duration_to_seconds(parts[2])
                    part = parts[3] if len(parts) >= 4 else None
                    if rem_s is not None and rem_s > 0:
                        return str(discovered_id), rem_s, part
        except Exception:
            pass

    # 4. Fallback: scontrol show job
    scontrol_bin = shutil.which("scontrol")
    if job_id and scontrol_bin:
        try:
            out = subprocess.check_output(
                [scontrol_bin, "show", "job", str(job_id)],
                text=True, errors="replace", stderr=subprocess.DEVNULL, timeout=5
            )
            if "JobState=RUNNING" in out:
                m_time = re.search(r"TimeLimit=([0-9\-:]+)", out)
                m_run = re.search(r"RunTime=([0-9\-:]+)", out)
                m_part = re.search(r"Partition=(\S+)", out)
                part = m_part.group(1) if m_part else None
                if m_time and m_run:
                    limit_s = parse_slurm_duration_to_seconds(m_time.group(1))
                    run_s = parse_slurm_duration_to_seconds(m_run.group(1))
                    if limit_s is not None and run_s is not None and limit_s > run_s:
                        return str(job_id), (limit_s - run_s), part
        except Exception:
            pass

    return None, 0.0, None


def select_active_job_trial(
    trials: List[TrialSpec],
    remaining_s: float,
    manifest: dict,
    safety_buffer_s: float = 1800.0,
    repo_root: Path = _REPO,
) -> Tuple[Optional[TrialSpec], List[TrialSpec]]:
    """
    Select the LONGEST trial that can safely finish before the active job expires.
    
    Returns
    -------
    (selected_local_trial, remaining_slurm_trials)
    """
    if remaining_s <= safety_buffer_s or not trials:
        return None, list(trials)

    usable_s = remaining_s - safety_buffer_s

    trial_durations = []
    for t in trials:
        dur_s, _ = estimate_trial_duration(t.resolved_config, manifest, repo_root=repo_root)
        trial_durations.append((dur_s, t))

    # Eligible trials that fit within usable time
    eligible = [(dur, t) for (dur, t) in trial_durations if dur <= usable_s]
    if not eligible:
        return None, list(trials)

    # Sort descending by duration: longest trial first
    eligible.sort(key=lambda x: (x[0], -x[1].trial_id), reverse=True)
    _, selected_trial = eligible[0]

    remaining_trials = [t for t in trials if t.trial_id != selected_trial.trial_id]
    return selected_trial, remaining_trials


def generate_slurm_script(trial: TrialSpec, config_path: Path, manifest: dict, repo_root: Path) -> Path:
    """Generate a standalone .sbatch job script for a trial."""
    slurm_cfg = manifest.get('slurm') or {}
    job_name = f"swp_{manifest['name']}_{trial.trial_id:03d}"
    partition = slurm_cfg.get('partition', 'gpu_high_priority')
    account = slurm_cfg.get('account', 'cbender')
    nodes = slurm_cfg.get('nodes', 1)
    ntasks = slurm_cfg.get('ntasks', 1)
    cpus_per_task = slurm_cfg.get('cpus_per_task', 4)
    gres = slurm_cfg.get('gres', 'gpu:nvidia_a100_80gb_pcie_3g.40gb')
    mem = slurm_cfg.get('mem')
    mem_per_cpu = slurm_cfg.get('mem_per_cpu')
    if 'qos' in slurm_cfg:
        qos = slurm_cfg['qos']
    else:
        qos = f"user_qos_{account}" if account else None

    # Dynamic walltime calculation if not manually specified
    if 'time' in slurm_cfg:
        walltime = str(slurm_cfg['time'])
    else:
        _, walltime = estimate_trial_duration(trial.resolved_config, manifest, repo_root=repo_root)

    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={trial.trial_dir}/logs/slurm_%j.out",
        f"#SBATCH --error={trial.trial_dir}/logs/slurm_%j.err",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
    ]
    if gres:
        sbatch_lines.append(f"#SBATCH --gres={gres}")
    if mem_per_cpu:
        sbatch_lines.append(f"#SBATCH --mem-per-cpu={mem_per_cpu}")
    elif mem:
        sbatch_lines.append(f"#SBATCH --mem={mem}")
    if account:
        sbatch_lines.append(f"#SBATCH --account={account}")
    if qos:
        sbatch_lines.append(f"#SBATCH --qos={qos}")

    output_root = trial.trial_dir.parent

    sbatch_lines.extend([
        "",
        "set -euo pipefail",
        f"cd {repo_root}",
        "",
        "# Load environment modules on compute node",
        "if [ -f /etc/profile.d/modules.sh ]; then",
        "    source /etc/profile.d/modules.sh",
        "elif [ -f /etc/profile.d/lmod.sh ]; then",
        "    source /etc/profile.d/lmod.sh",
        "fi",
        "module load python/3.14 2>/dev/null || true",
        "",
        "# Auto-detect virtual environment python",
        "if [ -f \".venv/bin/python\" ]; then",
        "    PYTHON_BIN=\".venv/bin/python\"",
        "else",
        "    PYTHON_BIN=\"python3\"",
        "fi",
        "",
        f"echo \"Starting Slurm Sweep Trial {trial.trial_id} ({trial.trial_name}) on $(hostname)...\"",
        f"$PYTHON_BIN -m nn_WFS.train --config {config_path}",
        "",
        "# Update intermediate comparative sweep report across all finished trials",
        "echo \"Updating intermediate sweep report...\"",
        f"$PYTHON_BIN -m nn_WFS.compare --experiment_dir {output_root} || true",
        "",
    ])

    script_path = trial.trial_dir / "submit.sbatch"
    with open(script_path, "w") as f:
        f.write("\n".join(sbatch_lines) + "\n")

    os.chmod(script_path, 0o755)
    return script_path


# ─────────────────────────────────────────────────────────────────────────────
# Packaged Slurm Worker Dispatch (Zero Allocation Downtime)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_GPU_PARTITION_LIMITS: Dict[str, int] = {
    'gpu_high_priority': 4,
    'high_priority': 4,
    'gpu_standard': 4,
    'standard': 4,
}
DEFAULT_MAX_GPU_NODES = 4


def resolve_max_slurm_workers(
    manifest: dict,
    partition: str,
    active_partition: Optional[str] = None,
    has_active_job: bool = False,
    cli_max_workers: Optional[int] = None,
) -> int:
    """
    Determine the maximum number of concurrent Slurm worker jobs allowed.
    Enforces UA HPC limits (default 4 for gpu_high_priority and gpu_standard).
    Deducts 1 if an active job is already using a GPU node in the same partition family.
    """
    if cli_max_workers is not None and cli_max_workers > 0:
        return cli_max_workers

    slurm_cfg = manifest.get('slurm') or {}
    for key in ('max_workers', 'max_concurrent'):
        if key in slurm_cfg and slurm_cfg[key] is not None:
            return int(slurm_cfg[key])

    limit = DEFAULT_GPU_PARTITION_LIMITS.get(partition, DEFAULT_MAX_GPU_NODES)
    if has_active_job and active_partition and partition:
        p_clean = partition.lower()
        ap_clean = active_partition.lower()
        if p_clean == ap_clean:
            limit = max(1, limit - 1)
        elif "high_priority" in p_clean and "high_priority" in ap_clean:
            limit = max(1, limit - 1)
        elif "standard" in p_clean and "standard" in ap_clean and "high_priority" not in p_clean and "high_priority" not in ap_clean:
            limit = max(1, limit - 1)

    return limit


def package_trials_into_workers(
    trials: List[TrialSpec],
    max_workers: int,
    manifest: dict,
    repo_root: Path = _REPO
) -> List[List[TrialSpec]]:
    """
    Distribute trials across at most max_workers worker jobs using
    Longest Processing Time (LPT) bin-packing to balance cumulative walltimes.
    """
    if not trials:
        return []

    num_workers = min(max(1, max_workers), len(trials))
    if num_workers <= 1:
        return [list(trials)]

    # Compute estimated duration for each trial
    trial_durations = []
    for t in trials:
        dur_s, _ = estimate_trial_duration(t.resolved_config, manifest, repo_root=repo_root)
        trial_durations.append((dur_s, t))

    # Sort descending by duration (LPT)
    trial_durations.sort(key=lambda x: x[0], reverse=True)

    # Buckets: list of [cumulative_duration_s, [TrialSpec, ...]]
    buckets: List[List[Any]] = [[0.0, []] for _ in range(num_workers)]

    for dur_s, trial in trial_durations:
        min_bucket = min(buckets, key=lambda b: b[0])
        min_bucket[0] += dur_s
        min_bucket[1].append(trial)

    # Return non-empty buckets
    return [b[1] for b in buckets if b[1]]


def generate_packaged_slurm_script(
    worker_idx: int,
    total_workers: int,
    worker_trials: List[TrialSpec],
    manifest: dict,
    repo_root: Path,
    output_root: Path,
) -> Path:
    """
    Generate a packaged .sbatch job script that executes multiple trials sequentially
    on a single allocated GPU node, eliminating allocation downtime between trials.
    """
    slurm_cfg = manifest.get('slurm') or {}
    job_name = f"swp_{manifest['name']}_w{worker_idx:02d}"
    partition = slurm_cfg.get('partition', 'gpu_high_priority')
    account = slurm_cfg.get('account', 'cbender')
    nodes = slurm_cfg.get('nodes', 1)
    ntasks = slurm_cfg.get('ntasks', 1)
    cpus_per_task = slurm_cfg.get('cpus_per_task', 4)
    gres = slurm_cfg.get('gres', 'gpu:nvidia_a100_80gb_pcie_3g.40gb')
    mem = slurm_cfg.get('mem')
    mem_per_cpu = slurm_cfg.get('mem_per_cpu')
    if 'qos' in slurm_cfg:
        qos = slurm_cfg['qos']
    else:
        qos = f"user_qos_{account}" if account else None

    # Calculate combined walltime: sum of trial compute times + 3600s buffer
    total_compute_s = 0.0
    for t in worker_trials:
        dur_s, _ = estimate_trial_duration(t.resolved_config, manifest, repo_root=repo_root)
        compute_s = max(300.0, dur_s - 3600.0)
        total_compute_s += compute_s

    total_job_s = max(7200.0, total_compute_s + 3600.0)
    walltime = format_seconds_to_slurm_time(total_job_s)

    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script_path = output_root / f"worker_{worker_idx:02d}_of_{total_workers:02d}.sbatch"

    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_root}/logs/slurm_worker_{worker_idx:02d}_%j.out",
        f"#SBATCH --error={output_root}/logs/slurm_worker_{worker_idx:02d}_%j.err",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
    ]
    if gres:
        sbatch_lines.append(f"#SBATCH --gres={gres}")
    if mem_per_cpu:
        sbatch_lines.append(f"#SBATCH --mem-per-cpu={mem_per_cpu}")
    elif mem:
        sbatch_lines.append(f"#SBATCH --mem={mem}")
    if account:
        sbatch_lines.append(f"#SBATCH --account={account}")
    if qos:
        sbatch_lines.append(f"#SBATCH --qos={qos}")

    sbatch_lines.extend([
        "",
        "set -euo pipefail",
        f"cd {repo_root}",
        "",
        "# Load environment modules on compute node",
        "if [ -f /etc/profile.d/modules.sh ]; then",
        "    source /etc/profile.d/modules.sh",
        "elif [ -f /etc/profile.d/lmod.sh ]; then",
        "    source /etc/profile.d/lmod.sh",
        "fi",
        "module load python/3.14 2>/dev/null || true",
        "",
        "# Auto-detect virtual environment python",
        "if [ -f \".venv/bin/python\" ]; then",
        "    PYTHON_BIN=\".venv/bin/python\"",
        "else",
        "    PYTHON_BIN=\"python3\"",
        "fi",
        "",
        "echo \"================================================================================\"",
        f"echo \"Starting Slurm Sweep Worker {worker_idx}/{total_workers} ({len(worker_trials)} trials packaged) on $(hostname)\"",
        "echo \"================================================================================\"",
    ])

    for step_i, trial in enumerate(worker_trials, 1):
        cfg_path = trial.trial_dir / "config_resolved.yaml"
        sbatch_lines.extend([
            "",
            "echo \"\"",
            "echo \"================================================================================\"",
            f"echo \"  [Worker {worker_idx}/{total_workers}] [{step_i}/{len(worker_trials)}] Starting Trial {trial.trial_id} ({trial.trial_name})\"",
            f"echo \"  Config: {cfg_path}\"",
            "echo \"================================================================================\"",
            f"if ! $PYTHON_BIN -m nn_WFS.train --config \"{cfg_path}\"; then",
            f"    echo \"[Worker Error] Trial {trial.trial_id} ({trial.trial_name}) failed with exit code $?.\"",
            "else",
            f"    echo \"[Worker Info] Trial {trial.trial_id} completed successfully.\"",
            "fi",
            "",
            "# Update intermediate comparative sweep report across all finished trials",
            "echo \"Updating intermediate sweep report...\"",
            f"$PYTHON_BIN -m nn_WFS.compare --experiment_dir \"{output_root}\" || true",
        ])

    sbatch_lines.extend([
        "",
        "echo \"\"",
        f"echo \"Worker {worker_idx}/{total_workers} completed all assigned trials.\"",
        "",
    ])

    with open(script_path, "w") as f:
        f.write("\n".join(sbatch_lines) + "\n")

    os.chmod(script_path, 0o755)
    return script_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI Commands
# ─────────────────────────────────────────────────────────────────────────────

def parse_trial_filter(filter_str: Optional[str]) -> Optional[set[int]]:
    """
    Parse a trial filter string into a set of integer trial IDs.
    Supports comma-separated values and hyphenated ranges:
    e.g. '8-12', '1,3,5', '8-12,14,16'
    """
    if not filter_str:
        return None
    ids = set()
    for part in filter_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start_str, end_str = part.split('-', 1)
            ids.update(range(int(start_str.strip()), int(end_str.strip()) + 1))
        else:
            ids.add(int(part))
    return ids


def cmd_plan(manifest_path: str, trials_filter: Optional[str] = None) -> None:
    """Print preview of the sweep trials without running execution."""
    manifest, base_cfg = load_sweep_manifest(manifest_path)
    trials = expand_sweep_trials(manifest, base_cfg)

    if trials_filter:
        valid_ids = parse_trial_filter(trials_filter)
        if valid_ids is not None:
            trials = [t for t in trials if t.trial_id in valid_ids]
            print(f"  [Info] Filtered preview to {len(trials)} trials matching: {trials_filter}")

    print(f"\n{'='*75}")
    print(f"  SWEEP PLAN: {manifest['name']}")
    print(f"  Description: {manifest.get('description', 'N/A')}")
    print(f"  Base Config: {manifest['_base_config_path']}")
    print(f"  Total Trials: {len(trials)}")
    print(f"{'='*75}\n")

    print(f"  {'ID':<5s} {'Trial Name':<38s} {'Overrides'}")
    print(f"  {'-'*4} {'-'*38} {'-'*30}")
    for t in trials:
        # Format overrides compactly
        diff_pairs = [f"{k}={v}" for k, v in t.overrides.items()]
        diff_str = ", ".join(diff_pairs)
        if len(diff_str) > 60:
            diff_str = diff_str[:57] + "..."
        print(f"  {t.trial_id:<5d} {t.trial_name:<38s} {diff_str}")
    print()


def cmd_run(
    manifest_path: str,
    mode: str = "sequential",
    max_trials: Optional[int] = None,
    trials_filter: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
    compare: bool = True,
    submit: bool = False,
    self_include: bool = True,
    active_job_hours: Optional[float] = None,
    max_workers: Optional[int] = None,
) -> None:
    """Execute the experiment sweep."""
    manifest, base_cfg = load_sweep_manifest(manifest_path)
    trials = expand_sweep_trials(manifest, base_cfg)

    if trials_filter:
        valid_ids = parse_trial_filter(trials_filter)
        if valid_ids is not None:
            trials = [t for t in trials if t.trial_id in valid_ids]
            print(f"  [Info] Filtered execution to {len(trials)} trials matching: {trials_filter}")

    if max_trials is not None and max_trials > 0:
        trials = trials[:max_trials]
        print(f"  [Info] Limiting execution to first {len(trials)} trials (--max_trials {max_trials})")

    output_root_ref = manifest.get('output_root', f"experiments/results/{manifest['name']}")
    output_root = trials[0].trial_dir.parent if trials else (_REPO / output_root_ref).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Save manifest snapshot in output root
    shutil.copy2(manifest['_manifest_path'], output_root / "sweep_manifest.yaml")

    print(f"\n{'='*75}")
    print(f"  STARTING SWEEP: {manifest['name']} (Mode: {mode.upper()})")
    print(f"  Trials to process: {len(trials)}")
    print(f"  Output Root: {output_root}")
    print(f"{'='*75}\n")

    if mode == "slurm":
        # Check active job allocation and self-inclusion
        active_job_id, remaining_s, active_partition = (None, 0.0, None)
        local_trial = None
        slurm_trials = list(trials)

        if self_include:
            active_job_id, remaining_s, active_partition = get_active_job_info(fallback_hours=active_job_hours)
            if active_job_id and remaining_s > 0:
                print(f"[Active Job Detected] SLURM_JOB_ID={active_job_id} ({remaining_s/3600.0:.2f}h remaining, partition={active_partition or 'unknown'})")
                local_trial, slurm_trials = select_active_job_trial(
                    trials, remaining_s, manifest, safety_buffer_s=1800.0, repo_root=_REPO
                )
                if local_trial is not None:
                    loc_dur_s, loc_wt = estimate_trial_duration(local_trial.resolved_config, manifest, repo_root=_REPO)
                    print(f"  [Active Job Claim] Claimed longest feasible trial: Trial {local_trial.trial_id:03d} ({local_trial.trial_name})")
                    print(f"  Estimated duration: {loc_dur_s/3600.0:.2f}h (walltime {loc_wt}) <= Remaining: {remaining_s/3600.0:.2f}h")
                    print(f"  Remaining {len(slurm_trials)} trials will be dispatched to Slurm.")
                else:
                    print(f"  [Active Job Info] No trial can safely finish within {remaining_s/3600.0:.2f}h (with buffer).")
                    print(f"  All {len(trials)} trials will be dispatched to Slurm.")

        # Prepare trial directories and individual submit.sbatch scripts
        for t in slurm_trials:
            cfg_path = prepare_trial_dir(t, manifest)
            generate_slurm_script(t, cfg_path, manifest, _REPO)

        slurm_cfg = manifest.get('slurm') or {}
        partition = slurm_cfg.get('partition', 'gpu_high_priority')
        resolved_workers = resolve_max_slurm_workers(
            manifest=manifest,
            partition=partition,
            active_partition=active_partition,
            has_active_job=(active_job_id is not None and local_trial is not None),
            cli_max_workers=max_workers,
        )

        # Package trials into at most resolved_workers worker jobs using LPT bin-packing
        worker_packages = package_trials_into_workers(slurm_trials, resolved_workers, manifest, repo_root=_REPO)
        total_workers = len(worker_packages)

        worker_scripts = []
        for w_idx, w_trials in enumerate(worker_packages, 1):
            w_sb = generate_packaged_slurm_script(
                worker_idx=w_idx,
                total_workers=total_workers,
                worker_trials=w_trials,
                manifest=manifest,
                repo_root=_REPO,
                output_root=output_root,
            )
            worker_scripts.append(w_sb)

        script_name = "submit_remaining.sh" if local_trial is not None else "submit_all.sh"
        master_script = output_root / script_name
        with open(master_script, "w") as f:
            f.write(f"#!/bin/bash\n# Master script to submit {len(worker_scripts)} packaged worker jobs ({len(slurm_trials)} trials total)\n")
            f.write(f"# Enforcing UA HPC node limit (max {resolved_workers} concurrent GPU nodes in {partition})\nset -e\n")
            for sb in worker_scripts:
                f.write(f"sbatch {sb}\n")
        os.chmod(master_script, 0o755)

        print(f"[Slurm Concurrency & Packaging]")
        print(f"  Partition: {partition} (node limit: {DEFAULT_GPU_PARTITION_LIMITS.get(partition, DEFAULT_MAX_GPU_NODES)})")
        same_part = bool(active_partition and (
            partition.lower() == active_partition.lower()
            or ("high_priority" in partition.lower() and "high_priority" in active_partition.lower())
            or ("standard" in partition.lower() and "standard" in active_partition.lower() and "high_priority" not in partition.lower() and "high_priority" not in active_partition.lower())
        ))
        print(f"  Active job in same partition: {'Yes (-1 node deducted)' if same_part and local_trial else 'No'}")
        print(f"  Allowed Slurm Worker Allocations: {resolved_workers}")
        print(f"  Packaged {len(slurm_trials)} trials into {total_workers} worker jobs (zero allocation downtime between trials).")
        for w_idx, w_trials in enumerate(worker_packages, 1):
            t_names = ", ".join(f"Trial {t.trial_id:03d}" for t in w_trials)
            print(f"    Worker {w_idx:02d}: {len(w_trials)} trial(s) [{t_names}] -> {worker_scripts[w_idx-1].name}")
        print(f"Master submit script: {master_script}\n")

        # Submission handling
        if dry_run or not submit:
            if not submit and not dry_run:
                print("\n[Slurm Dispatcher] Worker scripts generated. Submission held per AI Agent safety policy.")
                print(f"To submit Slurm worker jobs, execute:  bash {master_script}")
                print("Or rerun with '--submit' to dispatch automatically.\n")
            else:
                print("[Slurm Dispatcher] Dry run enabled. Worker scripts generated but not submitted.")
        else:
            sbatch_bin = shutil.which("sbatch")
            if not sbatch_bin:
                print("[Warning] 'sbatch' command not found in PATH. Run inside a Slurm login node.")
            else:
                print(f"\nSubmitting {len(worker_scripts)} packaged Slurm worker jobs...")
                for idx, sb in enumerate(worker_scripts, 1):
                    res = subprocess.run(["sbatch", str(sb)], capture_output=True, text=True)
                    if res.returncode == 0:
                        job_id = res.stdout.strip()
                        print(f"  [{idx}/{len(worker_scripts)}] Submitted: {sb.name} -> {job_id}")
                    else:
                        print(f"  [{idx}/{len(worker_scripts)}] Failed to submit {sb.name}: {res.stderr.strip()}")
                print(f"\nAll {len(worker_scripts)} worker jobs submitted to Slurm.")

        # If local_trial was claimed, run it locally now
        if local_trial is not None and not dry_run:
            print(f"\n{'='*75}")
            print(f"  [Active Job Local Execution] Running Trial {local_trial.trial_id:03d}: {local_trial.trial_name}")
            print(f"{'='*75}\n")
            local_cfg = prepare_trial_dir(local_trial, manifest)
            success = run_trial_subprocess(local_trial, local_cfg)
            print(f"\n[Active Job Local Execution] Trial {local_trial.trial_id:03d} {'COMPLETED' if success else 'FAILED'}.")
            if compare:
                try:
                    from nn_WFS.compare import compare_experiment_suite
                    compare_experiment_suite(output_root, target_wfe_nm=float(manifest.get('target_wfe_nm', 45.0)))
                except Exception as e:
                    print(f"[Warning] Intermediate comparison failed: {e}")

    else:  # mode == "sequential"
        completed_count = 0
        failed_count = 0
        skipped_count = 0

        for t in trials:
            cfg_path = prepare_trial_dir(t, manifest)

            # Check if already completed
            summary = read_trial_summary(t.trial_dir)
            if summary and summary.get('status') == "COMPLETED" and not force:
                print(f"  [Skip] Trial {t.trial_id:03d} ({t.trial_name}) already COMPLETED. Use --force to rerun.")
                skipped_count += 1
                continue

            if dry_run:
                print(f"  [Dry-run] Would execute Trial {t.trial_id:03d}: {t.trial_name}")
                continue

            success = run_trial_subprocess(t, cfg_path)
            if success:
                completed_count += 1
            else:
                failed_count += 1

        print(f"\n{'='*75}")
        print(f"  SWEEP RUN SUMMARY: {manifest['name']}")
        print(f"  Completed: {completed_count} | Failed: {failed_count} | Skipped: {skipped_count} | Total: {len(trials)}")
        print(f"{'='*75}\n")

        if not dry_run and compare:
            target_wfe = float(manifest.get('target_wfe_nm', 45.0))
            try:
                from nn_WFS.compare import compare_experiment_suite
                print(f"{'='*75}")
                print(f"  AUTOMATIC POST-SWEEP COMPARISON & ANALYSIS")
                print(f"{'='*75}\n")
                compare_experiment_suite(
                    output_root,
                    target_wfe_nm=target_wfe,
                )
            except Exception as e:
                print(f"[Warning] Automatic post-sweep comparison failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment sweep manager for nn_WFS")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: plan
    p_plan = subparsers.add_parser("plan", help="Preview trials generated by a sweep manifest")
    p_plan.add_argument("manifest", help="Path to sweep manifest YAML")
    p_plan.add_argument("--trials", type=str, default=None,
                        help="Filter execution to specific trial IDs or ranges (e.g. '8-12' or '1,3,5')")

    # Subcommand: run
    p_run = subparsers.add_parser("run", help="Run or dispatch a sweep")
    p_run.add_argument("manifest", help="Path to sweep manifest YAML")
    p_run.add_argument("--mode", choices=["sequential", "slurm"], default="sequential",
                       help="Execution backend: 'sequential' (in active shell/GPU) or 'slurm' (submit sbatch jobs)")
    p_run.add_argument("--max_trials", type=int, default=None, help="Limit execution to first N trials")
    p_run.add_argument("--trials", type=str, default=None,
                       help="Filter execution to specific trial IDs or ranges (e.g. '8-12' or '1,3,5')")
    p_run.add_argument("--dry_run", "--dry-run", action="store_true", help="Print plan or generate Slurm scripts without executing")
    p_run.add_argument("--force", action="store_true", help="Force rerun even if trial marked COMPLETED")
    p_run.add_argument("--compare", dest="compare", action="store_true", default=True,
                       help="Automatically run comparative analysis after sequential sweep completes (default: True)")
    p_run.add_argument("--no-compare", "--no_compare", dest="compare", action="store_false",
                       help="Disable automatic post-sweep comparison")
    p_run.add_argument("--submit", action="store_true", default=False,
                       help="Submit Slurm jobs via sbatch (default: False, generates scripts for review)")
    p_run.add_argument("--self-include", dest="self_include", action="store_true", default=True,
                       help="When in active Slurm job, claim longest feasible trial locally (default: True)")
    p_run.add_argument("--no-self-include", dest="self_include", action="store_false",
                       help="Do not claim any trial locally; dispatch all trials to Slurm")
    p_run.add_argument("--active-job-hours", type=float, default=None,
                       help="Override remaining hours for current active job (default: auto-detected)")
    p_run.add_argument("--max-workers", "--max_workers", "--max-concurrent", dest="max_workers", type=int, default=None,
                       help="Maximum number of parallel Slurm worker jobs/nodes to allocate (default: cluster partition limit, max 4)")

    # Subcommand: resume
    p_resume = subparsers.add_parser("resume", help="Resume incomplete or failed trials in a sweep")
    p_resume.add_argument("manifest", help="Path to sweep manifest YAML")
    p_resume.add_argument("--mode", choices=["sequential", "slurm"], default="sequential",
                          help="Execution backend: 'sequential' or 'slurm'")
    p_resume.add_argument("--trials", type=str, default=None,
                          help="Filter execution to specific trial IDs or ranges (e.g. '8-12' or '1,3,5')")
    p_resume.add_argument("--compare", dest="compare", action="store_true", default=True,
                          help="Automatically run comparative analysis after sequential sweep completes (default: True)")
    p_resume.add_argument("--no-compare", "--no_compare", dest="compare", action="store_false",
                          help="Disable automatic post-sweep comparison")
    p_resume.add_argument("--submit", action="store_true", default=False,
                          help="Submit Slurm jobs via sbatch (default: False, generates scripts for review)")
    p_resume.add_argument("--self-include", dest="self_include", action="store_true", default=True,
                          help="When in active Slurm job, claim longest feasible trial locally (default: True)")
    p_resume.add_argument("--no-self-include", dest="self_include", action="store_false",
                          help="Do not claim any trial locally; dispatch all trials to Slurm")
    p_resume.add_argument("--active-job-hours", type=float, default=None,
                          help="Override remaining hours for current active job (default: auto-detected)")
    p_resume.add_argument("--max-workers", "--max_workers", "--max-concurrent", dest="max_workers", type=int, default=None,
                          help="Maximum number of parallel Slurm worker jobs/nodes to allocate (default: cluster partition limit, max 4)")

    args = parser.parse_args()

    if args.command == "plan":
        cmd_plan(args.manifest, trials_filter=args.trials)
    elif args.command == "run":
        cmd_run(
            args.manifest,
            mode=args.mode,
            max_trials=args.max_trials,
            trials_filter=args.trials,
            dry_run=args.dry_run,
            force=args.force,
            compare=args.compare,
            submit=getattr(args, 'submit', False),
            self_include=getattr(args, 'self_include', True),
            active_job_hours=getattr(args, 'active_job_hours', None),
            max_workers=getattr(args, 'max_workers', None),
        )
    elif args.command == "resume":
        cmd_run(
            args.manifest,
            mode=args.mode,
            max_trials=None,
            trials_filter=args.trials,
            dry_run=False,
            force=False,
            compare=args.compare,
            submit=getattr(args, 'submit', False),
            self_include=getattr(args, 'self_include', True),
            active_job_hours=getattr(args, 'active_job_hours', None),
            max_workers=getattr(args, 'max_workers', None),
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
submit_datagen_array.py — Slurm array job generator and submitter for HPC data generation.

Reads the `hpc:` and `storage:` blocks from `config/data_generation.yaml`, constructs
a Slurm array batch script to run `make_training_data.py` across shards in parallel,
and optionally submits it via `sbatch`.

Usage
-----
    # Generate and inspect sbatch script without submitting:
    python scripts/submit_datagen_array.py --config config/data_generation.yaml --dry-run

    # Submit the array job to Slurm:
    python scripts/submit_datagen_array.py --config config/data_generation.yaml

    # Override number of tasks or partition:
    python scripts/submit_datagen_array.py --n-tasks 20 --partition standard
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import yaml


def generate_slurm_array_script(
    cfg_path: Path,
    cfg: dict,
    n_tasks: int,
    output_script: Path,
) -> str:
    hpc = cfg.get("hpc", {})
    storage = cfg.get("storage", {})

    account = hpc.get("account", "cbender")
    partition = hpc.get("partition", "standard")
    qos = hpc.get("qos", "user_default")
    time_limit = hpc.get("time_limit", "01:30:00")
    cpus = hpc.get("cpus_per_task", 4)
    mem_gb = hpc.get("mem_gb", 16)

    repo_dir = cfg_path.resolve().parent.parent
    log_dir = repo_dir / "tmp" / "datagen_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=cwfs_datagen",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --qos={qos}",
        f"#SBATCH --array=0-{n_tasks - 1}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={log_dir}/task_%A_%a.out",
        f"#SBATCH --error={log_dir}/task_%A_%a.err",
        "",
        "set -e",
        "",
        f"cd {repo_dir}",
        "",
        "# Ensure modules and project virtual environment are active",
        "if command -v module &> /dev/null; then",
        "    module load python/3.14 2>/dev/null || module load python/3.9 2>/dev/null || true",
        "fi",
        "",
        f"echo \"Starting datagen task ${{SLURM_ARRAY_TASK_ID}} / {n_tasks} on $(hostname)...\"",
        "",
        f"python3 make_training_data.py \\",
        f"    --config {cfg_path.resolve()} \\",
        f"    --task_id ${{SLURM_ARRAY_TASK_ID}} \\",
        f"    --n_tasks {n_tasks}",
        "",
        "echo \"Task ${SLURM_ARRAY_TASK_ID} finished successfully.\"",
    ]

    script_content = "\n".join(lines) + "\n"
    with open(output_script, "w", encoding="utf-8") as f:
        f.write(script_content)

    return script_content


def main():
    parser = argparse.ArgumentParser(description="Submit Slurm array for HPC CWFS data generation.")
    parser.add_argument("--config", default="config/data_generation.yaml", help="Path to data_generation.yaml")
    parser.add_argument("--n-tasks", type=int, default=None, help="Override number of array tasks")
    parser.add_argument("--partition", default=None, help="Override Slurm partition")
    parser.add_argument("--account", default=None, help="Override Slurm account")
    parser.add_argument("--output-script", default="tmp/submit_datagen.sh", help="Path to save generated sbatch script")
    parser.add_argument("--dry-run", action="store_true", help="Generate script without submitting")

    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Error: Config not found at {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.partition:
        cfg.setdefault("hpc", {})["partition"] = args.partition
    if args.account:
        cfg.setdefault("hpc", {})["account"] = args.account

    n_tasks = args.n_tasks or cfg.get("hpc", {}).get("n_tasks", 40)
    out_script = Path(args.output_script)
    out_script.parent.mkdir(parents=True, exist_ok=True)

    script_text = generate_slurm_array_script(cfg_path, cfg, n_tasks, out_script)
    print(f"Generated Slurm array script at: {out_script}")
    print(f"Total tasks: {n_tasks}")

    if args.dry_run:
        print("\n--- Dry Run Script Preview ---")
        print(script_text)
        print("Dry run completed. Script was not submitted.")
        return

    try:
        res = subprocess.run(["sbatch", str(out_script)], capture_output=True, text=True, check=True)
        print(f"Submitted to Slurm: {res.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Note: Could not run sbatch directly ({e}). Script written to {out_script}.")


if __name__ == "__main__":
    main()

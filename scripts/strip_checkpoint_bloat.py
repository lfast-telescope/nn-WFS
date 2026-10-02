#!/usr/bin/env python3
"""
strip_checkpoint_bloat.py — One-off cleanup for legacy checkpoint files.

Background
----------
The old CheckpointManager wrote a full training bundle each time a new
validation-WFE record was set, retaining the top-k by metric.  Each bundle
contained:

  - model_state        ← needed (deployed weights)
  - raw_model_state    ← needed (instantaneous weights for --raw_weights)
  - optim_state        ← BLOAT  (~2× model size: exp_avg + exp_avg_sq)
  - scheduler_state    ← BLOAT  (tiny, but redundant)
  - swa_state          ← BLOAT  (duplicate of model_state when SWA applied)
  - val_wfe_rms, label_mean/std, config, seed, … ← kept (small metadata)

This script strips the three bloat keys in-place (atomic rename), keeping
every evaluation-relevant field intact.  It also renames old epoch*_wfe*.pt
files to final_wfe*.pt to match the new naming convention, taking the best
checkpoint (lowest WFE) per seed directory.

Usage
-----
    # Dry-run (no writes):
    python scripts/strip_checkpoint_bloat.py --results_dir experiments/results

    # Apply:
    python scripts/strip_checkpoint_bloat.py --results_dir experiments/results --apply

    # Single sweep:
    python scripts/strip_checkpoint_bloat.py --results_dir experiments/results/cosine_tail_sweep --apply

Safety
------
- All writes are atomic (save to .tmp then os.replace).
- A file is only modified if it actually contains at least one bloat key.
- resume.pt files are always skipped.
- If a seed directory already contains a final_wfe*.pt, epoch files in the
  same directory are deleted (they are superseded), unless --keep_epoch_files
  is passed.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Try importing torch; give a clear message if not in venv ────────────────
try:
    import torch
except ImportError:
    print("ERROR: torch not importable.  Run with the project venv:")
    print("  /home/u7/warrenbfoster/git/.venv/bin/python3.14 scripts/strip_checkpoint_bloat.py ...")
    sys.exit(1)

# Keys to remove from every checkpoint
BLOAT_KEYS = {"optim_state", "scheduler_state", "swa_state"}

# Pattern for old per-epoch checkpoints
import re
_EPOCH_PAT = re.compile(r"^epoch\d+_wfe([\d.]+)nm\.pt$")
_FINAL_PAT = re.compile(r"^final_wfe([\d.]+)nm\.pt$")


def _wfe_from_name(name: str) -> float | None:
    """Extract the WFE float from an epoch*_wfe*.pt or final_wfe*.pt filename."""
    m = _EPOCH_PAT.match(name) or _FINAL_PAT.match(name)
    return float(m.group(1)) if m else None


def _strip_and_resave(path: Path, apply: bool) -> tuple[int, int]:
    """
    Load checkpoint, strip bloat keys, atomically resave in place.

    Returns (original_bytes, new_bytes).  If no bloat keys present, returns
    (size, size) without touching the file.
    """
    original_bytes = path.stat().st_size

    state = torch.load(path, map_location="cpu", weights_only=False)
    present_bloat = BLOAT_KEYS & set(state.keys())

    if not present_bloat:
        return original_bytes, original_bytes  # nothing to do

    if not apply:
        # Estimate new size: assume bloat ~ model_state size × multiplier
        # optim_state ≈ 2× model_state; scheduler_state tiny; swa_state ≈ 1×
        ms_size = sum(v.numel() * v.element_size()
                      for v in state["model_state"].values()
                      if hasattr(v, "numel"))
        bloat_est = 0
        if "optim_state" in present_bloat:
            bloat_est += ms_size * 2      # exp_avg + exp_avg_sq
        if "swa_state" in present_bloat:
            bloat_est += ms_size          # duplicate weight copy
        new_est = max(original_bytes - bloat_est, original_bytes // 5)
        return original_bytes, new_est

    for k in present_bloat:
        del state[k]

    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return original_bytes, path.stat().st_size


def _process_seed_dir(seed_dir: Path, apply: bool, keep_epoch_files: bool) -> dict:
    """
    Process one seed_* directory.

    Strategy:
    1. Strip bloat from every .pt file (epoch* and final*).
    2. If there are epoch*_wfe*.pt files and no final_wfe*.pt yet:
       - Pick the one with the lowest WFE value as the canonical final file.
       - Rename it to final_wfe{X}nm.pt.
       - Delete the other epoch files (they're redundant top-k duplicates).
    3. If a final_wfe*.pt already exists, delete any epoch* files alongside it.
    """
    results = {"files": [], "renamed": None, "deleted": []}

    pts = [p for p in seed_dir.glob("*.pt") if p.name != "resume.pt"]
    if not pts:
        return results

    epoch_pts = [p for p in pts if _EPOCH_PAT.match(p.name)]
    final_pts = [p for p in pts if _FINAL_PAT.match(p.name)]

    # ── Step 1: strip bloat from all .pt files ───────────────────────────────
    for pt in sorted(pts):
        orig, new = _strip_and_resave(pt, apply)
        results["files"].append({"path": pt, "before": orig, "after": new})

    # ── Step 2/3: normalise to a single final_wfe*.pt ────────────────────────
    if final_pts:
        # Already has a final file — delete any lingering epoch files
        if not keep_epoch_files and epoch_pts:
            for ep in epoch_pts:
                if apply:
                    ep.unlink()
                results["deleted"].append(ep)
    elif epoch_pts:
        # No final file yet — promote the best epoch file
        def _wfe(p):
            v = _wfe_from_name(p.name)
            return v if v is not None else float("inf")

        best = min(epoch_pts, key=_wfe)
        best_wfe = _wfe(best)
        final_name = f"final_wfe{best_wfe:.1f}nm.pt"
        final_path = seed_dir / final_name

        if apply:
            best.rename(final_path)
        results["renamed"] = (best, final_path)

        # Delete the other epoch files
        if not keep_epoch_files:
            for ep in epoch_pts:
                if ep == best:
                    continue
                if apply:
                    ep.unlink()
                results["deleted"].append(ep)

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="experiments/results",
                    help="Root directory to scan (default: experiments/results)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually modify files.  Without this flag the script "
                         "runs in dry-run mode and only prints what it would do.")
    ap.add_argument("--keep_epoch_files", action="store_true",
                    help="Do not delete superseded epoch*_wfe*.pt files "
                         "(useful if you want to inspect them first).")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.exists():
        print(f"ERROR: {results_root} does not exist.")
        sys.exit(1)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{'='*60}")
    print(f"  strip_checkpoint_bloat.py  [{mode}]")
    print(f"  Scanning: {results_root.resolve()}")
    print(f"{'='*60}\n")

    # Find all seed directories
    seed_dirs = sorted(results_root.rglob("seed_*"))
    seed_dirs = [d for d in seed_dirs if d.is_dir()]

    total_before = 0
    total_after  = 0
    total_files  = 0
    total_deleted = 0
    total_renamed = 0

    t0 = time.time()
    for sd in seed_dirs:
        res = _process_seed_dir(sd, args.apply, args.keep_epoch_files)

        dir_before = sum(r["before"] for r in res["files"])
        dir_after  = sum(r["after"]  for r in res["files"])
        saving     = dir_before - dir_after

        if not res["files"] and not res["renamed"] and not res["deleted"]:
            continue

        rel = sd.relative_to(results_root)
        print(f"  {rel}/")

        for r in res["files"]:
            status = "stripped" if r["after"] < r["before"] else "unchanged"
            before_mb = r["before"] / 1e6
            after_mb  = r["after"]  / 1e6
            flag = "→" if status == "stripped" else " "
            print(f"    {flag} {r['path'].name:<45}  "
                  f"{before_mb:6.1f} MB → {after_mb:6.1f} MB  ({status})")

        if res["renamed"]:
            src, dst = res["renamed"]
            verb = "renamed" if args.apply else "would rename"
            print(f"    ✓ {verb}: {src.name} → {dst.name}")
            total_renamed += 1

        for dp in res["deleted"]:
            verb = "deleted" if args.apply else "would delete"
            print(f"    ✗ {verb}: {dp.name}")
            total_deleted += 1

        if saving > 0:
            print(f"    Saving: {saving/1e6:.1f} MB  "
                  f"({dir_before/1e6:.1f} → {dir_after/1e6:.1f} MB)\n")
        else:
            print()

        total_before += dir_before
        total_after  += dir_after
        total_files  += len(res["files"])

    elapsed = time.time() - t0
    total_saving = total_before - total_after
    print(f"{'='*60}")
    print(f"  [{mode}] Done in {elapsed:.1f}s")
    print(f"  Files processed : {total_files}")
    print(f"  Files renamed   : {total_renamed}")
    print(f"  Files deleted   : {total_deleted}")
    print(f"  Space before    : {total_before/1e9:.2f} GB")
    print(f"  Space after     : {total_after/1e9:.2f} GB  (est. in dry-run)")
    print(f"  Space reclaimed : {total_saving/1e9:.2f} GB")
    if not args.apply:
        print("\n  Re-run with --apply to make these changes permanent.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""
ensemble.py — Utilities for multi-seed ensemble training and inference.

Supports:
- Deterministic preset seed sequence generation: [42, 276, ...] with subsequent seeds
  generated via np.random.default_rng(276).
- Configuration parsing for the 'ensemble' YAML block (int, list, dict, or None).
- Reproducible multi-library RNG seeding (PyTorch, NumPy, Python random).
- Ensemble prediction aggregation (mode='average' and mode='best').
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def generate_preset_seeds(n: int) -> List[int]:
    """
    Generate a deterministic list of `n` positive integer seeds.

    Rules:
    - First two seeds: [42, 276]
    - Subsequent seeds (n > 2): drawn deterministically without duplicates
      using `np.random.default_rng(276)`.
    """
    if n <= 0:
        return []
    base = [42, 276]
    if n <= len(base):
        return base[:n]

    rng = np.random.default_rng(276)
    seeds = list(base)
    seen = set(seeds)

    while len(seeds) < n:
        cand = int(rng.integers(1, 100_000))
        if cand not in seen:
            seen.add(cand)
            seeds.append(cand)

    return seeds


def resolve_ensemble_config(cfg: dict) -> Tuple[List[int], str, bool]:
    """
    Parse and resolve the `ensemble` setting from the configuration dictionary.

    Returns
    -------
    seeds : list[int]
        List of integer seeds to run. Defaults to [42].
    mode : str
        Inference aggregation mode: 'average' or 'best'.
    is_ensemble : bool
        True if multi-seed ensemble training is enabled (seeds > 1 or explicitly requested).
    """
    raw = cfg.get('ensemble', None)

    # 1. Omitted / None
    if raw is None:
        return [42], "best", False

    # 2. String handling (e.g. from CLI overrides: --ensemble=3 or --ensemble="[42, 276]")
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.lower() in ("none", "null", "false", ""):
            return [42], "best", False
        if raw.isdigit():
            raw = int(raw)
        elif raw.startswith('[') and raw.endswith(']'):
            items = [x.strip() for x in raw[1:-1].split(',') if x.strip()]
            raw = [int(x) for x in items]

    # 3. Single integer count N
    if isinstance(raw, int):
        n = max(1, raw)
        seeds = generate_preset_seeds(n)
        mode = "average" if n > 1 else "best"
        return seeds, mode, (n > 1)

    # 4. Explicit list of seeds
    if isinstance(raw, (list, tuple)):
        seeds = [int(s) for s in raw]
        if not seeds:
            return [42], "best", False
        mode = "average" if len(seeds) > 1 else "best"
        return seeds, mode, (len(seeds) > 1)

    # 5. Dictionary format: {seeds: ..., mode: ...}
    if isinstance(raw, dict):
        seeds_val = raw.get('seeds', None)
        mode_val = raw.get('mode', None)

        if isinstance(seeds_val, str):
            seeds_val = seeds_val.strip()
            if seeds_val.isdigit():
                seeds_val = int(seeds_val)
            elif seeds_val.startswith('[') and seeds_val.endswith(']'):
                items = [x.strip() for x in seeds_val[1:-1].split(',') if x.strip()]
                seeds_val = [int(x) for x in items]

        if seeds_val is None:
            seeds = [42]
            is_ens = False
        elif isinstance(seeds_val, int):
            n = max(1, seeds_val)
            seeds = generate_preset_seeds(n)
            is_ens = (n > 1)
        elif isinstance(seeds_val, (list, tuple)):
            seeds = [int(s) for s in seeds_val]
            if not seeds:
                seeds = [42]
            is_ens = (len(seeds) > 1)
        else:
            raise TypeError(f"Invalid type for ensemble.seeds: {type(seeds_val)}")

        if mode_val is None:
            mode = "average" if is_ens else "best"
        else:
            mode = str(mode_val).strip().lower()
            if mode not in ("average", "best"):
                raise ValueError(f"Unknown ensemble mode '{mode}'. Must be 'average' or 'best'.")

        return seeds, mode, is_ens

    raise TypeError(f"Unsupported 'ensemble' config type: {type(raw)}. Expected int, list, or dict.")


def seed_everything(seed: int) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch (CPU and CUDA) for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def average_predictions(pred_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Average a list of prediction tensors [N, n_outputs] across ensemble models.

    Parameters
    ----------
    pred_list : list of torch.Tensor
        Tensors of shape [N, n_outputs] in physical units (nm or m).

    Returns
    -------
    torch.Tensor
        Ensemble-averaged predictions of shape [N, n_outputs].
    """
    if not pred_list:
        raise ValueError("Cannot average an empty list of predictions.")
    stacked = torch.stack(pred_list, dim=0)   # [M, N, n_outputs]
    return torch.mean(stacked, dim=0)        # [N, n_outputs]


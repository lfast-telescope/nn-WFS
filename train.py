"""
train.py — Unified training script for TransformerCWFS and CNNCWFS.

Usage
-----
    python train.py --config config/transformer.yaml --hdf5_path /data/cwfs_1M.h5
    python train.py --config config/cnn.yaml         --hdf5_path /data/cwfs_1M.h5

All config values can be overridden on the command line using the
--section.key=value syntax (the = is required), e.g.:
    --training.epochs=100  --data.batch_size=64

Checkpoints are saved to config["logging"]["checkpoint_dir"] whenever validation
total WFE RMS improves.  The top-k best checkpoints are kept; older worse ones
are deleted automatically.
"""

# %%
import argparse
import heapq
import inspect
import math
import os
import re
import sys
import signal
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required:  pip install pyyaml")

# ── project imports ──────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from dataset import (
    CWFSDataset, train_val_test_split, subsample_indices, compute_label_stats, get_n_modes
)
from models.transformer_cwfs import TransformerCWFS
from models.cnn_cwfs import SIAMCNN, RODCNN, CNNCWFS
from models.toy_model import SLPCWFS
from contextlib import nullcontext
from utils.augmentation import D4Augment, validate_trained_modes_pairing
from utils.metrics import (
    per_mode_rms,
    total_wfe_rms,
    strehl_proxy,
    format_order_grouped_rms,
)
from utils.sparse_recorder import SparseRecorder
from utils.ensemble import resolve_ensemble_config, seed_everything, average_predictions
from utils.swa import ModelSWA

# ─────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply command-line key=value overrides to a nested config dict.
    Supports dotted paths, e.g. 'training.epochs=100'.
    """
    for item in overrides:
        if '=' not in item:
            continue
        key_path, value_str = item.split('=', 1)
        keys = key_path.lstrip('-').split('.')
        node = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        # attempt boolean / numeric coercion
        if value_str.lower() in ('true', 'yes', '1'):
            value = True
        elif value_str.lower() in ('false', 'no', '0'):
            value = False
        else:
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str
        node[keys[-1]] = value
    return cfg


def _resolve_z_score_labels(cfg: dict) -> bool:
    """
    Resolve whether Z-score label normalization is enabled.
    Checks data and training sections for 'z_score_labels' or 'normalize_labels'.
    Defaults to False if unspecified.
    """
    dc = cfg.get('data', {})
    tc = cfg.get('training', {})
    for container in (dc, tc):
        for key in ('z_score_labels', 'normalize_labels', 'zscore_norm', 'z_score'):
            if key in container:
                val = container[key]
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    return val.strip().lower() in ('true', 'yes', '1', 'on')
                return bool(val)
    return False


def _resolve_log_loss(cfg: dict) -> bool:
    """
    Resolve whether the loss function should be computed and returned as ln(MSE).
    Checks training and data sections for 'log_loss', 'ln_loss', 'log_mse', etc.
    Defaults to False if unspecified.
    """
    tc = cfg.get('training', {})
    dc = cfg.get('data', {})
    for container in (tc, dc):
        for key in ('log_loss', 'ln_loss', 'log_mse', 'use_log_loss', 'use_ln_loss', 'loss_ln'):
            if key in container:
                val = container[key]
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    return val.strip().lower() in ('true', 'yes', '1', 'on')
                return bool(val)
    return False


def _resolve_loss_type(cfg: dict) -> str:
    """
    Resolve loss function type.
    Options:
      - 'wfe_rms' (or 'wfe', 'rms_wfe'): Direct Root-Sum-Square WFE loss in nm
      - 'log_mse' (or 'log_loss', 'ln_mse'): Natural log of MSE
      - 'mse': Mean Squared Error
    """
    tc = cfg.get('training', {})
    dc = cfg.get('data', {})
    for container in (tc, dc):
        for k in ('loss_type', 'criterion'):
            if k in container:
                lt = str(container[k]).lower().strip()
                if lt in ('wfe_rms', 'wfe', 'rms_wfe', 'rms'):
                    return 'wfe_rms'
                if lt in ('log_mse', 'log_loss', 'ln_loss', 'ln_mse', 'logmse'):
                    return 'log_mse'
                if lt in ('mse', 'standard'):
                    return 'mse'
    if _resolve_log_loss(cfg):
        return 'log_mse'
    if not _resolve_z_score_labels(cfg):
        return 'wfe_rms'
    return 'mse'


def _resolve_swa(cfg: dict) -> tuple[bool, int]:
    """
    Resolve Stochastic Weight Averaging (SWA) configuration.
    Returns (enabled: bool, tail_epochs: int).
    Defaults to (False, 5).
    """
    tc = cfg.get('training', {})
    val = tc.get('swa', False)
    if isinstance(val, str):
        enabled = val.strip().lower() in ('true', 'yes', '1', 'on')
    else:
        enabled = bool(val)
    tail_epochs = int(tc.get('swa_tail_epochs', 5))
    return enabled, tail_epochs


def _resolve_tta(cfg: dict) -> bool:
    """
    Resolve Test-Time Augmentation (TTA) configuration.
    Checks 'evaluation.tta', 'training.tta', and top-level 'tta'.
    Defaults to False.
    """
    for section in ('evaluation', 'training', 'eval'):
        sec_dict = cfg.get(section, {})
        if isinstance(sec_dict, dict) and 'tta' in sec_dict:
            val = sec_dict['tta']
            if isinstance(val, str):
                return val.strip().lower() in ('true', 'yes', '1', 'on')
            return bool(val)
    if 'tta' in cfg:
        val = cfg['tta']
        if isinstance(val, str):
            return val.strip().lower() in ('true', 'yes', '1', 'on')
        return bool(val)
    return False


def _resolve_scheduler_config(cfg: dict) -> dict:
    """
    Resolve learning rate scheduler configuration.
    Supports structured 'training.scheduler' dict, string alias 'training.scheduler' or 'training.lr_scheduler',
    and direct parameters (e.g. min_lr, patience, factor, warmup_steps).
    Defaults to type='constant'.
    """
    tc = cfg.get('training', {})
    sched_raw = tc.get('scheduler', tc.get('lr_scheduler', 'constant'))
    tail_epochs = tc.get('tail_epochs', None)
    tail_ratio = float(tc.get('tail_ratio', 0.20))

    if isinstance(sched_raw, dict):
        sched_type = str(sched_raw.get('type', 'constant')).strip().lower()
        min_lr = float(sched_raw.get('min_lr', tc.get('min_lr', 1.0e-6)))
        patience = int(sched_raw.get('patience', tc.get('patience', 2)))
        factor = float(sched_raw.get('factor', tc.get('factor', 0.5)))
        warmup_steps = int(sched_raw.get('warmup_steps', tc.get('warmup_steps', 500)))
        if 'tail_epochs' in sched_raw and sched_raw['tail_epochs'] is not None:
            tail_epochs = int(sched_raw['tail_epochs'])
        if 'tail_ratio' in sched_raw and sched_raw['tail_ratio'] is not None:
            tail_ratio = float(sched_raw['tail_ratio'])
    elif isinstance(sched_raw, str):
        sched_type = sched_raw.strip().lower()
        min_lr = float(tc.get('min_lr', 1.0e-6))
        patience = int(tc.get('patience', 2))
        factor = float(tc.get('factor', 0.5))
        warmup_steps = int(tc.get('warmup_steps', 500))
    else:
        sched_type = 'constant'
        min_lr = float(tc.get('min_lr', 1.0e-6))
        patience = int(tc.get('patience', 2))
        factor = float(tc.get('factor', 0.5))
        warmup_steps = int(tc.get('warmup_steps', 500))

    # Canonicalize aliases for "Cosine with Extended Plateau Tail"
    if sched_type in (
        'cosine_tail',
        'cosine_with_plateau_tail',
        'cosine_plateau_tail',
        'cosine_extended_tail',
        'cosine_plateau',
        'cosine_with_tail',
    ):
        sched_type = 'cosine_tail'

    return {
        'type': sched_type,
        'min_lr': min_lr,
        'patience': patience,
        'factor': factor,
        'warmup_steps': warmup_steps,
        'tail_epochs': tail_epochs,
        'tail_ratio': tail_ratio,
    }


def _resolve_betas(cfg: dict) -> tuple[float, float]:
    """
    Resolve AdamW momentum (beta1) and second-moment EMA decay (beta2) coefficients.
    Supports:
    - 'training.betas': list, tuple, or string (e.g. [0.9, 0.999] or "0.9,0.999")
    - Granular keys 'training.beta1' / 'training.b1' and 'training.beta2' / 'training.b2'
    Defaults to (0.9, 0.999).
    """
    tc = cfg.get('training', {})
    b1, b2 = 0.9, 0.999

    if 'betas' in tc and tc['betas'] is not None:
        raw = tc['betas']
        if isinstance(raw, (list, tuple)):
            if len(raw) != 2:
                raise ValueError(f"'training.betas' must contain exactly 2 elements, got {len(raw)}: {raw}")
            b1, b2 = float(raw[0]), float(raw[1])
        elif isinstance(raw, str):
            cleaned = raw.strip().lstrip('([').rstrip(')]')
            parts = [p.strip() for p in cleaned.replace(',', ' ').split() if p.strip()]
            if len(parts) != 2:
                raise ValueError(f"Could not parse 'training.betas' string '{raw}' into 2 floats")
            b1, b2 = float(parts[0]), float(parts[1])
        else:
            raise TypeError(f"'training.betas' must be a list, tuple, or string, got {type(raw).__name__}")

    for k in ('beta1', 'b1'):
        if k in tc and tc[k] is not None:
            b1 = float(tc[k])
            break

    for k in ('beta2', 'b2'):
        if k in tc and tc[k] is not None:
            b2 = float(tc[k])
            break

    if not (0.0 <= b1 < 1.0):
        raise ValueError(f"Invalid beta1: {b1}. Must satisfy 0.0 <= beta1 < 1.0")
    if not (0.0 <= b2 < 1.0):
        raise ValueError(f"Invalid beta2: {b2}. Must satisfy 0.0 <= beta2 < 1.0")

    return (b1, b2)


def _resolve_dropout(cfg: dict) -> float:
    """
    Resolve model dropout probability in MLP regression head.
    Checks cfg['model']['dropout'], defaulting to 0.1 if unspecified.
    Validates 0.0 <= dropout < 1.0.
    """
    mc = cfg.get('model', {})
    raw = mc.get('dropout', 0.1)
    if raw is None:
        raw = 0.0
    try:
        dropout = float(raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid model.dropout value '{raw}': must be a float.") from e

    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"Invalid model.dropout: {dropout}. Must satisfy 0.0 <= dropout < 1.0")

    return dropout


def _resolve_base_ch(cfg: dict) -> int:
    """
    Resolve model base channels (stem width).
    Checks cfg['model']['base_ch'], defaulting to 32.
    Validates base_ch >= 8 and divisible by 2.
    """
    mc = cfg.get('model', {})
    raw = mc.get('base_ch', 32)
    if raw is None:
        raw = 32
    try:
        base_ch = int(raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid model.base_ch value '{raw}': must be an integer.") from e

    if base_ch < 8 or base_ch % 2 != 0:
        raise ValueError(f"Invalid model.base_ch: {base_ch}. Must be >= 8 and even.")

    return base_ch


def _resolve_stage_blocks(cfg: dict) -> int:
    """
    Resolve number of BasicBlocks per ResNet stage.
    Checks cfg['model']['stage_blocks'], defaulting to 2.
    Validates stage_blocks >= 1.
    """
    mc = cfg.get('model', {})
    raw = mc.get('stage_blocks', 2)
    if raw is None:
        raw = 2
    try:
        stage_blocks = int(raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid model.stage_blocks value '{raw}': must be an integer.") from e

    if stage_blocks < 1:
        raise ValueError(f"Invalid model.stage_blocks: {stage_blocks}. Must be >= 1.")

    return stage_blocks


def _resolve_train_subsample(cfg: dict) -> tuple[Optional[int], Optional[float]]:
    """
    Resolve training set subsampling parameters:
    (train_sample_count: Optional[int], train_sample_ratio: Optional[float]).
    """
    dc = cfg.get('data', {})
    tc = cfg.get('training', {})

    count = dc.get('train_sample_count', tc.get('train_sample_count', None))
    ratio = dc.get('train_sample_ratio', tc.get('train_sample_ratio', None))

    if count is not None:
        try:
            count = int(count)
            if count <= 0:
                raise ValueError(f"train_sample_count must be > 0, got {count}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid train_sample_count '{count}': {e}") from e

    if ratio is not None:
        try:
            ratio = float(ratio)
            if not (0.0 < ratio <= 1.0):
                raise ValueError(f"train_sample_ratio must be in (0.0, 1.0], got {ratio}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid train_sample_ratio '{ratio}': {e}") from e

    return count, ratio


def _resolve_amplitude_range(cfg: dict) -> Optional[tuple[float, float]]:
    """
    Resolve RMS amplitude range filtering from cfg['data']['amplitude_range_nm'].
    Supports tuple/list [min_nm, max_nm] or individual keys amplitude_min_nm / amplitude_max_nm.
    """
    dc = cfg.get('data', {})
    raw = dc.get('amplitude_range_nm', None)
    min_nm = dc.get('amplitude_min_nm', None)
    max_nm = dc.get('amplitude_max_nm', None)

    if raw is not None:
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return float(raw[0]), float(raw[1])
        elif isinstance(raw, str):
            cleaned = raw.strip().lstrip('([').rstrip(')]')
            parts = [float(p.strip()) for p in cleaned.replace(',', ' ').split() if p.strip()]
            if len(parts) == 2:
                return float(parts[0]), float(parts[1])
        raise ValueError(f"Invalid amplitude_range_nm: expected 2 values [min_nm, max_nm], got {raw}")

    if min_nm is not None or max_nm is not None:
        mn = float(min_nm) if min_nm is not None else 0.0
        mx = float(max_nm) if max_nm is not None else float('inf')
        return mn, mx

    return None


def _resolve_noise_config(cfg: dict) -> dict:
    """
    Resolve on-the-fly detector noise configuration from cfg['data']['noise'].
    Supports keys: enabled, type, photons_per_frame, photons_per_image, read_noise_e, seed.
    """
    dc = cfg.get('data', {})
    nc = dc.get('noise', {})
    if not isinstance(nc, dict):
        nc = {}

    enabled_raw = dc.get('noise_enabled', nc.get('enabled', False))
    if isinstance(enabled_raw, str):
        enabled = enabled_raw.strip().lower() in ('true', 'yes', '1', 'on')
    else:
        enabled = bool(enabled_raw)

    n_ph_raw = dc.get('photons_per_frame', dc.get('photons_per_image', nc.get('photons_per_frame', nc.get('photons_per_image', 1.0e6))))
    if n_ph_raw is None or str(n_ph_raw).strip().lower() in ('none', 'null', 'inf', 'clean'):
        photons_per_frame = float('inf')
        enabled = False
    else:
        photons_per_frame = float(n_ph_raw)

    read_noise = float(dc.get('read_noise_e', nc.get('read_noise_e', 0.0)))
    noise_type = str(nc.get('type', 'poisson')).strip().lower()
    seed = int(nc.get('seed', dc.get('split_seed', 42)))

    return {
        'enabled': enabled,
        'type': noise_type,
        'photons_per_frame': photons_per_frame,
        'photons_per_image': photons_per_frame,
        'read_noise_e': read_noise,
        'seed': seed,
    }


class WarmupReduceLROnPlateau:
    """
    Combines initial linear warmup over `warmup_steps` with standard
    torch.optim.lr_scheduler.ReduceLROnPlateau for epoch-level plateau reduction.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 0,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 2,
        min_lr: float = 1.0e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._step_count = 0
        if self.warmup_steps > 0:
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base_lr / float(self.warmup_steps)
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
        )

    def step_batch(self) -> None:
        """Call per batch during training to perform linear warmup if step_count < warmup_steps."""
        if self._step_count < self.warmup_steps:
            self._step_count += 1
            ratio = min(1.0, float(self._step_count + 1) / float(max(1, self.warmup_steps)))
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = base_lr * ratio

    def step(self, metrics: float) -> None:
        """Call per epoch after validation to update ReduceLROnPlateau."""
        self.plateau.step(metrics)

    def get_last_lr(self) -> list[float]:
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            'warmup_steps': self.warmup_steps,
            'base_lrs': self.base_lrs,
            'step_count': self._step_count,
            'plateau_state': self.plateau.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.warmup_steps = state.get('warmup_steps', self.warmup_steps)
        self.base_lrs = state.get('base_lrs', self.base_lrs)
        self._step_count = state.get('step_count', 0)
        if 'plateau_state' in state:
            self.plateau.load_state_dict(state['plateau_state'])


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: dict,
    steps_per_epoch: int,
    total_epochs: int,
):
    """
    Build learning rate scheduler based on resolved scheduler config.
    Supports:
      - 'constant': linear warmup -> constant at base_lr
      - 'cosine': linear warmup -> half-period cosine decay to min_lr
      - 'plateau': linear warmup -> epoch-level validation-triggered plateau reduction
    """
    sched_type = sched_cfg.get('type', 'constant').lower()
    warmup_steps = sched_cfg.get('warmup_steps', 0)
    base_lr = optimizer.param_groups[0]['lr']
    min_lr = sched_cfg.get('min_lr', 1.0e-6)

    if sched_type == 'constant':
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif sched_type == 'cosine':
        total_steps = total_epochs * steps_per_epoch
        decay_steps = max(1, total_steps - warmup_steps)
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(decay_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine_decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif sched_type == 'cosine_tail':
        total_steps = total_epochs * steps_per_epoch
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0
        remaining_steps = max(1, total_steps - warmup_steps)

        tail_epochs = sched_cfg.get('tail_epochs', None)
        tail_ratio = float(sched_cfg.get('tail_ratio', 0.20))

        if tail_epochs is not None and tail_epochs > 0:
            tail_steps = min(remaining_steps - 1, int(tail_epochs * steps_per_epoch))
        else:
            tail_steps = int(remaining_steps * max(0.0, min(tail_ratio, 0.95)))

        decay_steps = max(1, remaining_steps - tail_steps)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if step >= warmup_steps + decay_steps:
                return min_ratio
            progress = float(step - warmup_steps) / float(decay_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine_decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif sched_type == 'plateau':
        patience = sched_cfg.get('patience', 2)
        factor = sched_cfg.get('factor', 0.5)
        return WarmupReduceLROnPlateau(
            optimizer,
            warmup_steps=warmup_steps,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    else:
        raise ValueError(
            f"Unsupported scheduler type '{sched_type}'. Expected one of: 'constant', 'cosine', 'cosine_tail', 'plateau'"
        )


# ─────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate the model specified in cfg['model']['type'].

    Parameters
    ----------
    cfg : dict — full config loaded from YAML

    Returns
    -------
    nn.Module — TransformerCWFS or CNNCWFS
    """
    mc = cfg['model']
    model_type = mc['type'].lower()
    if 'dropout' in mc or model_type in ('rodcnn', 'cnn', 'transformer', 'toy'):
        mc['dropout'] = _resolve_dropout(cfg)
    if 'base_ch' in mc or model_type in ('rodcnn', 'cnn'):
        mc['base_ch'] = _resolve_base_ch(cfg)
    if 'stage_blocks' in mc or model_type in ('rodcnn', 'cnn'):
        mc['stage_blocks'] = _resolve_stage_blocks(cfg)
    kwargs = {k: v for k, v in mc.items() if k != 'type'}

    registry = {'transformer': TransformerCWFS, 'cnn': SIAMCNN, 'toy': SLPCWFS, 'rodcnn': RODCNN}
    if model_type not in registry:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(registry)}")

    cls = registry[model_type]
    accepted = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


# ─────────────────────────────────────────────────────────────────────
# Learning-rate schedule: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────

def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────
# Checkpoint management
# ─────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """Keeps the best save_top_k checkpoints by ascending metric (lower is better)."""

    def __init__(self, checkpoint_dir: str, save_top_k: int = 3):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self._heap: list[tuple[float, str]] = []   # (metric, path) — min-heap by negated value
        self._scan_existing()

    def _scan_existing(self) -> None:
        """Populate heap with existing checkpoints in self.dir."""
        for pt in self.dir.glob("epoch*.pt"):
            m = re.search(r"epoch(\d+)_.*wfe([\d\.]+)nm\.pt", pt.name)
            if m:
                metric = float(m.group(2)) * 1e-9
                heapq.heappush(self._heap, (-metric, str(pt)))
        while len(self._heap) > self.save_top_k:
            _, worst_path = heapq.heappop(self._heap)
            try:
                os.remove(worst_path)
            except FileNotFoundError:
                pass

    def save(self, state: dict, metric: float, epoch: int) -> None:
        path = str(self.dir / f"epoch{epoch:03d}_wfe{metric*1e9:.1f}nm.pt")
        torch.save(state, path)
        # heap stores (−metric, path) so that the worst checkpoint sits at top
        heapq.heappush(self._heap, (-metric, path))
        if len(self._heap) > self.save_top_k:
            _, worst_path = heapq.heappop(self._heap)
            try:
                os.remove(worst_path)
            except FileNotFoundError:
                pass

    def best_path(self) -> str | None:
        if not self._heap:
            return None
        return min(self._heap, key=lambda x: x[0])[1]   # smallest −metric = best

    def latest_checkpoint(self) -> tuple[Optional[int], Optional[str]]:
        """Find the checkpoint with the highest epoch number in self.dir."""
        best_ep = -1
        best_pt = None
        for pt in self.dir.glob("epoch*.pt"):
            m = re.search(r"epoch(\d+)_", pt.name)
            if m:
                ep = int(m.group(1))
                if ep > best_ep:
                    best_ep = ep
                    best_pt = str(pt)
        if best_ep >= 0:
            return best_ep, best_pt
        return None, None

    def best_checkpoint(self) -> tuple[Optional[float], Optional[int], Optional[str]]:
        """Returns (best_val_wfe, best_epoch, best_path)."""
        best_val = float('inf')
        best_ep = -1
        best_pt = None
        for pt in self.dir.glob("epoch*.pt"):
            m = re.search(r"epoch(\d+)_.*wfe([\d\.]+)nm\.pt", pt.name)
            if m:
                ep = int(m.group(1))
                wfe = float(m.group(2)) * 1e-9
                if wfe < best_val:
                    best_val = wfe
                    best_ep = ep
                    best_pt = str(pt)
        if best_pt:
            return best_val, best_ep, best_pt
        return None, None, None


class LogMSELoss(nn.Module):
    """
    Computes the natural logarithm of Mean Squared Error:
        loss = ln(MSE) = torch.log(torch.clamp(mse, min=eps))
    """
    def __init__(self, eps: float = 1e-30):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        val = self.mse(pred, target)
        return torch.log(torch.clamp(val, min=self.eps))


class RMSWFELoss(nn.Module):
    """
    Root-Sum-Square (RMS) Wavefront Error loss across modes:
        loss = mean_b [ sqrt( sum_{j} (pred_{b,j} - target_{b,j})^2 + eps ) ]

    When predictions and targets are in nanometers OPD, loss directly equals
    the batch-mean total Wavefront Error (WFE) in nanometres.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_sq = (pred - target).pow(2)
        return torch.sqrt(diff_sq.sum(dim=-1) + self.eps).mean()


# ─────────────────────────────────────────────────────────────────────
# One training / validation epoch
# ─────────────────────────────────────────────────────────────────────

def _run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    scheduler,
    device: torch.device,
    label_std: torch.Tensor,
    label_mean: torch.Tensor,
    grad_clip: float,
    log_interval: int,
    is_train: bool,
    mode_idx: torch.Tensor = None,
    model_type: str = 'cnn',
    accumulate_every: int = 1,
    k_pairs_train: int | None = None,
    log_loss: bool = False,
    loss_type: str = 'mse',
) -> dict:
    """
    Run one epoch.  Returns a dict of scalar metrics.

    During training (is_train=True) the model is updated with AMP and
    gradient clipping.  During validation the model runs in eval mode with
    no gradient computation.
    """
    is_rodcnn = model_type.lower() == 'rodcnn'
    model.train(is_train)
    if loss_type == 'wfe_rms':
        criterion = RMSWFELoss()
    elif loss_type == 'log_mse' or log_loss:
        criterion = LogMSELoss()
    else:
        criterion = nn.MSELoss()

    total_loss = torch.zeros((), device=device)
    all_pred   = []
    all_target = []
    t0 = time.time()
    t_last = t0
    last_batch = 0

    # Prepare label stats, selecting the same columns/order as the dataset
    lm = label_mean.to(device)
    ls = label_std.to(device)
    if mode_idx is not None:
        lm = lm[mode_idx]
        ls = ls[mode_idx]

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    t_iter_start = time.time()
    t_data_window = 0.0
    t_gpu_window = 0.0

    with ctx:
        for batch_idx, batch in enumerate(loader):
            t_data_window += time.time() - t_iter_start
            t_gpu_start = time.time()

            if is_rodcnn:
                I1 = batch['I1'].to(device, non_blocking=True)
                I2 = batch['I2'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)  # [B, n_outputs]
                k_p = k_pairs_train if is_train else None
                with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                    pred = model(I1, I2, k_pairs=k_p)                  # [B, n_outputs]
                    loss = criterion(pred, labels) / accumulate_every
            else:
                I1 = batch['I1'].to(device, non_blocking=True)
                I2 = batch['I2'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)   # z-scored
                input_mode = getattr(model, 'input_mode', 'pairs')
                with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                    if input_mode == 'two_stream':
                        pred = model(I1, I2)
                        loss = criterion(pred, labels)
                    elif input_mode == 'r_stack':
                        R = batch['R'].to(device, non_blocking=True)
                        pred = model(R)                        # [B*T², n_outputs]
                        TT = pred.shape[0] // labels.shape[0]
                        labels = labels.repeat_interleave(TT, dim=0)  # [B*T², n_modes]
                        loss = criterion(pred, labels)
                    else:  # 'pairs' (default)
                        r  = batch['r'].to(device, non_blocking=True)
                        pred = model(I1, I2, r)
                        loss = criterion(pred, labels)

            is_last_in_window = ((batch_idx + 1) % accumulate_every == 0) or (batch_idx + 1 == len(loader))
            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if is_last_in_window:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                    if hasattr(scheduler, 'step_batch'):
                        scheduler.step_batch()
                    elif not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.detach() * accumulate_every

            # accumulate detached predictions on GPU for physical metrics
            all_pred.append(pred.detach())
            all_target.append(labels.detach())

            t_gpu_window += time.time() - t_gpu_start

            # Intermediate logging
            elapsed = time.time() - t0
            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                avg_loss = (total_loss / (batch_idx + 1)).item()
                batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                n_examples_done = (batch_idx + 1) * labels.shape[0]
                ex_per_sec = n_examples_done / elapsed if elapsed > 0 else 0
                remaining_batches = len(loader) - (batch_idx + 1)
                eta_sec = remaining_batches / batches_per_sec if batches_per_sec > 0 else 0

                dt_window = time.time() - t_last
                d_batches = batch_idx + 1 - last_batch
                d_ex = d_batches * labels.shape[0]
                inst_rate = d_ex / dt_window if dt_window > 0 else ex_per_sec
                avg_data_ms = (t_data_window / d_batches) * 1000.0 if d_batches > 0 else 0.0
                avg_gpu_ms = (t_gpu_window / d_batches) * 1000.0 if d_batches > 0 else 0.0
                t_last = time.time()
                last_batch = batch_idx + 1
                t_data_window = 0.0
                t_gpu_window = 0.0

                phase = "train" if is_train else "val"
                loss_fmt = f"{avg_loss:6.2f} nm" if loss_type == 'wfe_rms' else (f"{avg_loss:6.2f}" if avg_loss >= 1.0 else f"{avg_loss:.4f}")
                print(f"  [{phase}] batch {batch_idx+1:4d}/{len(loader)}  "
                      f"loss={loss_fmt}  "
                      f"time={elapsed:6.0f}s  rate={inst_rate:5.1f} ex/s (gpu={avg_gpu_ms:4.1f}ms, data={avg_data_ms:4.1f}ms) eta={eta_sec:5.0f}s", end="")
                if is_train:
                    lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                    print(f"  lr={lr:.2e}", end="")
                print()

            t_iter_start = time.time()

    all_pred   = (torch.cat(all_pred,   dim=0) * ls + lm).cpu()
    all_target = (torch.cat(all_target, dim=0) * ls + lm).cpu()

    mode_rms  = per_mode_rms(all_pred, all_target)
    wfe_rms   = total_wfe_rms(all_pred, all_target).item()
    strehl    = strehl_proxy(torch.tensor(wfe_rms)).item()

    return {
        'loss':        (total_loss / len(loader)).item(),
        'wfe_rms':     wfe_rms,
        'strehl':      strehl,
        'mode_rms':    mode_rms.tolist(),
        'last_pred':   all_pred[-1].numpy(),
        'last_target': all_target[-1].numpy(),
    }


# ─────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────

NOLL_MODE_NAMES = {
    2: 'Z2 (x-tilt)', 3: 'Z3 (y-tilt)', 4: 'Z4 (defocus)',
    5: 'Z5 (obl-astig)', 6: 'Z6 (vert-astig)',
    7: 'Z7 (vert-coma)', 8: 'Z8 (horiz-coma)',
    9: 'Z9 (vert-trefoil)', 10: 'Z10 (obl-trefoil)',
    11: 'Z11 (spherical)',
    12: 'Z12 (2nd-vert-astig)', 13: 'Z13 (2nd-obl-astig)',
    14: 'Z14 (obl-quadrafoil)', 15: 'Z15 (vert-quadrafoil)',
}


def _format_mode_list(trained_modes: list[int]) -> str:
    """Pretty-print a Noll mode list, collapsing contiguous runs to Z{a}-Z{b}."""
    is_contiguous = trained_modes == list(range(trained_modes[0], trained_modes[-1] + 1))
    if is_contiguous and len(trained_modes) > 1:
        return f"Z{trained_modes[0]}–Z{trained_modes[-1]}"
    return ",".join(f"Z{m}" for m in trained_modes)


def _setup_slurm_gpu() -> None:
    """
    Auto-detect Slurm allocated GPU / MIG device if CUDA_VISIBLE_DEVICES is unset.
    On shared HPC nodes where jobs are assigned specific MIG slices (e.g. IDX:2),
    this prevents processes from defaulting to GPU 0 and contending with other jobs.
    Auto-discovers active job on the current node if $SLURM_JOB_ID is stale or missing.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return

    job_id = os.environ.get("SLURM_JOB_ID")
    out = ""
    if job_id:
        try:
            out = subprocess.check_output(
                ["scontrol", "show", "job", job_id, "-d"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except Exception:
            out = ""

    # If SLURM_JOB_ID was missing or stale (e.g. node relaunched), query active job for this user
    if not out or "Invalid job id" in out:
        try:
            node = subprocess.check_output(["hostname", "-s"], text=True, timeout=2).strip()
            user = os.environ.get("USER", "")
            sq = subprocess.check_output(
                ["squeue", "-u", user, "-w", node, "-h", "-o", "%i"],
                text=True,
                errors="replace",
                timeout=5,
            ).strip()
            if sq:
                job_id = sq.split()[0]
                os.environ["SLURM_JOB_ID"] = job_id
                out = subprocess.check_output(
                    ["scontrol", "show", "job", job_id, "-d"],
                    text=True,
                    errors="replace",
                    timeout=5,
                )
        except Exception:
            return

    try:
        m = re.search(r"\(IDX:([0-9,]+)\)", out)
        if m:
            indices = [int(x) for x in m.group(1).split(",") if x.isdigit()]
            if indices:
                idx = indices[0]
                mig_out = subprocess.check_output(
                    ["nvidia-smi", "-L"],
                    text=True,
                    errors="replace",
                    timeout=5,
                )
                mig_uuids = re.findall(r"(MIG-[a-f0-9-]+)", mig_out)
                if mig_uuids and 0 <= idx < len(mig_uuids):
                    target = mig_uuids[idx]
                    os.environ["CUDA_VISIBLE_DEVICES"] = target
                    print(f"[Slurm GPU] Auto-bound to assigned MIG device IDX {idx} ({target})")
                    return
                gpu_uuids = re.findall(r"(GPU-[a-f0-9-]+)", mig_out)
                if gpu_uuids and 0 <= idx < len(gpu_uuids):
                    target = gpu_uuids[idx]
                    os.environ["CUDA_VISIBLE_DEVICES"] = target
                    print(f"[Slurm GPU] Auto-bound to assigned GPU device IDX {idx} ({target})")
                    return
    except Exception:
        pass


_setup_slurm_gpu()


def _cleanup_orphaned_processes() -> None:
    """Terminate any lingering DataLoader or Inductor worker processes from prior crashed/interrupted runs."""
    try:
        current_pid = os.getpid()
        user = os.environ.get("USER", "")
        cmd = ["ps", "-u", user, "-o", "pid,ppid,args"] if user else ["ps", "-o", "pid,ppid,args"]
        out = subprocess.check_output(cmd, text=True, errors="replace")
        killed = []
        for line in out.splitlines():
            line_str = line.strip()
            # Match DataLoader forkserver managers, forkserver workers, or compile workers
            # (Do NOT target multiprocessing.resource_tracker, which is managed by the Python runtime)
            is_worker = (
                "multiprocessing.forkserver" in line_str
                or "compile_worker" in line_str
            )
            if is_worker:
                parts = line_str.split(None, 2)
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    pid = int(parts[0])
                    ppid = int(parts[1])
                    # Only kill processes that do not belong to current process and are truly orphaned
                    # (parent is init/systemd ppid=1 or parent process has terminated / does not exist)
                    if pid != current_pid and ppid != current_pid and (ppid == 1 or not os.path.exists(f"/proc/{ppid}")):
                        try:
                            os.kill(pid, signal.SIGKILL)
                            killed.append(pid)
                        except OSError:
                            pass
        if killed:
            print(f"[Cleanup] Terminated {len(killed)} lingering background worker process(es) from prior runs: {killed}")
    except Exception:
        pass


def read_seed_summary(ckpt_dir: Path) -> Optional[dict]:
    """Read seed_summary.yaml if present in the seed checkpoint directory."""
    path = ckpt_dir / "seed_summary.yaml"
    if path.exists():
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except Exception:
            return None
    return None


def write_seed_summary(ckpt_dir: Path, data: dict) -> None:
    """Save seed_summary.yaml in the seed checkpoint directory."""
    path = ckpt_dir / "seed_summary.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _check_seed_completed(
    run_ckpt_dir: Path,
    run_seed: int,
    target_epochs: int,
    trial_dir: Optional[Path] = None,
) -> Optional[dict]:
    """
    Check if run_seed has already completed target_epochs.
    Returns a dict with seed completion metrics if completed, or None if incomplete/unstarted.
    """
    # 1. Primary check: seed_summary.yaml in run_ckpt_dir
    summary = read_seed_summary(run_ckpt_dir)
    if summary and summary.get('status') == 'COMPLETED':
        completed_ep = summary.get('completed_epochs', 0)
        best_ckpt = summary.get('best_ckpt_path')
        if completed_ep >= target_epochs and best_ckpt and Path(best_ckpt).exists():
            return summary

    # 2. Check parent ensemble_summary.yaml if present
    parent_summary_path = run_ckpt_dir.parent / "ensemble_summary.yaml"
    if parent_summary_path.exists():
        try:
            with open(parent_summary_path) as f:
                ens_data = yaml.safe_load(f)
            for r in ens_data.get('runs', []):
                if r.get('seed') == run_seed:
                    ckpt_p = r.get('checkpoint')
                    if ckpt_p and Path(ckpt_p).exists():
                        val_wfe_nm = float(r.get('best_val_wfe_nm', float('inf')))
                        summary_data = {
                            'seed': run_seed,
                            'status': 'COMPLETED',
                            'best_val_wfe': val_wfe_nm * 1e-9,
                            'best_val_wfe_nm': val_wfe_nm,
                            'best_epoch': int(r.get('best_epoch', target_epochs)),
                            'best_ckpt_path': str(ckpt_p),
                            'completed_epochs': int(target_epochs),
                            'target_epochs': int(target_epochs),
                            'elapsed_s': 0.0,
                        }
                        write_seed_summary(run_ckpt_dir, summary_data)
                        return summary_data
        except Exception:
            pass

    # 3. Fallback: inspect existing checkpoints and logs
    pts = list(run_ckpt_dir.glob("epoch*.pt"))
    if pts:
        best_val = float('inf')
        best_ep = -1
        best_pt = None
        for pt in pts:
            m = re.search(r"epoch(\d+)_.*wfe([\d\.]+)nm\.pt", pt.name)
            if m:
                ep = int(m.group(1))
                wfe_nm = float(m.group(2))
                if wfe_nm < best_val:
                    best_val = wfe_nm
                    best_ep = ep
                    best_pt = str(pt)

        log_confirmed = False
        log_elapsed_s = 0.0
        search_dirs = []
        if trial_dir:
            search_dirs.append(trial_dir / "logs")
        search_dirs.append(run_ckpt_dir.parent.parent / "logs")

        target_pattern = re.compile(
            rf"\[TRAIN\] epoch s{run_seed}_e{target_epochs}/{target_epochs}:"
        )
        elapsed_pattern = re.compile(
            rf"\[\+\s*([\d\.]+)s\] \[TRAIN\] epoch s{run_seed}_e{target_epochs}/{target_epochs}:"
        )

        for ld in search_dirs:
            if ld.exists():
                for log_file in ld.rglob("*.txt"):
                    try:
                        with open(log_file, errors="replace") as lf:
                            content = lf.read()
                        if target_pattern.search(content):
                            log_confirmed = True
                            em = elapsed_pattern.search(content)
                            if em:
                                log_elapsed_s = float(em.group(1))
                            break
                    except Exception:
                        pass
                if log_confirmed:
                    break

        if log_confirmed and best_pt:
            summary_data = {
                'seed': run_seed,
                'status': 'COMPLETED',
                'best_val_wfe': best_val * 1e-9,
                'best_val_wfe_nm': best_val,
                'best_epoch': best_ep,
                'best_ckpt_path': best_pt,
                'completed_epochs': target_epochs,
                'target_epochs': target_epochs,
                'elapsed_s': log_elapsed_s,
            }
            write_seed_summary(run_ckpt_dir, summary_data)
            return summary_data

    return None


def _train_single_seed(
    cfg: dict,
    run_seed: int,
    train_ds: CWFSDataset,
    val_ds: CWFSDataset,
    device: torch.device,
    trained_modes: list[int],
    mode_idx: torch.Tensor | None,
    subset_mode: bool,
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
    ckpt_dir: Path,
    recorder: Optional[SparseRecorder] = None,
    seed_idx: int = 1,
    total_seeds: int = 1,
    resume_checkpoint: bool = True,
) -> dict:
    """
    Execute training and validation for a single random seed.
    """
    seed_everything(run_seed)

    dc = cfg['data']
    tc = cfg['training']
    lc = cfg['logging']
    mc = cfg['model']

    z_score_labels = _resolve_z_score_labels(cfg)
    loss_type = _resolve_loss_type(cfg)
    log_loss = (loss_type == 'log_mse')
    use_swa, swa_tail_epochs = _resolve_swa(cfg)
    is_rodcnn = mc['type'].lower() == 'rodcnn'
    n_workers = dc.get('num_workers', 4)
    batch_size = dc.get('batch_size', 4 if is_rodcnn else 64)
    accumulate_every = tc.get('accumulate_every', 1)
    k_pairs_train = mc.get('k_pairs_train', dc.get('k_pairs_train', 16))
    prefetch_factor = dc.get('prefetch_factor', 4) if n_workers > 0 else None

    # Deterministic loader generator
    gen = torch.Generator()
    gen.manual_seed(run_seed)

    do_validation = tc.get('validate', True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=gen,
        num_workers=n_workers, pin_memory=True, persistent_workers=(n_workers > 0),
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size if is_rodcnn else batch_size * 2, shuffle=False,
        num_workers=n_workers, pin_memory=True, persistent_workers=False,
        prefetch_factor=prefetch_factor,
    ) if do_validation else None

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    swa = ModelSWA(model) if use_swa else None

    compile_mode = tc.get('compile_mode', 'reduce-overhead')
    if compile_mode and device.type == 'cuda':
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"[Seed {run_seed}] Model compiled with mode '{compile_mode}'")
        except Exception as e:
            print(f"[Seed {run_seed}] Compilation failed; running eagerly: {e}")

    betas = _resolve_betas(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc['lr'],
        betas=betas,
        weight_decay=tc['weight_decay'],
    )
    epochs = tc['epochs']
    steps_per_epoch = math.ceil(len(train_loader) / accumulate_every)
    sched_cfg = _resolve_scheduler_config(cfg)
    scheduler = build_lr_scheduler(optimizer, sched_cfg, steps_per_epoch, epochs)

    use_amp = tc.get('amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    ckpt_mgr = CheckpointManager(str(ckpt_dir), save_top_k=lc.get('save_top_k', 3))

    es_patience = tc.get('early_stopping_patience', 30)
    es_min_delta = tc.get('early_stopping_min_delta', 0.0)
    es_counter = 0

    seed_label = f"Seed {run_seed} ({seed_idx}/{total_seeds})" if total_seeds > 1 else f"Seed {run_seed}"
    if sched_cfg['type'] == 'plateau':
        sched_info = f"plateau (patience={sched_cfg['patience']}, factor={sched_cfg['factor']}, min_lr={sched_cfg['min_lr']:.1e}, warmup_steps={sched_cfg['warmup_steps']})"
    elif sched_cfg['type'] == 'cosine_tail':
        tail_desc = f"tail_epochs={sched_cfg['tail_epochs']}" if sched_cfg['tail_epochs'] is not None else f"tail_ratio={sched_cfg['tail_ratio']:.2f}"
        sched_info = f"cosine_tail ({tail_desc}, warmup_steps={sched_cfg['warmup_steps']}, min_lr={sched_cfg['min_lr']:.1e})"
    else:
        sched_info = f"{sched_cfg['type']} (warmup_steps={sched_cfg['warmup_steps']}, min_lr={sched_cfg['min_lr']:.1e})"
    swa_info = f"enabled (tail_epochs={swa_tail_epochs})" if use_swa else "disabled"
    dropout_val = _resolve_dropout(cfg)
    base_ch_val = _resolve_base_ch(cfg)
    stage_blocks_val = _resolve_stage_blocks(cfg)
    print(f"\n{'─'*60}")
    print(f"  Training Run: {seed_label}")
    print(f"  Model        {mc['type']}  ({n_params:.2f}M params, base_ch={base_ch_val}, stage_blocks={stage_blocks_val}, dropout={dropout_val:.2f})")
    print(f"  Optimizer    AdamW (lr={tc['lr']:.1e}, betas=({betas[0]:.3f}, {betas[1]:.4f}), wd={tc['weight_decay']})")
    print(f"  Scheduler    {sched_info}")
    print(f"  SWA          {swa_info}")
    print(f"  Checkpoints  {ckpt_dir}")
    print(f"{'─'*60}")

    best_val_wfe = float('inf')
    best_epoch = -1
    start_epoch = 1

    # Check for mid-seed checkpoint resumption
    if resume_checkpoint:
        latest_ep, latest_ckpt_path = ckpt_mgr.latest_checkpoint()
        if latest_ckpt_path is not None and latest_ep is not None and latest_ep < epochs:
            try:
                ckpt_data = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
                raw_model = getattr(model, '_orig_mod', model)
                if 'raw_model_state' in ckpt_data:
                    raw_model.load_state_dict(ckpt_data['raw_model_state'])
                elif 'model_state' in ckpt_data:
                    raw_model.load_state_dict(ckpt_data['model_state'])
                if swa is not None and 'swa_state' in ckpt_data:
                    swa.load_state_dict(ckpt_data['swa_state'])
                if 'optim_state' in ckpt_data:
                    optimizer.load_state_dict(ckpt_data['optim_state'])
                if 'scheduler_state' in ckpt_data and scheduler is not None:
                    scheduler.load_state_dict(ckpt_data['scheduler_state'])
                start_epoch = latest_ep + 1
                b_val, b_ep, _ = ckpt_mgr.best_checkpoint()
                if b_val is not None and b_val < float('inf'):
                    best_val_wfe = b_val
                    best_epoch = b_ep
                else:
                    best_val_wfe = ckpt_data.get('val_wfe_rms', float('inf'))
                    best_epoch = latest_ep
                print(f"  [Resume] Resuming {seed_label} from checkpoint: {Path(latest_ckpt_path).name} (starting at Epoch {start_epoch}/{epochs})")
                if recorder is not None:
                    recorder.log_message(f"[{seed_label}] Resuming from {Path(latest_ckpt_path).name} at Epoch {start_epoch}/{epochs}")
            except Exception as e:
                print(f"  [Resume Warning] Failed to load checkpoint {latest_ckpt_path}: {e}. Training from Epoch 1.")

    t_train_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        banner_prefix = f"[{seed_label}] " if total_seeds > 1 else ""
        print(f"\n{'='*60}")
        print(f"{banner_prefix}Epoch {epoch}/{epochs}")
        print('='*60)

        t_epoch = time.time()
        train_metrics = _run_epoch(
            model, train_loader, optimizer, scaler, scheduler, device,
            label_std, label_mean,
            grad_clip=tc.get('grad_clip', 1.0),
            log_interval=lc.get('log_interval', 100),
            is_train=True,
            mode_idx=mode_idx if subset_mode else None,
            model_type=mc['type'],
            accumulate_every=accumulate_every,
            k_pairs_train=k_pairs_train,
            log_loss=log_loss,
            loss_type=loss_type,
        )

        if use_swa and epoch >= (epochs - swa_tail_epochs + 1):
            swa.update(model)
            print(f"  [SWA] Updated running average (epoch {epoch}/{epochs}, {swa.n_models} models accumulated)")

        if do_validation and val_loader is not None:
            raw_model = getattr(model, '_orig_mod', model)
            val_ctx = swa.apply_shadow(raw_model) if (swa is not None and swa.n_models > 0) else nullcontext()
            with val_ctx:
                val_metrics = _run_epoch(
                    raw_model, val_loader, optimizer, scaler, scheduler, device,
                    label_std, label_mean,
                    grad_clip=tc.get('grad_clip', 1.0),
                    log_interval=0,
                    is_train=False,
                    mode_idx=mode_idx if subset_mode else None,
                    model_type=mc['type'],
                    accumulate_every=1,
                    k_pairs_train=None,
                    log_loss=log_loss,
                    loss_type=loss_type,
                )
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            val_metrics = None

        print(f"  Train loss={train_metrics['loss']:.4f}  "
              f"WFE={train_metrics['wfe_rms']*1e9:.1f} nm  "
              f"Strehl={train_metrics['strehl']:.3f}")

        if val_metrics is not None:
            print(f"  Val   loss={val_metrics['loss']:.4f}  "
                  f"WFE={val_metrics['wfe_rms']*1e9:.1f} nm  "
                  f"Strehl={val_metrics['strehl']:.3f}")

            subset_note = " [subset mode]" if subset_mode else ""
            detailed_epoch = lc.get('detailed_epoch_report', False)
            if detailed_epoch:
                print(f"  Per-mode val RMS (nm):{subset_note}")
                for i, rms in enumerate(val_metrics['mode_rms']):
                    mode_j = trained_modes[i]
                    name = NOLL_MODE_NAMES.get(mode_j, f"Z{mode_j}")
                    print(f"    {name:<28s} {rms*1e9:6.1f}")
            else:
                print(f"  Val RMS by Zernike order (nm):{subset_note}")
                for line in format_order_grouped_rms(trained_modes, val_metrics['mode_rms']):
                    print(line)

            # Save validation pupil reconstruction figure to test folder
            if recorder is not None and 'last_pred' in val_metrics:
                fig_prefix = f"seed{run_seed}_val" if total_seeds > 1 else "val"
                fig_path = recorder.save_pupil_reconstruction(
                    epoch=epoch,
                    c_true=val_metrics['last_target'],
                    c_pred=val_metrics['last_pred'],
                    trained_modes=trained_modes,
                    prefix=fig_prefix,
                )
                if fig_path:
                    print(f"  Pupil figure saved: {Path(fig_path).name}")

            val_wfe = val_metrics['wfe_rms']
            if val_wfe < best_val_wfe - es_min_delta:
                best_val_wfe = val_wfe
                best_epoch = epoch
                es_counter = 0
                raw_model = getattr(model, '_orig_mod', model)

                if swa is not None and swa.n_models > 0:
                    with swa.apply_shadow(raw_model):
                        save_model_state = raw_model.state_dict()
                    raw_state = raw_model.state_dict()
                    swa_state_dict = swa.state_dict()
                else:
                    save_model_state = raw_model.state_dict()
                    raw_state = None
                    swa_state_dict = None

                ckpt_state = {
                    'epoch':       epoch,
                    'model_state': save_model_state,
                    'optim_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'val_wfe_rms': val_wfe,
                    'label_mean':  label_mean.numpy(),
                    'label_std':   label_std.numpy(),
                    'config':      deepcopy(cfg),
                    'seed':        run_seed,
                    'z_score_labels': z_score_labels,
                    'log_loss':    log_loss,
                }
                if raw_state is not None:
                    ckpt_state['raw_model_state'] = raw_state
                if swa_state_dict is not None:
                    ckpt_state['swa_state'] = swa_state_dict
                ckpt_mgr.save(ckpt_state, val_wfe, epoch)
                saved_path = str(ckpt_mgr.dir / f"epoch{epoch:03d}_wfe{val_wfe*1e9:.1f}nm.pt")
                print(f"  *** New best val WFE: {val_wfe*1e9:.3f} nm — checkpoint saved ***")
                if recorder is not None:
                    recorder.record_checkpoint(
                        epoch=epoch,
                        metric_name="val_wfe_rms",
                        metric_val=val_wfe * 1e9,
                        checkpoint_path=saved_path,
                    )
            else:
                es_counter += 1
                print(f"  No improvement ({es_counter}/{es_patience})")
        else:
            print("  [Validation skipped]")
            train_wfe = train_metrics['wfe_rms']
            if train_wfe < best_val_wfe - es_min_delta:
                best_val_wfe = train_wfe
                best_epoch = epoch
                es_counter = 0
                raw_model = getattr(model, '_orig_mod', model)
                ckpt_state = {
                    'epoch':       epoch,
                    'model_state': raw_model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'val_wfe_rms': train_wfe,
                    'label_mean':  label_mean.numpy(),
                    'label_std':   label_std.numpy(),
                    'config':      deepcopy(cfg),
                    'seed':        run_seed,
                    'z_score_labels': z_score_labels,
                    'log_loss':    log_loss,
                }
                ckpt_mgr.save(ckpt_state, train_wfe, epoch)
                saved_path = str(ckpt_mgr.dir / f"epoch{epoch:03d}_train_wfe{train_wfe*1e9:.1f}nm.pt")
                print(f"  *** [No val] Checkpoint saved (train WFE: {train_wfe*1e9:.3f} nm) ***")
                if recorder is not None:
                    recorder.record_checkpoint(
                        epoch=epoch,
                        metric_name="train_wfe_rms",
                        metric_val=train_wfe * 1e9,
                        checkpoint_path=saved_path,
                    )
            else:
                es_counter += 1

        # Step epoch-level scheduler (ReduceLROnPlateau / WarmupReduceLROnPlateau)
        if isinstance(scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, WarmupReduceLROnPlateau)):
            step_metric = val_metrics['wfe_rms'] * 1e9 if val_metrics is not None else train_metrics['wfe_rms'] * 1e9
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(step_metric)
            curr_lr = optimizer.param_groups[0]['lr']
            if curr_lr < prev_lr:
                print(f"  [Scheduler] Learning rate reduced: {prev_lr:.2e} -> {curr_lr:.2e}")
                if recorder is not None:
                    recorder.log_message(f"[{seed_label}] Epoch {epoch}: LR reduced from {prev_lr:.2e} to {curr_lr:.2e}")

        epoch_elapsed = time.time() - t_epoch
        if epoch_elapsed >= 60:
            print(f"  Total epoch time: {epoch_elapsed:.1f}s ({epoch_elapsed / 60:.1f} min)")
        else:
            print(f"  Total epoch time: {epoch_elapsed:.1f}s")

        if recorder is not None:
            step_id = f"s{run_seed}_e{epoch}/{epochs}" if total_seeds > 1 else f"{epoch}/{epochs}"
            metrics_dict = {
                'seed': run_seed,
                'train_loss': train_metrics['loss'],
                'train_wfe_nm': train_metrics['wfe_rms'] * 1e9,
                'train_strehl': train_metrics['strehl'],
            }
            if val_metrics is not None:
                metrics_dict.update({
                    'val_loss': val_metrics['loss'],
                    'val_wfe_nm': val_metrics['wfe_rms'] * 1e9,
                    'val_strehl': val_metrics['strehl'],
                })
            recorder.record_step(
                step=step_id,
                metrics=metrics_dict,
                phase="train",
                step_name="epoch",
                epoch_time_s=epoch_elapsed,
                lr=optimizer.param_groups[0]['lr'],
            )

        if es_counter >= es_patience:
            print(f"  Early stopping triggered after {epoch} epochs.")
            if recorder is not None:
                recorder.log_message(f"[{seed_label}] Early stopping triggered after {epoch} epochs.")
            break

    # Final SWA evaluation across full validation set
    if swa is not None and swa.n_models > 0 and do_validation and val_loader is not None:
        raw_model = getattr(model, '_orig_mod', model)
        with swa.apply_shadow(raw_model):
            final_swa_metrics = _run_epoch(
                raw_model, val_loader, optimizer, scaler, scheduler, device,
                label_std, label_mean,
                grad_clip=tc.get('grad_clip', 1.0),
                log_interval=0,
                is_train=False,
                mode_idx=mode_idx if subset_mode else None,
                model_type=mc['type'],
                accumulate_every=1,
                k_pairs_train=None,
                log_loss=log_loss,
                loss_type=loss_type,
            )
        final_swa_wfe = final_swa_metrics['wfe_rms']
        print(f"  [SWA Final Evaluation] Val WFE: {final_swa_wfe*1e9:.3f} nm (vs previous best: {best_val_wfe*1e9:.3f} nm)")
        if final_swa_wfe < best_val_wfe:
            best_val_wfe = final_swa_wfe
            best_epoch = epochs
            with swa.apply_shadow(raw_model):
                save_model_state = raw_model.state_dict()
            ckpt_state = {
                'epoch':       epochs,
                'model_state': save_model_state,
                'raw_model_state': raw_model.state_dict(),
                'swa_state':   swa.state_dict(),
                'optim_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_wfe_rms': final_swa_wfe,
                'label_mean':  label_mean.numpy(),
                'label_std':   label_std.numpy(),
                'config':      deepcopy(cfg),
                'seed':        run_seed,
                'z_score_labels': z_score_labels,
                'log_loss':    log_loss,
            }
            ckpt_mgr.save(ckpt_state, final_swa_wfe, epochs)
            print(f"  *** [SWA] New best val WFE: {final_swa_wfe*1e9:.3f} nm — checkpoint saved ***")

    total_elapsed = time.time() - t_train_start
    best_path = ckpt_mgr.best_path()
    print(f"\n[{seed_label}] Training complete. Best val WFE: {best_val_wfe*1e9:.1f} nm at epoch {best_epoch} "
          f"(elapsed {total_elapsed/60:.1f} min)")
    print(f"Best checkpoint: {best_path}")

    # Save seed_summary.yaml
    seed_summary_data = {
        'seed': run_seed,
        'status': 'COMPLETED',
        'best_val_wfe': float(best_val_wfe),
        'best_val_wfe_nm': float(best_val_wfe * 1e9),
        'best_epoch': int(best_epoch),
        'best_ckpt_path': str(best_path) if best_path else None,
        'completed_epochs': int(epochs),
        'target_epochs': int(epochs),
        'elapsed_s': float(total_elapsed),
    }
    write_seed_summary(ckpt_dir, seed_summary_data)

    # Clean up model & workers before next run
    del model, optimizer, scheduler, scaler, train_loader
    if val_loader is not None:
        del val_loader
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    import gc; gc.collect()
    _cleanup_orphaned_processes()

    return {
        'seed': run_seed,
        'best_val_wfe': best_val_wfe,
        'best_epoch': best_epoch,
        'best_ckpt_path': best_path,
        'ckpt_dir': str(ckpt_dir),
        'elapsed_s': total_elapsed,
    }


def train(cfg: dict) -> None:
    """
    Full training run as specified by cfg.
    Supports single-seed training or multi-seed ensemble training with
    post-training test set evaluation (mode='average' or mode='best').

    Parameters
    ----------
    cfg : dict — config dict (typically loaded from a YAML file)
    """
    _setup_slurm_gpu()
    _cleanup_orphaned_processes()

    dc  = cfg['data']
    tc  = cfg['training']
    lc  = cfg['logging']
    mc  = cfg['model']

    hdf5_path = dc.get('hdf5_path')
    if not hdf5_path:
        raise ValueError("data.hdf5_path must be set in config or via --hdf5_path")

    # Resolve ensemble settings
    seeds, ensemble_mode, is_ensemble = resolve_ensemble_config(cfg)
    if is_ensemble:
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE TRAINING ENABLED")
        print(f"  Mode   : {ensemble_mode.upper()}")
        print(f"  Seeds  : {seeds} ({len(seeds)} total runs)")
        print(f"{'='*60}\n")

    n_modes_hdf5 = get_n_modes(hdf5_path)
    trained_modes = mc.get('trained_modes')
    if isinstance(trained_modes, str) and trained_modes.strip().lower() in ("all", "default", "none", "null", ""):
        trained_modes = None

    if trained_modes is None:
        if n_modes_hdf5 < 4:
            raise ValueError(
                f"HDF5 dataset has only {n_modes_hdf5} modes; cannot default to "
                f"modes minus piston/tip/tilt (requires at least 4 modes, Z4+)."
                f"set model.trained_modes e.g.  [5, 6, 7, 8, 9, 10]"
            )
        trained_modes = list(range(4, n_modes_hdf5 + 1))
        mc['trained_modes'] = trained_modes
        print(f"model.trained_modes not specified; defaulting to all available modes "
              f"minus piston/tip/tilt: {_format_mode_list(trained_modes)} ({len(trained_modes)} modes)")
    trained_modes = list(trained_modes)
    if len(trained_modes) == 0:
        raise ValueError("model.trained_modes must be a non-empty list of Noll indices.")
    if len(set(trained_modes)) != len(trained_modes):
        raise ValueError(f"model.trained_modes contains duplicate entries: {trained_modes}")
    if trained_modes != sorted(trained_modes):
        raise ValueError(f"model.trained_modes must be strictly ascending, got {trained_modes}")
    if trained_modes[0] < 1 or trained_modes[-1] > n_modes_hdf5:
        raise ValueError(
            f"model.trained_modes must be Noll indices in [1, {n_modes_hdf5}] "
            f"(HDF5 has {n_modes_hdf5} label columns); got {trained_modes}"
        )
    validate_trained_modes_pairing(trained_modes)

    mode_columns = [m - 1 for m in trained_modes]   # 0-based HDF5 column indices
    mode_idx     = torch.as_tensor(mode_columns, dtype=torch.long)
    n_outputs    = len(trained_modes)
    mc['n_outputs'] = n_outputs   # derived value consumed by build_model()

    subset_mode = n_outputs < n_modes_hdf5
    if subset_mode:
        print(f"\nSubset mode: training on {_format_mode_list(trained_modes)} "
              f"({n_outputs} modes) from HDF5 with {n_modes_hdf5} available modes")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── shared splits & label statistics ──────────────────────────────
    ratios = dc.get('split_ratios', [0.80, 0.10, 0.10])
    split_seed = dc.get('split_seed', 42)
    amp_range = _resolve_amplitude_range(cfg)
    train_idx, val_idx, test_idx = train_val_test_split(hdf5_path, ratios, split_seed, amplitude_range_nm=amp_range)
    if amp_range is not None:
        total_amp_samples = len(train_idx) + len(val_idx) + len(test_idx)
        print(f"RMS Amplitude Filter: [{amp_range[0]:.1f}, {amp_range[1]:.1f}] nm (matching: {total_amp_samples} examples)")

    orig_train_count = len(train_idx)
    train_count, train_ratio = _resolve_train_subsample(cfg)
    if train_count is not None:
        train_idx = subsample_indices(train_idx, count=train_count, seed=split_seed)
        print(f"Training Set Size: {len(train_idx)} examples (subsampled from {orig_train_count}, count={train_count})")
    elif train_ratio is not None and train_ratio < 1.0:
        train_idx = subsample_indices(train_idx, ratio=train_ratio, seed=split_seed)
        print(f"Training Set Size: {len(train_idx)} examples (subsampled from {orig_train_count}, {train_ratio*100:.1f}%)")

    val_sample_ratio = tc.get('val_sample_ratio', tc.get('val_fraction', dc.get('val_sample_ratio', 1.0)))
    if val_sample_ratio is None:
        val_sample_ratio = 1.0
    val_sample_ratio = float(val_sample_ratio)
    if not (0.0 < val_sample_ratio <= 1.0):
        raise ValueError(f"val_sample_ratio must be in (0.0, 1.0], got {val_sample_ratio}")

    original_val_count = len(val_idx)
    if val_sample_ratio < 1.0:
        val_idx = subsample_indices(val_idx, ratio=val_sample_ratio, seed=split_seed)
        print(f"Shared Split (seed={split_seed}): {len(train_idx)} train / {len(val_idx)} val "
              f"(subsampled from {original_val_count}, {val_sample_ratio*100:.1f}%) / {len(test_idx)} test")
    else:
        print(f"Shared Split (seed={split_seed}): {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    z_score_labels = _resolve_z_score_labels(cfg)
    loss_type = _resolve_loss_type(cfg)
    if z_score_labels:
        print("Computing label statistics for Z-score normalization...")
        stats = compute_label_stats(hdf5_path, train_idx)
        label_mean = torch.from_numpy(stats['mean'])
        label_std  = torch.from_numpy(stats['std'])
        dataset_label_stats = stats
        label_scale = 1.0
    else:
        print("Z-score label normalization: False (training on raw physical units scaled to nm OPD)")
        label_mean = torch.zeros(n_modes_hdf5, dtype=torch.float32)
        # Raw HDF5 labels are in metres OPD (~1e-7 m).
        # Setting label_std = 1e-9 scales raw labels to nanometres: (labels_m * 1e9).
        # When evaluating metrics: pred_phys = pred_nm * 1e-9 maps back to physical metres.
        label_std  = torch.full((n_modes_hdf5,), 1e-9, dtype=torch.float32)
        dataset_label_stats = None
        label_scale = 1e9

    noise_cfg = _resolve_noise_config(cfg)
    if noise_cfg['enabled']:
        print(f"Noise Injection: ENABLED (type: {noise_cfg['type']}, N_ph: {noise_cfg['photons_per_frame']:.1e} photons/frame, read_noise: {noise_cfg['read_noise_e']:.1f} e-)")
    else:
        print("Noise Injection: DISABLED (clean data)")

    log_loss = (loss_type == 'log_mse')
    if loss_type == 'wfe_rms':
        print("Loss function: Direct RMS WFE Loss (Root-Sum-Square OPD in nm)")
    elif log_loss:
        print("Loss function: ln(MSE) enabled (loss = ln(MSE))")
    else:
        print("Loss function: MSE Loss")

    # ── datasets ──────────────────────────────────────────────────────
    is_rodcnn = mc['type'].lower() == 'rodcnn'
    input_mode = mc.get('input_mode', 'pairs')
    return_stacks = True if is_rodcnn else dc.get('return_stacks', False)
    compute_r_stack = (return_stacks and input_mode == 'r_stack')
    augment = D4Augment(trained_modes) if dc.get('augment', True) else None
    preload = dc.get('preload', False)

    train_ds = CWFSDataset(hdf5_path, train_idx, label_stats=dataset_label_stats, transform=augment,
                           return_stacks=return_stacks,
                           compute_r_stack=compute_r_stack,
                           mode_columns=mode_columns if subset_mode else None,
                           preload=preload,
                           label_scale=label_scale,
                           noise_cfg=noise_cfg,
                           is_train=True)
    val_ds   = CWFSDataset(hdf5_path, val_idx,   label_stats=dataset_label_stats,
                           return_stacks=return_stacks,
                           compute_r_stack=compute_r_stack,
                           mode_columns=mode_columns if subset_mode else None,
                           preload=preload,
                           label_scale=label_scale,
                           noise_cfg=noise_cfg,
                           is_train=False)
    test_ds  = CWFSDataset(hdf5_path, test_idx,  label_stats=dataset_label_stats,
                           return_stacks=return_stacks,
                           compute_r_stack=compute_r_stack,
                           mode_columns=mode_columns if subset_mode else None,
                           preload=preload,
                           label_scale=label_scale,
                           noise_cfg=noise_cfg,
                           is_train=False)

    # ── sparse recorder ───────────────────────────────────────────────
    recorder = None
    if lc.get('sparse_record', True):
        task_name = f"train_{mc['type']}" + ("_ensemble" if is_ensemble else "")
        recorder = SparseRecorder(
            task_name=task_name,
            output_dir=lc.get('sparse_dir', None),
            config=cfg,
        )
        print(f"HPC Sparse Recording initialized: {recorder.filepath}")

    base_ckpt_dir = Path(lc['checkpoint_dir'])
    base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    resume_seeds = tc.get('resume_seeds', True)
    resume_epochs = tc.get('resume_epochs', True)

    # ── multi-seed training loop ──────────────────────────────────────
    run_results = []
    t_all_start = time.time()
    try:
        for idx, run_seed in enumerate(seeds, 1):
            run_ckpt_dir = base_ckpt_dir / f"seed_{run_seed}" if is_ensemble else base_ckpt_dir
            run_ckpt_dir.mkdir(parents=True, exist_ok=True)

            # Check if seed has already completed
            if resume_seeds:
                completed_seed_info = _check_seed_completed(
                    run_ckpt_dir=run_ckpt_dir,
                    run_seed=run_seed,
                    target_epochs=tc['epochs'],
                    trial_dir=base_ckpt_dir.parent if base_ckpt_dir.parent.exists() else None,
                )
                if completed_seed_info is not None:
                    seed_label = f"Seed {run_seed} ({idx}/{len(seeds)})" if len(seeds) > 1 else f"Seed {run_seed}"
                    best_wfe = completed_seed_info['best_val_wfe']
                    best_ep = completed_seed_info['best_epoch']
                    best_pt = completed_seed_info['best_ckpt_path']
                    print(f"\n{'─'*60}")
                    print(f"  [Resume] Skipping completed {seed_label}")
                    print(f"  Best Val WFE : {best_wfe*1e9:.3f} nm (epoch {best_ep})")
                    print(f"  Checkpoint   : {best_pt}")
                    print(f"{'─'*60}")
                    if recorder is not None:
                        recorder.log_message(
                            f"[{seed_label}] Skipped - already completed ({best_wfe*1e9:.3f} nm at epoch {best_ep})"
                        )
                    run_results.append({
                        'seed': run_seed,
                        'best_val_wfe': best_wfe,
                        'best_epoch': best_ep,
                        'best_ckpt_path': best_pt,
                        'ckpt_dir': str(run_ckpt_dir),
                        'elapsed_s': completed_seed_info.get('elapsed_s', 0.0),
                    })
                    continue

            res = _train_single_seed(
                cfg=cfg,
                run_seed=run_seed,
                train_ds=train_ds,
                val_ds=val_ds,
                device=device,
                trained_modes=trained_modes,
                mode_idx=mode_idx,
                subset_mode=subset_mode,
                label_mean=label_mean,
                label_std=label_std,
                ckpt_dir=run_ckpt_dir,
                recorder=recorder,
                seed_idx=idx,
                total_seeds=len(seeds),
                resume_checkpoint=resume_epochs,
            )
            run_results.append(res)

        total_training_time = time.time() - t_all_start

        # ── post-training test evaluation ──────────────────────────────
        print(f"\n{'='*60}")
        header_eval = f"TEST SET EVALUATION ({'ENSEMBLE: ' + ensemble_mode.upper() if is_ensemble else 'SINGLE MODEL'})"
        print(header_eval)
        print('='*60)

        from evaluate import load_checkpoint, _predict, _metrics_table, _print_table

        n_workers = dc.get('num_workers', 4)
        batch_size = dc.get('batch_size', 4 if is_rodcnn else 64)
        prefetch_factor = dc.get('prefetch_factor', 4) if n_workers > 0 else None
        test_loader = DataLoader(
            test_ds, batch_size=batch_size if is_rodcnn else batch_size * 2, shuffle=False,
            num_workers=n_workers, pin_memory=True, persistent_workers=False,
            prefetch_factor=prefetch_factor,
        )

        mode_names = [NOLL_MODE_NAMES.get(m, f"Z{m}") for m in trained_modes]
        lm_eval = label_mean[mode_columns] if subset_mode else label_mean
        ls_eval = label_std[mode_columns] if subset_mode else label_std

        final_test_metrics = None
        summary_rows = []
        use_tta = _resolve_tta(cfg)
        if use_tta:
            print("Test-Time Augmentation (D4 TTA) enabled for test set evaluation.")

        if ensemble_mode == "best" or not is_ensemble:
            best_run = min(run_results, key=lambda x: x['best_val_wfe'])
            print(f"Selected best run: Seed {best_run['seed']} (Val WFE: {best_run['best_val_wfe']*1e9:.2f} nm)")
            print(f"Loading checkpoint: {best_run['best_ckpt_path']}")
            model, _ = load_checkpoint(best_run['best_ckpt_path'], device)
            pred, target = _predict(
                model, test_loader, lm_eval, ls_eval, device,
                labels_are_zscored=True,
                tta=use_tta, trained_modes=trained_modes,
            )
            final_test_metrics = _metrics_table(pred, target)
            header_str = f"Test Set Evaluation: Best Model (Seed {best_run['seed']})" if is_ensemble else "Test Set Evaluation"
            _print_table(final_test_metrics, header=header_str, mode_names=mode_names)
            best_run['test_wfe_rms_nm'] = final_test_metrics['wfe_rms_nm']
            best_run['test_strehl'] = final_test_metrics['strehl']
            summary_rows.append([f"Seed {best_run['seed']} (Best)", f"{final_test_metrics['wfe_rms_nm']:.2f}", f"{final_test_metrics['strehl']:.4f}"])

            if is_ensemble and best_run['best_ckpt_path']:
                import shutil
                best_target = base_ckpt_dir / "best_ensemble.pt"
                try:
                    shutil.copy2(best_run['best_ckpt_path'], best_target)
                    print(f"Best ensemble checkpoint saved to: {best_target}")
                except Exception:
                    pass
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        elif ensemble_mode == "average":
            print(f"Evaluating all {len(run_results)} models on test set and averaging predictions...")
            all_preds = []
            test_target = None
            for r in run_results:
                model, _ = load_checkpoint(r['best_ckpt_path'], device)
                pred, test_target = _predict(
                    model, test_loader, lm_eval, ls_eval, device,
                    labels_are_zscored=True,
                    tta=use_tta, trained_modes=trained_modes,
                )
                all_preds.append(pred)
                m = _metrics_table(pred, test_target)
                r['test_wfe_rms_nm'] = m['wfe_rms_nm']
                r['test_strehl'] = m['strehl']
                summary_rows.append([f"Seed {r['seed']}", f"{m['wfe_rms_nm']:.2f}", f"{m['strehl']:.4f}"])
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            ens_pred = average_predictions(all_preds)
            final_test_metrics = _metrics_table(ens_pred, test_target)
            _print_table(final_test_metrics, header=f"Test Set Evaluation: Ensemble Average ({len(run_results)} models)", mode_names=mode_names)
            summary_rows.append(["Ensemble Average", f"{final_test_metrics['wfe_rms_nm']:.2f}", f"{final_test_metrics['strehl']:.4f}"])

            # Comparative performance table
            print(f"\n{'─'*60}")
            print("Ensemble Performance Summary:")
            print(f"  {'Model / Seed':<25s} {'Test WFE (nm)':<15s} {'Strehl':<10s}")
            print(f"  {'-'*25} {'-'*15} {'-'*10}")
            for label, wfe, s in summary_rows:
                print(f"  {label:<25s} {wfe:<15s} {s:<10s}")
            print(f"{'─'*60}\n")

        # Record in SparseRecorder
        if recorder is not None:
            recorder.record_table(
                title=f"Test Set Evaluation ({'Ensemble ' + ensemble_mode if is_ensemble else 'Single Run'})",
                headers=["Run", "Test WFE (nm)", "Strehl"],
                rows=summary_rows,
            )
            recorder.record_step(
                step="final_test",
                metrics={
                    'test_wfe_rms_nm': final_test_metrics['wfe_rms_nm'],
                    'test_strehl': final_test_metrics['strehl'],
                },
                phase="test",
                step_name="ensemble_eval",
            )
            recorder.log_message(
                f"Training & Evaluation complete. Total elapsed: {total_training_time/60:.1f} min. "
                f"Final Test WFE: {final_test_metrics['wfe_rms_nm']:.2f} nm, Strehl: {final_test_metrics['strehl']:.4f}"
            )
            recorder.close(status="COMPLETED")

        # Save ensemble summary YAML
        summary_data = {
            'ensemble_mode': ensemble_mode,
            'is_ensemble': is_ensemble,
            'seeds': seeds,
            'split_seed': split_seed,
            'runs': [
                {
                    'seed': r['seed'],
                    'best_val_wfe_nm': r['best_val_wfe'] * 1e9,
                    'best_epoch': r['best_epoch'],
                    'checkpoint': r['best_ckpt_path'],
                    'test_wfe_rms_nm': r.get('test_wfe_rms_nm', None),
                    'test_strehl': r.get('test_strehl', None),
                }
                for r in run_results
            ],
            'final_test_metrics': final_test_metrics,
        }
        summary_path = base_ckpt_dir / "ensemble_summary.yaml"
        with open(summary_path, 'w') as f:
            yaml.safe_dump(summary_data, f, sort_keys=False)
        print(f"Ensemble summary saved to: {summary_path}")

    except Exception as e:
        if recorder is not None:
            recorder.log_message(f"[ERROR] Training aborted with exception: {e}")
            recorder.close(status=f"FAILED ({type(e).__name__})")
        raise



# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def _parse_args(cmd_args=None):
    parser = argparse.ArgumentParser(description="Train TransformerCWFS or CNNCWFS")
    parser.add_argument('--config', required=True,
                        help="Path to YAML config file (e.g. config/transformer.yaml)")
    parser.add_argument('--hdf5_path', default=None,
                        help="Path to the HDF5 training dataset (overrides config)")
    parser.add_argument('--sparse_dir', default=None,
                        help="Directory for sparse HPC logs (defaults to nn_WFS/tmp)")
    parser.add_argument('--no_sparse_record', action='store_true',
                        help="Disable sparse HPC log recording")
    parser.add_argument('--no_validation', action='store_true',
                        help="Skip validation epochs (useful for throughput profiling)")
    parser.add_argument('--val_sample_ratio', type=float, default=None,
                        help="Subsample fraction of validation examples (0.0 < ratio <= 1.0)")
    parser.add_argument('--z_score_labels', dest='z_score_labels', action='store_true', default=None,
                        help="Enable Z-score normalization of training labels")
    parser.add_argument('--no_z_score_labels', dest='z_score_labels', action='store_false',
                        help="Disable Z-score normalization of training labels (default)")
    parser.add_argument('--log_loss', dest='log_loss', action='store_true', default=None,
                        help="Return loss function as ln(MSE)")
    parser.add_argument('--no_log_loss', dest='log_loss', action='store_false',
                        help="Disable ln(MSE) loss (default: standard MSE)")
    parser.add_argument('--ln_loss', dest='log_loss', action='store_true', default=None,
                        help="Alias for --log_loss")
    parser.add_argument('--scheduler', '--lr_scheduler', dest='scheduler', default=None,
                        choices=['constant', 'cosine', 'cosine_tail', 'cosine_plateau_tail', 'plateau'],
                        help="Learning rate scheduler type ('constant', 'cosine', 'cosine_tail', 'plateau')")
    parser.add_argument('--min_lr', type=float, default=None,
                        help="Minimum learning rate floor for cosine decay or plateau")
    parser.add_argument('--tail_epochs', type=int, default=None,
                        help="Number of tail epochs to hold at minimum LR plateau (for cosine_tail scheduler)")
    parser.add_argument('--tail_ratio', type=float, default=None,
                        help="Fraction of training spent in extended tail plateau (for cosine_tail scheduler, default: 0.20)")
    parser.add_argument('--resume_seeds', dest='resume_seeds', action='store_true', default=None,
                        help="Skip seeds that have already completed target epochs (default: True)")
    parser.add_argument('--no_resume_seeds', dest='resume_seeds', action='store_false',
                        help="Do not skip completed seeds; retrain from start")
    parser.add_argument('--resume_epochs', dest='resume_epochs', action='store_true', default=None,
                        help="Resume incomplete seeds from their latest checkpoint (default: True)")
    parser.add_argument('--no_resume_epochs', dest='resume_epochs', action='store_false',
                        help="Do not resume incomplete seeds from checkpoints; start at Epoch 1")
    parser.add_argument('--swa', dest='swa', action='store_true', default=None,
                        help="Enable Stochastic Weight Averaging (SWA) during cosine tail epochs")
    parser.add_argument('--no_swa', dest='swa', action='store_false',
                        help="Disable Stochastic Weight Averaging (default)")
    parser.add_argument('--swa_tail_epochs', type=int, default=None,
                        help="Number of tail epochs to accumulate SWA weights (default: 5)")
    parser.add_argument('--tta', dest='tta', action='store_true', default=None,
                        help="Enable D4 dihedral Test-Time Augmentation during test evaluation")
    parser.add_argument('--no_tta', dest='tta', action='store_false',
                        help="Disable Test-Time Augmentation (default)")
    parser.add_argument('--betas', nargs=2, type=float, default=None, metavar=('BETA1', 'BETA2'),
                        help="AdamW momentum (beta1) and second-moment decay (beta2) (default: 0.9 0.999)")
    parser.add_argument('--beta1', type=float, default=None,
                        help="AdamW momentum coefficient beta1 (default: 0.9)")
    parser.add_argument('--beta2', type=float, default=None,
                        help="AdamW second-moment decay coefficient beta2 (default: 0.999)")
    parser.add_argument('--dropout', type=float, default=None,
                        help="Dropout probability in MLPHead (default: from config, usually 0.0)")
    parser.add_argument('--base_ch', type=int, default=None,
                        help="ResNet stem base channels (default: from config, usually 32)")
    parser.add_argument('--stage_blocks', type=int, default=None,
                        help="BasicBlocks per ResNet stage (default: from config, usually 2)")
    parser.add_argument('--train_sample_count', type=int, default=None,
                        help="Subsample exact count of training examples (e.g. 5000)")
    parser.add_argument('--train_sample_ratio', type=float, default=None,
                        help="Subsample fraction of training examples (0.0 < ratio <= 1.0)")
    parser.add_argument('--amplitude_range_nm', nargs=2, type=float, default=None, metavar=('MIN_NM', 'MAX_NM'),
                        help="Filter dataset by RMS WFE range [min_nm, max_nm] in nm")
    parser.add_argument('--amplitude_min_nm', type=float, default=None,
                        help="Minimum RMS WFE threshold in nm")
    parser.add_argument('--amplitude_max_nm', type=float, default=None,
                        help="Maximum RMS WFE threshold in nm")
    parser.add_argument('--noise_enabled', dest='noise_enabled', action='store_true', default=None,
                        help="Enable on-the-fly detector noise injection")
    parser.add_argument('--no_noise', dest='noise_enabled', action='store_false',
                        help="Disable detector noise injection (clean data)")
    parser.add_argument('--photons_per_frame', type=float, default=None,
                        help="Incident photon flux per frame N_ph (default: from config, e.g. 1e6)")
    parser.add_argument('--photons_per_image', type=float, default=None,
                        help="Alias for --photons_per_frame")
    parser.add_argument('--read_noise_e', type=float, default=None,
                        help="Gaussian readout noise std in RMS electrons (e.g. 2.0)")
    # absorb arbitrary key=value overrides
    args, overrides = parser.parse_known_args(cmd_args)
    return args, overrides


if __name__ == '__main__':
    args, overrides = _parse_args()
    cfg = _load_config(args.config)

    if args.hdf5_path:
        overrides.append(f'data.hdf5_path={args.hdf5_path}')
    if args.sparse_dir:
        overrides.append(f'logging.sparse_dir={args.sparse_dir}')
    if args.no_sparse_record:
        overrides.append('logging.sparse_record=false')
    if args.no_validation:
        overrides.append('training.validate=false')
    if args.val_sample_ratio is not None:
        overrides.append(f'training.val_sample_ratio={args.val_sample_ratio}')
    if args.train_sample_count is not None:
        overrides.append(f'data.train_sample_count={args.train_sample_count}')
    if args.train_sample_ratio is not None:
        overrides.append(f'data.train_sample_ratio={args.train_sample_ratio}')
    if args.amplitude_range_nm is not None:
        overrides.append(f'data.amplitude_range_nm=[{args.amplitude_range_nm[0]},{args.amplitude_range_nm[1]}]')
    if args.amplitude_min_nm is not None:
        overrides.append(f'data.amplitude_min_nm={args.amplitude_min_nm}')
    if args.amplitude_max_nm is not None:
        overrides.append(f'data.amplitude_max_nm={args.amplitude_max_nm}')
    if args.noise_enabled is not None:
        overrides.append(f'data.noise.enabled={str(args.noise_enabled).lower()}')
    if args.photons_per_frame is not None:
        overrides.append(f'data.noise.photons_per_frame={args.photons_per_frame}')
    if args.photons_per_image is not None:
        overrides.append(f'data.noise.photons_per_image={args.photons_per_image}')
    if args.read_noise_e is not None:
        overrides.append(f'data.noise.read_noise_e={args.read_noise_e}')
    if args.z_score_labels is not None:
        overrides.append(f'data.z_score_labels={str(args.z_score_labels).lower()}')
    if args.log_loss is not None:
        overrides.append(f'training.log_loss={str(args.log_loss).lower()}')
    if args.scheduler is not None:
        overrides.append(f'training.scheduler.type={args.scheduler}')
    if args.min_lr is not None:
        overrides.append(f'training.scheduler.min_lr={args.min_lr}')
    if args.tail_epochs is not None:
        overrides.append(f'training.scheduler.tail_epochs={args.tail_epochs}')
    if args.tail_ratio is not None:
        overrides.append(f'training.scheduler.tail_ratio={args.tail_ratio}')
    if args.resume_seeds is not None:
        overrides.append(f'training.resume_seeds={str(args.resume_seeds).lower()}')
    if args.resume_epochs is not None:
        overrides.append(f'training.resume_epochs={str(args.resume_epochs).lower()}')
    if args.swa is not None:
        overrides.append(f'training.swa={str(args.swa).lower()}')
    if args.swa_tail_epochs is not None:
        overrides.append(f'training.swa_tail_epochs={args.swa_tail_epochs}')
    if args.tta is not None:
        overrides.append(f'evaluation.tta={str(args.tta).lower()}')
    if args.betas is not None:
        overrides.append(f'training.betas=[{args.betas[0]},{args.betas[1]}]')
    if args.beta1 is not None:
        overrides.append(f'training.beta1={args.beta1}')
    if args.beta2 is not None:
        overrides.append(f'training.beta2={args.beta2}')
    if args.dropout is not None:
        overrides.append(f'model.dropout={args.dropout}')
    if args.base_ch is not None:
        overrides.append(f'model.base_ch={args.base_ch}')
    if args.stage_blocks is not None:
        overrides.append(f'model.stage_blocks={args.stage_blocks}')

    cfg = _apply_overrides(cfg, overrides)
    train(cfg)


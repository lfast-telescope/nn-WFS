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
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required:  pip install pyyaml")

# ── project imports ──────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from dataset import (
    CWFSDataset, train_val_test_split, compute_label_stats,
    GroupedBatchSampler,
)
from models.transformer_cwfs import TransformerCWFS
from models.cnn_cwfs import SIAMCNN, RODCNN, CNNCWFS
from utils.augmentation import D4Augment
from utils.metrics import per_mode_rms, total_wfe_rms, strehl_proxy


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
            node = node.setdefault(k, {})
        # attempt numeric coercion
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
        node[keys[-1]] = value
    return cfg


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
    kwargs = {k: v for k, v in mc.items() if k != 'type'}

    if model_type == 'transformer':
        return TransformerCWFS(**kwargs)
    elif model_type == 'cnn':
        return SIAMCNN(**kwargs)
    elif model_type == 'rodcnn':
        return RODCNN(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose 'transformer', 'cnn', or 'rodcnn'."
        )


# ─────────────────────────────────────────────────────────────────────
# Learning-rate schedule: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────

def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
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


# ─────────────────────────────────────────────────────────────────────
# One training / validation epoch
# ─────────────────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scaler,
    scheduler,
    device: torch.device,
    label_std: torch.Tensor,
    label_mean: torch.Tensor,
    grad_clip: float,
    log_interval: int,
    is_train: bool,
) -> dict:
    """
    Run one epoch.  Returns a dict of scalar metrics.

    During training (is_train=True) the model is updated with AMP and
    gradient clipping.  During validation the model runs in eval mode with
    no gradient computation.

    Loss
    ----
    L1 on z-scored Zernike labels so each mode is weighted equally regardless
    of its physical amplitude.  Physical-unit metrics (WFE rms, Strehl proxy)
    are computed after denormalising predictions.
    """
    model.train(is_train)
    criterion = nn.L1Loss()

    total_loss = 0.0
    all_pred   = []
    all_target = []
    t0 = time.time()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            I1     = batch['I1'].to(device, non_blocking=True)
            I2     = batch['I2'].to(device, non_blocking=True)
            r      = batch['r'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)   # z-scored

            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                if isinstance(model, RODCNN):
                    pred_all = model(I1, I2)              # [B², n_outputs]
                    pred     = pred_all.mean(dim=0)        # [n_outputs]
                    loss     = criterion(pred, labels[0]) # labels[i] identical for all i
                else:
                    pred = model(I1, I2, r)               # [B, n_outputs]
                    loss = criterion(pred, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                scheduler.step()

            total_loss += loss.item()

            # accumulate denormalised predictions for physical metrics
            lm = label_mean.to(device)
            ls = label_std.to(device)
            if isinstance(model, RODCNN):
                all_pred.append((pred.detach().unsqueeze(0) * ls + lm).cpu())
                all_target.append((labels[0:1].detach() * ls + lm).cpu())
            else:
                all_pred.append((pred.detach() * ls + lm).cpu())
                all_target.append((labels.detach() * ls + lm).cpu())

            if is_train and log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                print(f"  batch {batch_idx+1}/{len(loader)}  "
                      f"loss={total_loss/(batch_idx+1):.4f}  "
                      f"lr={lr:.2e}  elapsed={elapsed:.0f}s")

    all_pred   = torch.cat(all_pred,   dim=0)
    all_target = torch.cat(all_target, dim=0)

    mode_rms  = per_mode_rms(all_pred, all_target)
    wfe_rms   = total_wfe_rms(all_pred, all_target).item()
    strehl    = strehl_proxy(torch.tensor(wfe_rms)).item()

    return {
        'loss':     total_loss / len(loader),
        'wfe_rms':  wfe_rms,
        'strehl':   strehl,
        'mode_rms': mode_rms.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────

NOLL_NAMES = [
    'Z2 (x-tilt)', 'Z3 (y-tilt)', 'Z4 (defocus)',
    'Z5 (obl-astig)', 'Z6 (vert-astig)',
    'Z7 (vert-coma)', 'Z8 (horiz-coma)',
    'Z9 (vert-trefoil)', 'Z10 (obl-trefoil)',
    'Z11 (spherical)',
    'Z12 (2nd-vert-astig)', 'Z13 (2nd-obl-astig)',
    'Z14', 'Z15',
]


def train(cfg: dict) -> None:
    """
    Full training run as specified by cfg.

    Parameters
    ----------
    cfg : dict — config dict (typically loaded from a YAML file)
    """
    dc  = cfg['data']
    tc  = cfg['training']
    lc  = cfg['logging']
    mc  = cfg['model']

    hdf5_path = dc.get('hdf5_path')
    if not hdf5_path:
        raise ValueError("data.hdf5_path must be set in config or via --hdf5_path")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── splits & label statistics ──────────────────────────────────────
    ratios = dc.get('split_ratios', [0.80, 0.10, 0.10])
    seed   = dc.get('split_seed', 42)
    train_idx, val_idx, _ = train_val_test_split(hdf5_path, ratios, seed)

    print(f"Split: {len(train_idx)} train / {len(val_idx)} val")
    print("Computing label statistics...")
    stats = compute_label_stats(hdf5_path, train_idx)
    label_mean = torch.from_numpy(stats['mean'])
    label_std  = torch.from_numpy(stats['std'])

    # ── datasets & loaders ────────────────────────────────────────────
    augment = D4Augment() if dc.get('augment', True) else None
    train_ds = CWFSDataset(hdf5_path, train_idx, label_stats=stats, transform=augment)
    val_ds   = CWFSDataset(hdf5_path, val_idx,   label_stats=stats)

    n_workers = dc.get('num_workers', 4)
    batch_size = dc.get('batch_size', 128)
    if mc['type'].lower() == 'rodcnn':
        group_size    = dc.get('group_size', batch_size)
        train_sampler = GroupedBatchSampler(
            train_idx, group_size, batch_size=batch_size, shuffle=True,
        )
        val_sampler   = GroupedBatchSampler(
            val_idx, group_size, batch_size=batch_size, shuffle=False,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler,
            num_workers=n_workers, pin_memory=True, persistent_workers=(n_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_sampler=val_sampler,
            num_workers=n_workers, pin_memory=True, persistent_workers=(n_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=True, persistent_workers=(n_workers > 0),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=n_workers, pin_memory=True, persistent_workers=(n_workers > 0),
        )

    # ── model ─────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {mc['type']}  ({n_params:.2f} M parameters)")

    # ── optimiser & scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc['lr'],
        weight_decay=tc['weight_decay'],
    )
    epochs       = tc['epochs']
    warmup_steps = tc.get('warmup_steps', 500)
    total_steps  = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, warmup_steps, total_steps),
    )

    use_amp = tc.get('amp', True) and device.type == 'cuda'
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ── checkpoint manager ────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(lc['checkpoint_dir'], save_top_k=lc.get('save_top_k', 3))

    # ── training loop ─────────────────────────────────────────────────
    best_val_wfe = float('inf')
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print('='*60)

        train_metrics = _run_epoch(
            model, train_loader, optimizer, scaler, scheduler, device,
            label_std, label_mean,
            grad_clip=tc.get('grad_clip', 1.0),
            log_interval=lc.get('log_interval', 100),
            is_train=True,
        )
        val_metrics = _run_epoch(
            model, val_loader, optimizer, scaler, scheduler, device,
            label_std, label_mean,
            grad_clip=tc.get('grad_clip', 1.0),
            log_interval=0,
            is_train=False,
        )

        print(f"  Train loss={train_metrics['loss']:.4f}  "
              f"WFE={train_metrics['wfe_rms']*1e9:.1f} nm  "
              f"Strehl={train_metrics['strehl']:.3f}")
        print(f"  Val   loss={val_metrics['loss']:.4f}  "
              f"WFE={val_metrics['wfe_rms']*1e9:.1f} nm  "
              f"Strehl={val_metrics['strehl']:.3f}")
        print("  Per-mode val RMS (nm):")
        for name, rms in zip(NOLL_NAMES, val_metrics['mode_rms']):
            print(f"    {name:<28s} {rms*1e9:6.1f}")

        val_wfe = val_metrics['wfe_rms']
        if val_wfe < best_val_wfe:
            best_val_wfe = val_wfe
            ckpt_state = {
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_wfe_rms': val_wfe,
                'label_mean':  label_mean.numpy(),
                'label_std':   label_std.numpy(),
                'config':      deepcopy(cfg),
            }
            ckpt_mgr.save(ckpt_state, val_wfe, epoch)
            print(f"  *** New best val WFE: {val_wfe*1e9:.1f} nm — checkpoint saved ***")

    print(f"\nTraining complete.  Best val WFE: {best_val_wfe*1e9:.1f} nm")
    print(f"Best checkpoint: {ckpt_mgr.best_path()}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Train TransformerCWFS or CNNCWFS")
    parser.add_argument('--config', required=True,
                        help="Path to YAML config file (e.g. config/transformer.yaml)")
    parser.add_argument('--hdf5_path', default=None,
                        help="Path to the HDF5 training dataset (overrides config)")
    # absorb arbitrary key=value overrides
    args, overrides = parser.parse_known_args()
    return args, overrides


if __name__ == '__main__':
    args, overrides = _parse_args()
    cfg = _load_config(args.config)

    if args.hdf5_path:
        overrides.append(f'data.hdf5_path={args.hdf5_path}')

    cfg = _apply_overrides(cfg, overrides)
    train(cfg)

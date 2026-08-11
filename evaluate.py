"""
evaluate.py — Evaluation, comparison, and on-sky fine-tuning for CWFS models.

Usage
-----
    # Evaluate a single checkpoint on the synthetic test set
    python evaluate.py --ckpt checkpoints/transformer/epoch050_wfe12.3nm.pt \\
                       --hdf5_path /data/cwfs_1M.h5

    # Side-by-side comparison of two checkpoints
    python evaluate.py --compare \\
                       --ckpt_transformer checkpoints/transformer/best.pt \\
                       --ckpt_cnn        checkpoints/cnn/best.pt \\
                       --hdf5_path /data/cwfs_1M.h5

    # On-sky inference (no labels required)
    python evaluate.py --ckpt checkpoints/transformer/best.pt --onsky_dir /data/onsky/

    # On-sky fine-tuning (freeze backbone, retrain head on labelled on-sky frames)
    python evaluate.py --ckpt checkpoints/transformer/best.pt \\
                       --onsky_dir /data/onsky/ --fine_tune --ft_epochs 20

    # Roddier-channel ablation on the test set
    python evaluate.py --ckpt checkpoints/transformer/best.pt \\
                       --hdf5_path /data/cwfs_1M.h5 --ablate_roddier
"""

# %%
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from dataset import CWFSDataset, train_val_test_split, get_n_modes
from train import build_model, NOLL_MODE_NAMES
from models.cnn_cwfs import RODCNN
from utils.metrics import per_mode_rms, total_wfe_rms, strehl_proxy


def trained_modes_from_config(model_cfg: dict) -> Optional[list]:
    """
    Recover the ordered list of Noll indices a checkpoint was trained on.

    Prefers the explicit `trained_modes` field (current schema).  Falls back
    to `n_outputs` for legacy checkpoints, reconstructing
    `trained_modes = list(range(1, n_outputs + 1))` (Z1..Zn) -- this matches
    the *actual* (bug-for-bug) behaviour those old checkpoints were trained
    under (contiguous truncation from column 0), not the newer Z2-start
    convention.  Returns None if neither field is present.
    """
    trained_modes = model_cfg.get('trained_modes')
    if trained_modes is not None:
        return list(trained_modes)
    n_outputs = model_cfg.get('n_outputs')
    if n_outputs is not None:
        return list(range(1, n_outputs + 1))
    return None


def mode_names_from_config(model_cfg: dict) -> Optional[list]:
    """Display names (e.g. 'Z5 (obl-astig)') for a checkpoint's trained modes."""
    trained_modes = trained_modes_from_config(model_cfg)
    if trained_modes is None:
        return None
    return [NOLL_MODE_NAMES.get(m, f"Z{m}") for m in trained_modes]


# ─────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str, device: torch.device) -> tuple[nn.Module, dict]:
    """
    Load a model and its training metadata from a checkpoint saved by train.py.

    Parameters
    ----------
    ckpt_path : str
    device    : torch.device

    Returns
    -------
    model : nn.Module  — loaded model in eval mode on device
    meta  : dict       — {'label_mean': ndarray, 'label_std': ndarray,
                           'val_wfe_rms': float, 'epoch': int, 'config': dict}
    """
    state = torch.load(ckpt_path, map_location=device)
    model = build_model(state['config']).to(device)
    model.load_state_dict(state['model_state'])
    model.eval()
    meta = {
        'label_mean':  state['label_mean'],
        'label_std':   state['label_std'],
        'val_wfe_rms': state.get('val_wfe_rms', float('nan')),
        'epoch':       state.get('epoch', -1),
        'config':      state['config'],
    }
    return model, meta


# ─────────────────────────────────────────────────────────────────────
# Internal evaluation engine
# ─────────────────────────────────────────────────────────────────────

def _predict(
    model: nn.Module,
    loader: DataLoader,
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
    device: torch.device,
    zero_r: bool = False,
    labels_are_zscored: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run model inference over a DataLoader and return denormalised predictions
    and targets (both in physical units).

    Parameters
    ----------
    zero_r : bool
        If True, replace the Roddier channel r with zeros to ablate its
        contribution.  Used by ablation_roddier().
    labels_are_zscored : bool
        Set True (default) when the DataLoader was built with label_stats set
        (i.e. labels in the batch are already z-scored and must be
        denormalised).  Set False when label_stats=None was used and the
        batch labels are already in physical units (e.g. in compare_models).
    """
    all_pred, all_target = [], []
    lm = label_mean.to(device)
    ls = label_std.to(device)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            I1     = batch['I1'].to(device, non_blocking=True)
            I2     = batch['I2'].to(device, non_blocking=True)
            r      = batch['r'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            if isinstance(model, RODCNN):
                pred_all   = model(I1, I2)                       # [B², n_outputs]
                pred       = pred_all.mean(dim=0, keepdim=True)  # [1, n_outputs]
                labels_eff = labels[0:1]
            else:
                if zero_r:
                    r = torch.zeros_like(r)
                pred       = model(I1, I2, r)                    # [B, n_outputs]
                labels_eff = labels

            pred_phys   = pred * ls + lm
            target_phys = labels_eff * ls + lm if labels_are_zscored else labels_eff

            all_pred.append(pred_phys.cpu())
            all_target.append(target_phys.cpu())

    return torch.cat(all_pred, 0), torch.cat(all_target, 0)


def _metrics_table(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute the full set of evaluation metrics from denormalised tensors."""
    mode_rms = per_mode_rms(pred, target)
    wfe      = total_wfe_rms(pred, target).item()
    s        = strehl_proxy(torch.tensor(wfe)).item()
    return {
        'mode_rms_nm': (mode_rms * 1e9).tolist(),
        'wfe_rms_nm':  wfe * 1e9,
        'strehl':      s,
    }


def _print_table(metrics: dict, header: str = '', mode_names: Optional[list] = None) -> None:
    """Pretty-print a metrics dict to stdout."""
    if header:
        print(f"\n{'─'*60}")
        print(header)
        print('─'*60)
    print(f"  Total WFE rms : {metrics['wfe_rms_nm']:.2f} nm")
    print(f"  Strehl proxy  : {metrics['strehl']:.4f}")
    print("  Per-mode RMS (nm):")
    names = mode_names if mode_names is not None else [f"mode {i+1}" for i in range(len(metrics['mode_rms_nm']))]
    for name, rms in zip(names, metrics['mode_rms_nm']):
        print(f"    {name:<30s} {rms:6.2f}")


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def eval_synthetic(
    model: nn.Module,
    test_loader: DataLoader,
    label_mean: np.ndarray,
    label_std: np.ndarray,
    device: Optional[torch.device] = None,
    mode_names: Optional[list] = None,
) -> dict:
    """
    Evaluate model on the synthetic test set.

    Parameters
    ----------
    model       : nn.Module (already on device, eval mode)
    test_loader : DataLoader yielding z-scored {'I1','I2','r','labels'}
    label_mean  : ndarray — training-set label mean
    label_std   : ndarray — training-set label std
    device      : torch.device (defaults to model's first parameter device)
    mode_names  : list[str], optional — per-mode display names (see
        `mode_names_from_config`); falls back to generic names if omitted.

    Returns
    -------
    dict with keys 'mode_rms_nm' (list), 'wfe_rms_nm' (float), 'strehl' (float)
    """
    if device is None:
        device = next(model.parameters()).device
    lm = torch.from_numpy(label_mean)
    ls = torch.from_numpy(label_std)
    pred, target = _predict(model, test_loader, lm, ls, device)
    metrics = _metrics_table(pred, target)
    _print_table(metrics, header="Synthetic test-set evaluation", mode_names=mode_names)
    return metrics


def ablation_roddier(
    model: nn.Module,
    test_loader: DataLoader,
    label_mean: np.ndarray,
    label_std: np.ndarray,
    device: Optional[torch.device] = None,
    mode_names: Optional[list] = None,
) -> dict:
    """
    Compare model performance with and without the Roddier r channel.

    Runs two inference passes:
        1. Normal inference (with r)
        2. r replaced by zeros (ablated)

    Returns
    -------
    dict with keys 'with_r' and 'without_r', each a metrics dict.
    """
    if device is None:
        device = next(model.parameters()).device
    lm = torch.from_numpy(label_mean)
    ls = torch.from_numpy(label_std)

    pred_full, target = _predict(model, test_loader, lm, ls, device, zero_r=False)
    pred_ablated, _   = _predict(model, test_loader, lm, ls, device, zero_r=True)

    m_full    = _metrics_table(pred_full,    target)
    m_ablated = _metrics_table(pred_ablated, target)

    _print_table(m_full,    header="Ablation: WITH Roddier signal r", mode_names=mode_names)
    _print_table(m_ablated, header="Ablation: WITHOUT Roddier signal r (r = 0)", mode_names=mode_names)

    delta_wfe = m_ablated['wfe_rms_nm'] - m_full['wfe_rms_nm']
    print(f"\n  WFE degradation without r: +{delta_wfe:.2f} nm  "
          f"({delta_wfe/m_full['wfe_rms_nm']*100:.1f}% increase)")

    return {'with_r': m_full, 'without_r': m_ablated}


def compare_models(
    ckpt_transformer: str,
    ckpt_cnn: str,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load two checkpoints and print a side-by-side per-mode comparison table.

    Parameters
    ----------
    ckpt_transformer : str — path to TransformerCWFS checkpoint
    ckpt_cnn         : str — path to CNNCWFS checkpoint
    test_loader      : DataLoader  — must yield raw (unnormalised) batches,
                       i.e. CWFSDataset created with label_stats=None.
                       Each model's own normalisation stats are loaded from
                       its checkpoint.

    Returns
    -------
    dict with keys 'transformer' and 'cnn', each a metrics dict.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    mode_names_by_model = {}
    for name, ckpt_path in [('transformer', ckpt_transformer), ('cnn', ckpt_cnn)]:
        model, meta = load_checkpoint(ckpt_path, device)
        lm = torch.from_numpy(meta['label_mean'])
        ls = torch.from_numpy(meta['label_std'])
        mode_names_by_model[name] = mode_names_from_config(meta['config']['model'])
        # test_loader returns raw physical labels (label_stats=None).
        # _predict must denorm predictions only (labels_are_zscored=False).
        pred, target = _predict(model, test_loader, lm, ls, device,
                                labels_are_zscored=False)
        results[name] = _metrics_table(pred, target)
        _print_table(results[name],
                     header=f"{name.upper()}  (epoch {meta['epoch']}, "
                            f"val WFE {meta['val_wfe_rms']*1e9:.1f} nm)",
                     mode_names=mode_names_by_model[name])

    # Difference table
    print(f"\n{'─'*60}")
    print("Per-mode difference:  CNN − Transformer  (nm, positive = CNN worse)")
    print('─'*60)
    diff_names = mode_names_by_model['transformer'] or mode_names_by_model['cnn'] or \
        [f"mode {i+1}" for i in range(len(results['transformer']['mode_rms_nm']))]
    for name, t_rms, c_rms in zip(
        diff_names,
        results['transformer']['mode_rms_nm'],
        results['cnn']['mode_rms_nm'],
    ):
        diff = c_rms - t_rms
        marker = ' ←' if abs(diff) > 2.0 else ''
        print(f"  {name:<30s}  {diff:+6.2f} nm{marker}")

    return results


def eval_onsky(
    model: nn.Module,
    onsky_loader: DataLoader,
    label_mean: np.ndarray,
    label_std: np.ndarray,
    device: Optional[torch.device] = None,
    fine_tune: bool = False,
    ft_epochs: int = 20,
    ft_lr: float = 1e-4,
    ft_label_mean: Optional[np.ndarray] = None,
    ft_label_std: Optional[np.ndarray] = None,
) -> dict:
    """
    Run the model on on-sky data and optionally fine-tune the regression head.

    On-sky inference
    ----------------
    If fine_tune=False, onsky_loader need not contain labels; the function
    returns predictions only (target metrics will be NaN).

    On-sky fine-tuning protocol
    ---------------------------
    If fine_tune=True:
        1. Freeze all parameters except the MLPHead.
        2. Train MLPHead for ft_epochs on the labelled on-sky calibration
           batches in onsky_loader.
        3. Report per-mode RMS on the fine-tuned model over the same loader.

    The backbone is deliberately kept frozen to prevent catastrophic
    forgetting of the synthetic-data priors.

    Parameters
    ----------
    onsky_loader : DataLoader
        Batches of {'I1', 'I2', 'r', 'labels'} where labels are raw physical
        coefficients (not z-scored).  If fine_tune=False, 'labels' may be
        absent or zero.
    ft_label_mean / ft_label_std : optional ndarray[14]
        Normalisation stats for the on-sky label distribution.  Defaults to
        the synthetic training stats if not provided.

    Returns
    -------
    dict with keys 'pred' (Tensor[N,14]), and optionally 'metrics' (dict).
    """
    if device is None:
        device = next(model.parameters()).device

    lm = torch.from_numpy(label_mean)
    ls = torch.from_numpy(label_std)

    if not fine_tune:
        model.eval()
        pred, target = _predict(model, onsky_loader, lm, ls, device)
        result = {'pred': pred}
        has_labels = not torch.all(target == 0)
        if has_labels:
            result['metrics'] = _metrics_table(pred, target)
            _print_table(result['metrics'], header="On-sky evaluation (no fine-tuning)")
        return result

    # ── fine-tuning: freeze backbone, unfreeze head ─────────────────
    model = _freeze_backbone(model)

    ft_lm = torch.from_numpy(ft_label_mean if ft_label_mean is not None else label_mean)
    ft_ls = torch.from_numpy(ft_label_std  if ft_label_std  is not None else label_std)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ft_lr, weight_decay=1e-2,
    )
    criterion = nn.L1Loss()

    print(f"\nFine-tuning MLPHead for {ft_epochs} epochs (backbone frozen)...")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params")

    ft_lm_dev = ft_lm.to(device)
    ft_ls_dev = ft_ls.to(device)

    for epoch in range(1, ft_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for batch in onsky_loader:
            I1     = batch['I1'].to(device, non_blocking=True)
            I2     = batch['I2'].to(device, non_blocking=True)
            r      = batch['r'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            # z-score labels with on-sky stats before L1 loss
            labels_norm = (labels - ft_lm_dev) / (ft_ls_dev + 1e-8)
            pred = model(I1, I2, r)
            loss = criterion(pred, labels_norm)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if epoch % max(1, ft_epochs // 5) == 0 or epoch == ft_epochs:
            print(f"  epoch {epoch:3d}/{ft_epochs}  loss={epoch_loss/n_batches:.4f}")

    # ── post-fine-tune evaluation ────────────────────────────────────
    pred, target = _predict(model, onsky_loader, ft_lm, ft_ls, device)
    metrics = _metrics_table(pred, target)
    _print_table(metrics, header="On-sky evaluation (after fine-tuning)")

    return {'pred': pred, 'metrics': metrics, 'model': model}


# ─────────────────────────────────────────────────────────────────────
# Backbone freezing helper
# ─────────────────────────────────────────────────────────────────────

def _freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters except the final MLPHead (accessed as model.head).

    Works for both TransformerCWFS and CNNCWFS which both expose a .head
    attribute pointing to the MLPHead regression layer.

    Raises AttributeError if the model does not have a .head attribute.
    """
    if not hasattr(model, 'head'):
        raise AttributeError(
            f"{type(model).__name__} does not expose a .head attribute. "
            "Freeze backbone manually before calling eval_onsky(fine_tune=True)."
        )
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    return model


# ─────────────────────────────────────────────────────────────────────
# Utility: build a test DataLoader from an HDF5 file + checkpoint meta
# ─────────────────────────────────────────────────────────────────────

def make_test_loader(
    hdf5_path: str,
    label_stats: Optional[dict],
    split_ratios: tuple = (0.80, 0.10, 0.10),
    split_seed: int = 42,
    batch_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    """
    Convenience function: build a DataLoader for the held-out test split.

    Parameters
    ----------
    hdf5_path   : str
    label_stats : dict {'mean': ndarray, 'std': ndarray} or None
        Pass the checkpoint meta['label_mean'] / meta['label_std'] here when
        comparing models.  Pass None for the raw-label loader used in compare_models().
    """
    _, _, test_idx = train_val_test_split(hdf5_path, split_ratios, split_seed)
    ds = CWFSDataset(hdf5_path, test_idx, label_stats=label_stats)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate CWFS checkpoints")
    p.add_argument('--ckpt',              default=None, help="Single checkpoint path")
    p.add_argument('--hdf5_path',         default=None, help="HDF5 test dataset")
    p.add_argument('--compare',           action='store_true')
    p.add_argument('--ckpt_transformer',  default=None)
    p.add_argument('--ckpt_cnn',          default=None)
    p.add_argument('--onsky_dir',         default=None, help="Directory of on-sky PSF .npy files")
    p.add_argument('--fine_tune',         action='store_true')
    p.add_argument('--ft_epochs',         type=int, default=20)
    p.add_argument('--ablate_roddier',    action='store_true')
    p.add_argument('--batch_size',        type=int, default=256)
    p.add_argument('--num_workers',       type=int, default=4)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── side-by-side comparison ──────────────────────────────────────
    if args.compare:
        if not (args.ckpt_transformer and args.ckpt_cnn and args.hdf5_path):
            raise ValueError("--compare requires --ckpt_transformer, --ckpt_cnn, --hdf5_path")
        n_modes_hdf5 = get_n_modes(args.hdf5_path)
        for ckpt_name, ckpt_path in [('ckpt_transformer', args.ckpt_transformer),
                                     ('ckpt_cnn', args.ckpt_cnn)]:
            _, _meta = load_checkpoint(ckpt_path, device)
            _tm = trained_modes_from_config(_meta['config']['model'])
            if _tm is not None and max(_tm) > n_modes_hdf5:
                raise ValueError(
                    f"n_modes mismatch: HDF5 has {n_modes_hdf5} label columns but "
                    f"{ckpt_name} model.trained_modes references Z{max(_tm)}."
                )
        # raw-label loader so each model uses its own normalisation stats
        test_loader = make_test_loader(args.hdf5_path, label_stats=None,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers)
        compare_models(args.ckpt_transformer, args.ckpt_cnn, test_loader, device)
        sys.exit(0)

    # ── single checkpoint evaluation ────────────────────────────────
    if not args.ckpt:
        raise ValueError("--ckpt is required")

    model, meta = load_checkpoint(args.ckpt, device)
    lm = meta['label_mean']
    ls = meta['label_std']
    label_stats = {'mean': lm, 'std': ls}

    if args.hdf5_path:
        n_modes_hdf5   = get_n_modes(args.hdf5_path)
        trained_modes_ckpt = trained_modes_from_config(meta['config']['model'])
        if trained_modes_ckpt is not None and max(trained_modes_ckpt) > n_modes_hdf5:
            raise ValueError(
                f"n_modes mismatch: HDF5 has {n_modes_hdf5} label columns but "
                f"checkpoint model.trained_modes references Z{max(trained_modes_ckpt)}.  "
                f"Ensure the checkpoint and dataset were produced with the same n_modes."
            )
        mode_names = mode_names_from_config(meta['config']['model'])
        test_loader = make_test_loader(args.hdf5_path, label_stats=label_stats,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers)
        eval_synthetic(model, test_loader, lm, ls, device, mode_names=mode_names)

        if args.ablate_roddier:
            ablation_roddier(model, test_loader, lm, ls, device, mode_names=mode_names)

    if args.onsky_dir:
        # Minimal on-sky loader: load all PSF pairs from directory as numpy arrays.
        # Expected filename convention:  <stem>_I1.npy, <stem>_I2.npy  (float32, 256×256)
        # Labels file (optional for inference):  <stem>_labels.npy  (float32, 14)
        onsky_path = Path(args.onsky_dir)
        I1_files = sorted(onsky_path.glob('*_I1.npy'))
        if not I1_files:
            raise FileNotFoundError(f"No *_I1.npy files found in {args.onsky_dir}")

        I1_arr, I2_arr, labels_arr = [], [], []
        for f1 in I1_files:
            f2 = f1.with_name(f1.name.replace('_I1.npy', '_I2.npy'))
            fl = f1.with_name(f1.name.replace('_I1.npy', '_labels.npy'))
            i1 = np.load(f1).astype(np.float32)
            i2 = np.load(f2).astype(np.float32)
            I1_arr.append(i1[None])   # add channel dim
            I2_arr.append(i2[None])
            if fl.exists():
                labels_arr.append(np.load(fl).astype(np.float32))
            else:
                labels_arr.append(np.zeros(14, dtype=np.float32))

        I1_t = torch.from_numpy(np.stack(I1_arr))     # [N, 1, H, W]
        I2_t = torch.from_numpy(np.stack(I2_arr))
        r_t  = (I1_t - I2_t) / (I1_t + I2_t + 1e-6)
        L_t  = torch.from_numpy(np.stack(labels_arr))  # [N, 14]

        # z-score labels with synthetic stats (fine-tuning will use on-sky stats)
        L_norm = (L_t - torch.from_numpy(lm)) / (torch.from_numpy(ls) + 1e-8)

        onsky_ds     = TensorDataset(I1_t, I2_t, r_t, L_norm)
        # wrap to match the dict-based API expected by eval_onsky
        class _DictWrapper(torch.utils.data.Dataset):
            def __init__(self, ds): self.ds = ds
            def __len__(self):      return len(self.ds)
            def __getitem__(self, i):
                i1, i2, r, lb = self.ds[i]
                return {'I1': i1, 'I2': i2, 'r': r, 'labels': lb}
        onsky_loader = DataLoader(_DictWrapper(onsky_ds),
                                  batch_size=args.batch_size, shuffle=False)

        eval_onsky(model, onsky_loader, lm, ls, device,
                   fine_tune=args.fine_tune, ft_epochs=args.ft_epochs)

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
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from dataset import CWFSDataset, train_val_test_split, get_n_modes, EPS_RODDIER, apply_detector_noise
from train import build_model, NOLL_MODE_NAMES
from models.cnn_cwfs import RODCNN
from utils.metrics import per_mode_rms, total_wfe_rms, strehl_proxy, format_order_grouped_rms
from utils.sparse_recorder import SparseRecorder
from utils.ensemble import average_predictions
from utils.augmentation import build_d4_label_matrices, apply_d4_tta


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

def load_checkpoint(
    ckpt_path: str,
    device: torch.device,
    use_raw: bool = False,
) -> tuple[nn.Module, dict]:
    """
    Load a model and its training metadata from a checkpoint saved by train.py.

    Parameters
    ----------
    ckpt_path : str
    device    : torch.device
    use_raw   : bool, optional
        If True and 'raw_model_state' exists in the checkpoint (e.g. from SWA),
        loads the raw instantaneous training weights instead of the SWA weights.

    Returns
    -------
    model : nn.Module  — loaded model in eval mode on device
    meta  : dict       — {'label_mean': ndarray, 'label_std': ndarray,
                           'val_wfe_rms': float, 'epoch': int, 'config': dict,
                           'has_swa': bool, 'using_raw': bool}
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(state['config']).to(device)

    using_raw = False
    if use_raw and 'raw_model_state' in state:
        model_state = state['raw_model_state']
        using_raw = True
    else:
        model_state = state['model_state']

    if any(k.startswith('_orig_mod.') for k in model_state.keys()):
        model_state = {k.replace('_orig_mod.', '', 1): v for k, v in model_state.items()}
    model.load_state_dict(model_state)
    model.eval()
    meta = {
        'label_mean':  state['label_mean'],
        'label_std':   state['label_std'],
        'val_wfe_rms': state.get('val_wfe_rms', float('nan')),
        'epoch':       state.get('epoch', -1),
        'config':      state['config'],
        'has_swa':     'swa_state' in state,
        'using_raw':   using_raw,
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
    recorder: Optional[SparseRecorder] = None,
    tta: bool = False,
    trained_modes: Optional[list[int]] = None,
    noise_cfg: Optional[dict] = None,
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
    recorder : SparseRecorder, optional
        Sparse recorder to log batch progress during inference.
    tta : bool, optional
        If True, apply Test-Time Augmentation (TTA) using the D4 dihedral group (8 operations).
    trained_modes : list of int, optional
        Ascending Noll indices corresponding to the model outputs. Required for TTA inversion.
    noise_cfg : dict, optional
        Detector noise configuration dict. If None, checks loader.dataset.noise_cfg.
    """
    all_pred, all_target = [], []
    lm = label_mean.to(device)
    ls = label_std.to(device)
    total_batches = len(loader)

    ds = getattr(loader, 'dataset', None)
    underlying_ds = getattr(ds, 'dataset', ds)
    if noise_cfg is None and underlying_ds is not None:
        noise_cfg = getattr(underlying_ds, 'noise_cfg', None)
    T = getattr(underlying_ds, 'T', 1) if underlying_ds is not None else 1

    raw_model = getattr(model, '_orig_mod', model)
    is_rodcnn = isinstance(raw_model, RODCNN)
    input_mode = getattr(raw_model, 'input_mode', 'two_stream' if is_rodcnn else 'pairs')
    model_takes_r = (not is_rodcnn and input_mode == 'pairs')

    d4_matrices = None
    if tta:
        if trained_modes is None:
            n_out = getattr(model, 'n_outputs', None)
            if n_out is not None:
                trained_modes = list(range(4, 4 + n_out))
        if trained_modes is not None:
            d4_matrices = build_d4_label_matrices(trained_modes)

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            I1     = batch['I1'].to(device, non_blocking=True)
            I2     = batch['I2'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            sample_id = batch.get('sample_id')
            if sample_id is not None and isinstance(sample_id, torch.Tensor):
                sample_id = sample_id.to(device, non_blocking=True)

            if noise_cfg is not None and noise_cfg.get('enabled', False):
                I1 = apply_detector_noise(I1, T=T, noise_cfg=noise_cfg, is_train=False, sample_id=sample_id, stream_id=0)
                I2 = apply_detector_noise(I2, T=T, noise_cfg=noise_cfg, is_train=False, sample_id=sample_id, stream_id=1)

            if tta and d4_matrices is not None:
                if model_takes_r:
                    if noise_cfg is not None and noise_cfg.get('enabled', False):
                        r_batch = (I1 - I2) / (I1 + I2 + EPS_RODDIER)
                    else:
                        r_batch = batch.get('r')
                        if r_batch is not None:
                            r_batch = r_batch.to(device, non_blocking=True)
                else:
                    r_batch = None
                pred_phys = apply_d4_tta(
                    model, I1, I2, d4_matrices,
                    r=r_batch, zero_r=zero_r,
                    label_mean=lm, label_std=ls,
                )
                labels_eff = labels
            elif is_rodcnn:
                pred       = model(I1, I2)                       # [B, n_outputs]
                labels_eff = labels
                pred_phys   = pred * ls + lm
            elif input_mode == 'two_stream':
                pred       = model(I1, I2)                       # [B, n_outputs]
                labels_eff = labels
                pred_phys   = pred * ls + lm
            elif input_mode == 'r_stack':
                if noise_cfg is not None and noise_cfg.get('enabled', False):
                    T_cnt = I1.shape[1]
                    I1_exp = I1.unsqueeze(2)
                    I2_exp = I2.unsqueeze(1)
                    R = (I1_exp - I2_exp) / (I1_exp + I2_exp + EPS_RODDIER)
                    R = R.reshape(I1.shape[0], T_cnt * T_cnt, *R.shape[3:])
                else:
                    R = batch['R'].to(device, non_blocking=True)
                pred       = model(R)                            # [B*T², n_outputs]
                TT = pred.shape[0] // labels.shape[0]
                labels_eff = labels.repeat_interleave(TT, dim=0) # [B*T², n_modes]
                pred_phys   = pred * ls + lm
            else:  # 'pairs' (default)
                if noise_cfg is not None and noise_cfg.get('enabled', False):
                    r = (I1 - I2) / (I1 + I2 + EPS_RODDIER)
                else:
                    r = batch['r'].to(device, non_blocking=True)
                if zero_r:
                    r = torch.zeros_like(r)
                pred       = model(I1, I2, r)                    # [B, n_outputs]
                labels_eff = labels
                pred_phys   = pred * ls + lm

            target_phys = labels_eff * ls + lm if labels_are_zscored else labels_eff

            all_pred.append(pred_phys.cpu())
            all_target.append(target_phys.cpu())

            elapsed = time.time() - t0
            done = batch_idx + 1
            if done % max(1, total_batches // 5) == 0 or done == total_batches:
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_batches - done) / rate if rate > 0 else 0
                print(f"  [eval] batch {done:3d}/{total_batches}  ({done/total_batches*100:5.1f}%)  "
                      f"elapsed={elapsed:5.1f}s  rate={rate:4.2f} batch/s  eta={eta:5.1f}s")
                if recorder is not None:
                    recorder.record_step(
                        step=f"{done}/{total_batches}",
                        metrics={"progress_pct": done / total_batches * 100.0, "rate_batch_per_sec": rate, "eta_s": eta},
                        phase="eval",
                        step_name="batch",
                    )

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


def _print_table(
    metrics: dict,
    header: str = '',
    mode_names: Optional[list] = None,
    trained_modes: Optional[list[int]] = None,
) -> None:
    """Pretty-print a metrics dict to stdout."""
    if header:
        print(f"\n{'─'*60}")
        print(header)
        print('─'*60)
    print(f"  Total WFE rms : {metrics['wfe_rms_nm']:.2f} nm")
    print(f"  Strehl proxy  : {metrics['strehl']:.4f}")

    # Print summary grouped by Zernike order if modes are identifiable
    extracted_modes = trained_modes
    if extracted_modes is None and mode_names:
        try:
            parsed = []
            for name in mode_names:
                m = re.match(r'Z(\d+)', name)
                if m:
                    parsed.append(int(m.group(1)))
                else:
                    parsed = None
                    break
            extracted_modes = parsed
        except Exception:
            extracted_modes = None

    if extracted_modes and len(extracted_modes) == len(metrics['mode_rms_nm']):
        print("  RMS by Zernike order (nm):")
        for line in format_order_grouped_rms(extracted_modes, metrics['mode_rms_nm'], scale=1.0):
            print(line)

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
    recorder: Optional[SparseRecorder] = None,
    tta: bool = False,
    trained_modes: Optional[list[int]] = None,
    noise_cfg: Optional[dict] = None,
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
    recorder    : SparseRecorder, optional — HPC sparse recorder
    tta         : bool, optional — if True, apply D4 dihedral test-time augmentation
    trained_modes : list[int], optional — trained Noll indices for TTA inversion
    noise_cfg   : dict, optional — detector noise configuration dict

    Returns
    -------
    dict with keys 'mode_rms_nm' (list), 'wfe_rms_nm' (float), 'strehl' (float)
    """
    if device is None:
        device = next(model.parameters()).device
    lm = torch.from_numpy(label_mean)
    ls = torch.from_numpy(label_std)
    pred, target = _predict(
        model, test_loader, lm, ls, device,
        recorder=recorder, tta=tta, trained_modes=trained_modes,
        noise_cfg=noise_cfg,
    )
    metrics = _metrics_table(pred, target)
    _print_table(metrics, header="Synthetic test-set evaluation", mode_names=mode_names)

    if recorder is not None:
        recorder.record_step(
            step="final",
            metrics={"wfe_rms_nm": metrics["wfe_rms_nm"], "strehl": metrics["strehl"]},
            phase="test",
            step_name="eval",
        )
        names = mode_names if mode_names is not None else [f"mode {i+1}" for i in range(len(metrics["mode_rms_nm"]))]
        table_rows = [[n, f"{r:.2f}"] for n, r in zip(names, metrics["mode_rms_nm"])]
        recorder.record_table(
            title=f"Synthetic Test Set Evaluation (WFE: {metrics['wfe_rms_nm']:.2f} nm, Strehl: {metrics['strehl']:.4f})",
            headers=["Mode", "RMS Error (nm)"],
            rows=table_rows,
        )

    return metrics


def eval_ensemble(
    ckpt_paths: list[str],
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    mode: str = "average",
    mode_names: Optional[list] = None,
    labels_are_zscored: bool = False,
    recorder: Optional[SparseRecorder] = None,
    tta: bool = False,
    use_raw: bool = False,
    noise_cfg: Optional[dict] = None,
) -> dict:
    """
    Evaluate an ensemble of model checkpoints on the test set.

    Parameters
    ----------
    ckpt_paths  : list[str] — paths to checkpoints
    test_loader : DataLoader yielding {'I1', 'I2', 'r', 'labels'}
    device      : torch.device, optional
    mode        : str — 'average' or 'best'
    mode_names  : list[str], optional — per-mode display names
    labels_are_zscored : bool — True if test_loader batches are pre-zscored
    recorder    : SparseRecorder, optional — HPC sparse recorder
    tta         : bool, optional — if True, apply D4 dihedral test-time augmentation
    use_raw     : bool, optional — if True, load raw instantaneous weights bypassing SWA
    noise_cfg   : dict, optional — detector noise configuration dict

    Returns
    -------
    dict with metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not ckpt_paths:
        raise ValueError("eval_ensemble requires at least one checkpoint path.")

    models_and_meta = []
    for p in ckpt_paths:
        m, meta = load_checkpoint(p, device, use_raw=use_raw)
        models_and_meta.append((p, m, meta))

    if mode == "best":
        best_ckpt, best_model, best_meta = min(
            models_and_meta,
            key=lambda item: item[2].get('val_wfe_rms', float('inf'))
        )
        print(f"\n[Ensemble: best] Selected model {Path(best_ckpt).name} "
              f"with val WFE {best_meta.get('val_wfe_rms', float('nan'))*1e9:.2f} nm")
        lm = torch.from_numpy(best_meta['label_mean'])
        ls = torch.from_numpy(best_meta['label_std'])
        resolved_mode_names = mode_names or mode_names_from_config(best_meta['config']['model'])
        tm = trained_modes_from_config(best_meta['config']['model'])
        pred, target = _predict(
            best_model, test_loader, lm, ls, device,
            labels_are_zscored=labels_are_zscored, recorder=recorder,
            tta=tta, trained_modes=tm,
            noise_cfg=noise_cfg,
        )
        metrics = _metrics_table(pred, target)
        _print_table(metrics, header=f"Ensemble Test Set Evaluation: Best Model ({Path(best_ckpt).name})",
                     mode_names=resolved_mode_names)
        return metrics

    elif mode == "average":
        all_preds = []
        target_ref = None
        indiv_results = []
        resolved_mode_names = None

        for idx, (p, m, meta) in enumerate(models_and_meta, 1):
            lm = torch.from_numpy(meta['label_mean'])
            ls = torch.from_numpy(meta['label_std'])
            if resolved_mode_names is None:
                resolved_mode_names = mode_names or mode_names_from_config(meta['config']['model'])
            tm = trained_modes_from_config(meta['config']['model'])
            print(f"  [Ensemble {idx}/{len(models_and_meta)}] Running inference for {Path(p).name}...")
            pred, target_ref = _predict(
                m, test_loader, lm, ls, device,
                labels_are_zscored=labels_are_zscored,
                tta=tta, trained_modes=tm,
                noise_cfg=noise_cfg,
            )

            all_preds.append(pred)
            m_metrics = _metrics_table(pred, target_ref)
            indiv_results.append((Path(p).name, m_metrics))

        ens_pred = average_predictions(all_preds)
        metrics = _metrics_table(ens_pred, target_ref)
        _print_table(metrics, header=f"Ensemble Test Set Evaluation: Average of {len(models_and_meta)} models",
                     mode_names=resolved_mode_names)

        # Print comparison table
        print(f"\n{'─'*60}")
        print("Ensemble Models Comparison:")
        print(f"  {'Model Checkpoint':<35s} {'Test WFE (nm)':<15s} {'Strehl':<10s}")
        print(f"  {'-'*35} {'-'*15} {'-'*10}")
        rows = []
        for name, m_res in indiv_results:
            print(f"  {name:<35s} {m_res['wfe_rms_nm']:<15.2f} {m_res['strehl']:<10.4f}")
            rows.append([name, f"{m_res['wfe_rms_nm']:.2f}", f"{m_res['strehl']:.4f}"])
        print(f"  {'Ensemble Average':<35s} {metrics['wfe_rms_nm']:<15.2f} {metrics['strehl']:<10.4f}")
        rows.append(["Ensemble Average", f"{metrics['wfe_rms_nm']:.2f}", f"{metrics['strehl']:.4f}"])
        print(f"{'─'*60}\n")

        if recorder is not None:
            recorder.record_table(
                title=f"Ensemble Test Evaluation ({len(models_and_meta)} models)",
                headers=["Checkpoint", "Test WFE (nm)", "Strehl"],
                rows=rows,
            )
        return metrics

    else:
        raise ValueError(f"Unknown ensemble mode '{mode}'. Must be 'average' or 'best'.")


def ablation_roddier(
    model: nn.Module,
    test_loader: DataLoader,
    label_mean: np.ndarray,
    label_std: np.ndarray,
    device: Optional[torch.device] = None,
    mode_names: Optional[list] = None,
    recorder: Optional[SparseRecorder] = None,
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

    if recorder is not None:
        names = mode_names if mode_names is not None else [f"mode {i+1}" for i in range(len(m_full["mode_rms_nm"]))]
        rows = [
            [n, f"{r_f:.2f}", f"{r_a:.2f}", f"{r_a - r_f:+.2f}"]
            for n, r_f, r_a in zip(names, m_full["mode_rms_nm"], m_ablated["mode_rms_nm"])
        ]
        recorder.record_table(
            title=f"Roddier Ablation (with r: {m_full['wfe_rms_nm']:.2f} nm, without r: {m_ablated['wfe_rms_nm']:.2f} nm, delta: +{delta_wfe:.2f} nm)",
            headers=["Mode", "With r (nm)", "Without r (nm)", "Delta (nm)"],
            rows=rows,
        )

    return {'with_r': m_full, 'without_r': m_ablated}


def compare_models(
    ckpt_transformer: str,
    ckpt_cnn: str,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    recorder: Optional[SparseRecorder] = None,
    tta: bool = False,
    use_raw: bool = False,
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
    recorder         : SparseRecorder, optional
    tta              : bool, optional — if True, apply D4 test-time augmentation
    use_raw          : bool, optional — if True, load raw weights bypassing SWA

    Returns
    -------
    dict with keys 'transformer' and 'cnn', each a metrics dict.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    mode_names_by_model = {}
    for name, ckpt_path in [('transformer', ckpt_transformer), ('cnn', ckpt_cnn)]:
        model, meta = load_checkpoint(ckpt_path, device, use_raw=use_raw)
        lm = torch.from_numpy(meta['label_mean'])
        ls = torch.from_numpy(meta['label_std'])
        tm = trained_modes_from_config(meta['config']['model'])
        mode_names_by_model[name] = mode_names_from_config(meta['config']['model'])
        # test_loader returns raw physical labels (label_stats=None).
        # _predict must denorm predictions only (labels_are_zscored=False).
        pred, target = _predict(
            model, test_loader, lm, ls, device,
            labels_are_zscored=False, tta=tta, trained_modes=tm,
        )
        results[name] = _metrics_table(pred, target)
        raw_tag = " [RAW]" if meta.get('using_raw') else (" [SWA]" if meta.get('has_swa') else "")
        _print_table(results[name],
                     header=f"{name.upper()}{raw_tag}  (epoch {meta['epoch']}, "
                            f"val WFE {meta['val_wfe_rms']*1e9:.1f} nm)",
                     mode_names=mode_names_by_model[name])

    # Difference table
    print(f"\n{'─'*60}")
    print("Per-mode difference:  CNN − Transformer  (nm, positive = CNN worse)")
    print('─'*60)
    diff_names = mode_names_by_model['transformer'] or mode_names_by_model['cnn'] or \
        [f"mode {i+1}" for i in range(len(results['transformer']['mode_rms_nm']))]
    diff_rows = []
    for name, t_rms, c_rms in zip(
        diff_names,
        results['transformer']['mode_rms_nm'],
        results['cnn']['mode_rms_nm'],
    ):
        diff = c_rms - t_rms
        marker = ' ←' if abs(diff) > 2.0 else ''
        print(f"  {name:<30s}  {diff:+6.2f} nm{marker}")
        diff_rows.append([name, f"{t_rms:.2f}", f"{c_rms:.2f}", f"{diff:+6.2f}"])

    if recorder is not None:
        recorder.record_table(
            title=f"Model Comparison: CNN ({results['cnn']['wfe_rms_nm']:.2f} nm) vs Transformer ({results['transformer']['wfe_rms_nm']:.2f} nm)",
            headers=["Mode", "Transformer (nm)", "CNN (nm)", "Diff (CNN-Trans)"],
            rows=diff_rows,
        )

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
    recorder: Optional[SparseRecorder] = None,
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
    recorder : SparseRecorder, optional

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
            if recorder is not None:
                recorder.record_step(
                    step="onsky_eval",
                    metrics={"wfe_rms_nm": result['metrics']["wfe_rms_nm"], "strehl": result['metrics']["strehl"]},
                    phase="onsky",
                )
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
    if recorder is not None:
        recorder.log_message(f"Fine-tuning MLPHead for {ft_epochs} epochs (trainable {trainable:,}/{total:,} params)")

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
            avg_l = epoch_loss / n_batches
            print(f"  epoch {epoch:3d}/{ft_epochs}  loss={avg_l:.4f}")
            if recorder is not None:
                recorder.record_step(
                    step=f"{epoch}/{ft_epochs}",
                    metrics={"loss": avg_l},
                    phase="finetune",
                    step_name="epoch",
                )

    # ── post-fine-tune evaluation ────────────────────────────────────
    pred, target = _predict(model, onsky_loader, ft_lm, ft_ls, device)
    metrics = _metrics_table(pred, target)
    _print_table(metrics, header="On-sky evaluation (after fine-tuning)")

    if recorder is not None:
        recorder.record_step(
            step="post_finetune",
            metrics={"wfe_rms_nm": metrics["wfe_rms_nm"], "strehl": metrics["strehl"]},
            phase="onsky",
        )

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
    mode_columns: Optional[list] = None,
    return_stacks: bool = False,
    noise_cfg: Optional[dict] = None,
) -> DataLoader:
    """
    Convenience function: build a DataLoader for the held-out test split.

    Parameters
    ----------
    hdf5_path     : str
    label_stats   : dict {'mean': ndarray, 'std': ndarray} or None
        Pass the checkpoint meta['label_mean'] / meta['label_std'] here when
        comparing models.  Pass None for the raw-label loader used in compare_models().
    mode_columns  : list of int, optional
        0-based label column indices to select.
    return_stacks : bool
        If True, return temporal frame stacks per example (needed for RODCNN).
    noise_cfg     : dict or None, optional
        Detector noise configuration dict.
    """
    _, _, test_idx = train_val_test_split(hdf5_path, split_ratios, split_seed)
    ds = CWFSDataset(
        hdf5_path, test_idx,
        label_stats=label_stats,
        mode_columns=mode_columns,
        return_stacks=return_stacks,
        noise_cfg=noise_cfg,
        is_train=False,
    )
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
    p.add_argument('--ensemble_dir',     default=None, help="Directory containing ensemble checkpoints or ensemble_summary.yaml")
    p.add_argument('--ckpts',            nargs='+', default=None, help="List of checkpoint paths to evaluate as an ensemble")
    p.add_argument('--ensemble_mode',    choices=['average', 'best'], default='average', help="Ensemble evaluation mode ('average' or 'best')")
    p.add_argument('--sparse_dir',        default=None, help="Directory for sparse HPC logs (defaults to nn_WFS/tmp)")
    p.add_argument('--no_sparse_record',  action='store_true', help="Disable sparse HPC log recording")
    p.add_argument('--tta', dest='tta',   action='store_true', default=False, help="Enable D4 test-time augmentation (8x dihedral ensemble)")
    p.add_argument('--no_tta', dest='tta', action='store_false', help="Disable D4 test-time augmentation")
    p.add_argument('--raw_weights',       action='store_true', default=False, help="Load raw instantaneous training weights instead of SWA averaged weights")
    p.add_argument('--noise_enabled',     dest='noise_enabled', action='store_true', default=None, help="Enable on-the-fly detector noise injection during evaluation")
    p.add_argument('--no_noise',          dest='noise_enabled', action='store_false', help="Disable detector noise injection")
    p.add_argument('--photons_per_frame', type=float, default=None, help="Incident photons per frame (e.g. 1e5)")
    p.add_argument('--photons_per_image', type=float, default=None, help="Alias for --photons_per_frame")
    p.add_argument('--read_noise_e',      type=float, default=None, help="Gaussian readout noise std in RMS electrons (e.g. 2.0)")
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    noise_cfg = None
    if getattr(args, 'noise_enabled', None) is not None or getattr(args, 'photons_per_frame', None) is not None or getattr(args, 'photons_per_image', None) is not None or getattr(args, 'read_noise_e', None) is not None:
        n_ph = args.photons_per_frame if args.photons_per_frame is not None else (args.photons_per_image if args.photons_per_image is not None else 1e6)
        noise_cfg = {
            'enabled': bool(args.noise_enabled) if args.noise_enabled is not None else True,
            'photons_per_frame': float(n_ph),
            'photons_per_image': float(n_ph),
            'read_noise_e': float(args.read_noise_e) if args.read_noise_e is not None else 0.0,
            'seed': 42,
        }
        if noise_cfg['enabled']:
            print(f"Evaluation Noise Injection: ENABLED (N_ph: {noise_cfg['photons_per_frame']:.1e} photons/frame, read_noise: {noise_cfg['read_noise_e']:.1f} e-)")
        else:
            print("Evaluation Noise Injection: DISABLED (clean data)")

    recorder = None
    if not args.no_sparse_record:
        task_name = "eval_ensemble" if (args.ckpts or args.ensemble_dir) else ("eval_compare" if args.compare else ("eval_onsky" if args.onsky_dir else "eval_synthetic"))
        recorder = SparseRecorder(
            task_name=task_name,
            output_dir=args.sparse_dir,
            config=vars(args),
        )
        print(f"HPC Sparse Recording initialized: {recorder.filepath}")

    try:
        # ── ensemble evaluation ──────────────────────────────────────────
        if args.ckpts or args.ensemble_dir:
            if not args.hdf5_path:
                raise ValueError("--ensemble_dir / --ckpts requires --hdf5_path")

            ckpt_paths = []
            if args.ckpts:
                ckpt_paths = [str(Path(p).resolve()) for p in args.ckpts]
            elif args.ensemble_dir:
                ens_dir = Path(args.ensemble_dir)
                summary_yaml = ens_dir / "ensemble_summary.yaml"
                if summary_yaml.exists():
                    import yaml
                    with open(summary_yaml) as f:
                        summary_data = yaml.safe_load(f)
                    for r in summary_data.get('runs', []):
                        if r.get('checkpoint') and Path(r['checkpoint']).exists():
                            ckpt_paths.append(r['checkpoint'])
                if not ckpt_paths:
                    seed_dirs = sorted([d for d in ens_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
                    if seed_dirs:
                        for sd in seed_dirs:
                            pts = sorted(sd.glob("*.pt"))
                            if pts:
                                ckpt_paths.append(str(pts[-1]))
                    else:
                        ckpt_paths = [str(p) for p in sorted(ens_dir.glob("*.pt")) if "best_ensemble" not in p.name]

            if not ckpt_paths:
                raise FileNotFoundError(f"No valid checkpoints found in {args.ensemble_dir or args.ckpts}")

            print(f"Found {len(ckpt_paths)} ensemble checkpoint(s):")
            for cp in ckpt_paths:
                print(f"  - {cp}")

            _, first_meta = load_checkpoint(ckpt_paths[0], device, use_raw=args.raw_weights)
            first_mc = first_meta['config']['model']
            is_rodcnn = first_mc['type'].lower() == 'rodcnn'
            n_modes_hdf5 = get_n_modes(args.hdf5_path)
            tm = trained_modes_from_config(first_mc)
            mode_cols = [m - 1 for m in tm] if (tm is not None and len(tm) < n_modes_hdf5) else None

            # test loader with label_stats=None so each model denormalizes with its own stats
            test_loader = make_test_loader(args.hdf5_path, label_stats=None,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           mode_columns=mode_cols,
                                           return_stacks=is_rodcnn,
                                           noise_cfg=noise_cfg)
            mode_names = mode_names_from_config(first_mc)
            eval_ensemble(
                ckpt_paths, test_loader, device, mode=args.ensemble_mode,
                mode_names=mode_names, labels_are_zscored=False, recorder=recorder,
                tta=args.tta, use_raw=args.raw_weights,
                noise_cfg=noise_cfg,
            )
            if recorder is not None:
                recorder.close(status="COMPLETED")
            sys.exit(0)

        # ── side-by-side comparison ──────────────────────────────────────
        if args.compare:
            if not (args.ckpt_transformer and args.ckpt_cnn and args.hdf5_path):
                raise ValueError("--compare requires --ckpt_transformer, --ckpt_cnn, --hdf5_path")
            n_modes_hdf5 = get_n_modes(args.hdf5_path)
            for ckpt_name, ckpt_path in [('ckpt_transformer', args.ckpt_transformer),
                                         ('ckpt_cnn', args.ckpt_cnn)]:
                _, _meta = load_checkpoint(ckpt_path, device, use_raw=args.raw_weights)
                _tm = trained_modes_from_config(_meta['config']['model'])
                if _tm is not None and max(_tm) > n_modes_hdf5:
                    raise ValueError(
                        f"n_modes mismatch: HDF5 has {n_modes_hdf5} label columns but "
                        f"{ckpt_name} model.trained_modes references Z{max(_tm)}."
                    )
            # raw-label loader so each model uses its own normalisation stats
            test_loader = make_test_loader(args.hdf5_path, label_stats=None,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           noise_cfg=noise_cfg)
            compare_models(
                args.ckpt_transformer, args.ckpt_cnn, test_loader, device,
                recorder=recorder, tta=args.tta, use_raw=args.raw_weights,
            )
            if recorder is not None:
                recorder.close(status="COMPLETED")
            sys.exit(0)

        # ── single checkpoint evaluation ────────────────────────────────
        if not args.ckpt:
            raise ValueError("--ckpt is required (or specify --ensemble_dir / --ckpts)")

        model, meta = load_checkpoint(args.ckpt, device, use_raw=args.raw_weights)
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
            mode_columns = [m - 1 for m in trained_modes_ckpt] if (trained_modes_ckpt is not None and len(trained_modes_ckpt) < n_modes_hdf5) else None
            is_rodcnn = isinstance(model, RODCNN)
            mode_names = mode_names_from_config(meta['config']['model'])
            test_loader = make_test_loader(args.hdf5_path, label_stats=label_stats,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           mode_columns=mode_columns,
                                           return_stacks=is_rodcnn,
                                           noise_cfg=noise_cfg)
            lm_eval = lm[mode_columns] if mode_columns is not None else lm
            ls_eval = ls[mode_columns] if mode_columns is not None else ls
            eval_synthetic(
                model, test_loader, lm_eval, ls_eval, device,
                mode_names=mode_names, recorder=recorder,
                tta=args.tta, trained_modes=trained_modes_ckpt,
                noise_cfg=noise_cfg,
            )

            if args.ablate_roddier:
                ablation_roddier(model, test_loader, lm_eval, ls_eval, device, mode_names=mode_names, recorder=recorder)

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
                       fine_tune=args.fine_tune, ft_epochs=args.ft_epochs,
                       recorder=recorder)

        if recorder is not None:
            recorder.close(status="COMPLETED")

    except Exception as e:
        if recorder is not None:
            recorder.log_message(f"[ERROR] Run failed with exception: {e}")
            recorder.close(status=f"FAILED ({type(e).__name__})")
        raise


from typing import Optional, cast, Union, Tuple, List
import math
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

# Roddier signal stabilisation constant
EPS_RODDIER = 1e-6


def apply_detector_noise(
    I: torch.Tensor,
    T: Optional[int] = None,
    noise_cfg: Optional[dict] = None,
    is_train: bool = True,
    sample_id: Optional[Union[int, torch.Tensor, list, np.integer]] = None,
    stream_id: int = 0,
) -> torch.Tensor:
    """
    Inject Poisson photon shot noise and Gaussian readout noise on-the-fly.
    Supports execution on CPU and GPU (CUDA) devices, for single images or batched tensors.

    Parameters
    ----------
    I : torch.Tensor
        PSF tensor of shape [1, H, W], [T, H, W], [B, 1, H, W], or [B, T, H, W].
        In the consolidated dataset, sum across all T frames is ~1.0 (each frame sum ~ 1/T).
    T : int or None
        Number of frames in sequence. If None, defaults to 1.
    noise_cfg : dict or None
        Configuration dict with keys:
          'enabled' (bool): whether noise injection is active
          'photons_per_frame' (float, alias 'photons_per_image'): N_ph per 2D frame
          'read_noise_e' (float): Gaussian read noise std in electrons (default 0.0)
          'seed' (int): base seed for deterministic val/test noise
    is_train : bool
        If True, samples stochastic noise independently across calls on the tensor's device.
        If False, uses deterministic generator seeded with (base_seed, sample_id, stream_id).
    sample_id : int, Tensor, list, or None
        Unique sample index (or batch of indices) for deterministic seeding in eval mode.
    stream_id : int
        0 for I1 (intra), 1 for I2 (extra), ensures distinct noise realizations.

    Returns
    -------
    torch.Tensor with same shape, dtype, and device as I.
    """
    if noise_cfg is None or not noise_cfg.get('enabled', False):
        return I

    n_ph = float(noise_cfg.get('photons_per_frame', noise_cfg.get('photons_per_image', 1e6)))
    if n_ph <= 0 or math.isinf(n_ph):
        return I

    read_noise_e = float(noise_cfg.get('read_noise_e', 0.0))
    base_seed = int(noise_cfg.get('seed', 42))

    orig_dtype = I.dtype
    orig_device = I.device
    I_float = I.to(torch.float32)

    # Scale frame to unit total flux: each frame sums to ~1.0
    scale_to_unit = float(max(1, T)) if T is not None else 1.0
    I_unit = I_float * scale_to_unit

    # Expected photon count per pixel
    mu = torch.clamp(I_unit * n_ph, min=0.0)

    if is_train:
        # Stochastic training noise on tensor device (GPU / CPU)
        noisy_counts = torch.poisson(mu)
        if read_noise_e > 0.0:
            noisy_counts = noisy_counts + torch.randn_like(mu) * read_noise_e
    else:
        # Deterministic validation / test noise
        # Batched sample_ids matching leading dimension
        is_batched = (
            sample_id is not None
            and mu.dim() >= 2
            and (
                (isinstance(sample_id, torch.Tensor) and sample_id.dim() >= 1 and sample_id.shape[0] == mu.shape[0])
                or (not isinstance(sample_id, torch.Tensor) and hasattr(sample_id, '__len__') and len(sample_id) == mu.shape[0])
            )
        )
        if is_batched:
            noisy_counts = torch.empty_like(mu)
            has_read = (read_noise_e > 0.0)
            sample_ids_list = sample_id.cpu().tolist() if isinstance(sample_id, torch.Tensor) else list(sample_id)
            for b, s_id in enumerate(sample_ids_list):
                det_seed = int((base_seed * 1000003 + int(s_id) * 7919 + stream_id * 31 + 17) & 0x7FFFFFFF)
                gen = torch.Generator(device=orig_device).manual_seed(det_seed)
                noisy_counts[b] = torch.poisson(mu[b], generator=gen)
                if has_read:
                    noisy_counts[b] += torch.randn(mu[b].shape, generator=gen, dtype=torch.float32, device=orig_device) * read_noise_e
        else:
            if sample_id is None:
                s_id = 0
            elif isinstance(sample_id, torch.Tensor):
                s_id = int(sample_id.item())
            else:
                s_id = int(sample_id)
            det_seed = int((base_seed * 1000003 + s_id * 7919 + stream_id * 31 + 17) & 0x7FFFFFFF)
            gen = torch.Generator(device=orig_device).manual_seed(det_seed)
            noisy_counts = torch.poisson(mu, generator=gen)
            if read_noise_e > 0.0:
                noise_read = torch.randn(mu.shape, generator=gen, dtype=torch.float32, device=orig_device) * read_noise_e
                noisy_counts = noisy_counts + noise_read

    # Clamp to non-negative and scale back to original dataset intensity scale
    noisy_counts = torch.clamp(noisy_counts, min=0.0)
    noisy_unit = noisy_counts / n_ph
    noisy_I = noisy_unit / scale_to_unit

    return noisy_I.to(orig_dtype)


class CWFSDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for Curvature WFS training data stored in HDF5.

    Supported HDF5 schemas
    ----------------------
    Temporal (5-D, written by make_training_data.py):
        psfs   : float16  [N, 2, T, H, W]
                    channel 0 = I1 (intra-focal, T frames)
                    channel 1 = I2 (extra-focal, T frames)
        labels : float32  [N, n_modes]

    Legacy (4-D):
        psfs   : float16  [N, 2, H, W]  — channel 0 = I1, channel 1 = I2
        labels : float32  [N, n_modes]

    Temporal frame-pair expansion
    -----------------------------
    Because I1 and I2 are temporally incoherent, every combination of one
    I1 frame with one I2 frame is a valid, independent training sample for the
    same Zernike label.  For T frames per stream, each HDF5 example contributes
    T² dataset items.  __len__ therefore returns len(indices) * T * T, and
    __getitem__ maps a flat index k to (example, frame_i, frame_j):

        example = k // (T * T)
        frame_i = (k //  T   ) % T   ← which I1 frame
        frame_j =  k           % T   ← which I2 frame

    The train/val/test split is performed at the example level (by
    train_val_test_split), so all T² pairs from a given example belong to
    exactly one split — no data leakage.

    The Roddier signal r = (I1 - I2) / (I1 + I2 + eps) is computed
    on-the-fly from the selected frame pair.

    Parameters
    ----------
    hdf5_path : str
    indices : array-like of int
        HDF5 example indices (example-level, not item-level).
        Use train_val_test_split() to generate these.
    label_stats : dict or None
        Optional {'mean': ndarray, 'std': ndarray} for z-score normalisation.
    transform : callable or None
        Applied to the sample dict after construction.  Supports both
        return_stacks=False (keys 'I1','I2','r') and return_stacks=True
        (keys 'I1','I2','R') sample shapes.
    return_stacks : bool
        If False (default): T² item expansion — each example yields T² items,
        each item returns {I1:[1,H,W], I2:[1,H,W], r:[1,H,W], labels}.
        If True: 1 item per example, returns
        {I1:[T,H,W], I2:[T,H,W], labels} (and optionally 'R':[T²,H,W] if
        compute_r_stack=True).
        Required for input_mode='two_stream', 'r_stack', or 'rodcnn'.
    mode_columns : array-like of int or None
        If set, 0-based HDF5 label-column indices to select (subset mode
        training), in the given order — e.g. [4,5,6,7,8,9,10,11,12,13,14]
        selects Noll modes Z5..Z15.  Selection occurs after z-score
        normalisation but before the augmentation transform, ensuring
        D4Augment operates on correctly-sized, correctly-ordered labels.
    compute_r_stack : bool
        If True and return_stacks is True, compute all T² Roddier combinations
        R = (I1 - I2) / (I1 + I2 + eps) on the CPU and include 'R' in the sample dict.
        Default False (avoids ~17MB CPU allocation/IPC transfer when models compute
        Roddier signals on GPU or only consume raw I1/I2 streams).
    preload : bool
        If True, load all psfs and labels for this split into RAM-resident numpy
        arrays at __init__ time (one-time ~8.8 s sequential HDF5 read for 1500
        examples).  Subsequent __getitem__ calls read from RAM (~3.7 ms each)
        instead of from HDF5 (~347 ms each under the current gzip chunk=64 layout).
        With num_workers>0, forked worker processes inherit the array via Linux
        copy-on-write — the array is read-only so pages are never duplicated.
        Default False (lazy HDF5 mode, appropriate when RAM is limited or the
        HDF5 file has already been re-chunked to chunk=1).
    """

    def __init__(self, hdf5_path, indices, label_stats=None, transform=None,
                 return_stacks=False, mode_columns=None, compute_r_stack=False,
                 preload=False, label_scale=1.0, noise_cfg=None, is_train=True):
        self.path = str(hdf5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_stats = label_stats
        self.transform = transform
        self.return_stacks = return_stacks
        self.compute_r_stack = compute_r_stack
        self.label_scale = float(label_scale)
        self.noise_cfg = noise_cfg
        self.is_train = bool(is_train)
        # if set, select (and reorder) these 0-based label columns before augmentation
        self.mode_idx = torch.as_tensor(mode_columns, dtype=torch.long) if mode_columns is not None else None
        self._file = None          # opened lazily; one handle per DataLoader worker
        self._psfs_ds: Optional[h5py.Dataset] = None
        self._labels_ds: Optional[h5py.Dataset] = None

        # Detect schema and store T (frames per stream).
        with h5py.File(self.path, 'r') as f:
            shape = f['psfs'].shape   # (N, 2, T, H, W) or (N, 2, H, W)
        self._temporal = (len(shape) == 5)
        self.T = int(shape[2]) if self._temporal else 1

        # ── Optional preload into RAM (for small datasets) ─────────────
        self._psfs_mem   = None   # float16 ndarray [N, 2, T, H, W] or [N, 2, H, W]
        self._labels_mem = None   # float32 ndarray [N, n_modes]
        if preload:
            N = len(self.indices)
            sort_perm   = np.argsort(self.indices)          # local positions sorted by HDF5 row
            inv_perm    = np.argsort(sort_perm)             # inverse: restores original order
            sorted_rows = self.indices[sort_perm]           # global HDF5 rows, ascending

            print(f"  Preloading {N} examples into RAM (path={self.path})…", flush=True)
            with h5py.File(self.path, 'r') as f:
                psfs_ds   = f['psfs']
                labels_ds = f['labels']
                full_n    = psfs_ds.shape[0]
                chunk_n   = psfs_ds.chunks[0] if psfs_ds.chunks is not None else 1
                psf_shape = psfs_ds.shape[1:]
                n_modes   = labels_ds.shape[1]

                self._psfs_mem   = np.empty((N, *psf_shape), dtype='float16')
                self._labels_mem = np.empty((N, n_modes),    dtype='float32')

                for cs in range(0, full_n, chunk_n):
                    ce   = min(cs + chunk_n, full_n)
                    mask = (sorted_rows >= cs) & (sorted_rows < ce)
                    if not mask.any():
                        continue
                    chunk      = psfs_ds[cs:ce]           # contiguous slice read (fast)
                    lbl_chunk  = labels_ds[cs:ce]
                    local_rows = sorted_rows[mask] - cs
                    dest       = inv_perm[np.where(mask)[0]]
                    self._psfs_mem[dest]   = chunk[local_rows]
                    self._labels_mem[dest] = lbl_chunk[local_rows]

            gb = self._psfs_mem.nbytes / 1e9
            print(f"  Preload complete — {gb:.3f} GB float16 in RAM.", flush=True)

    # ------------------------------------------------------------------
    # pickling: drop the open file handle so forked workers open fresh;
    # _psfs_mem / _labels_mem are plain numpy arrays — safe to fork/COW.
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None
        state['_psfs_ds'] = None
        state['_labels_ds'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'noise_cfg' not in self.__dict__:
            self.noise_cfg = None
        if 'is_train' not in self.__dict__:
            self.is_train = False

    def __len__(self):
        if self.return_stacks:
            return len(self.indices)
        return len(self.indices) * self.T * self.T

    def __getitem__(self, idx):
        noise_cfg = getattr(self, 'noise_cfg', None)
        is_train = getattr(self, 'is_train', False)

        # ── Preloaded path: read from RAM, skip HDF5 entirely ──────────
        if self._psfs_mem is not None:
            if self.return_stacks:
                if self._temporal:
                    I1 = torch.from_numpy(self._psfs_mem[idx, 0].astype(np.float32))  # [T,H,W]
                    I2 = torch.from_numpy(self._psfs_mem[idx, 1].astype(np.float32))  # [T,H,W]
                else:
                    I1 = torch.from_numpy(self._psfs_mem[idx, 0].astype(np.float32)).unsqueeze(0)
                    I2 = torch.from_numpy(self._psfs_mem[idx, 1].astype(np.float32)).unsqueeze(0)

                sample_id = int(self.indices[idx])
                sample = {'I1': I1, 'I2': I2, 'sample_id': sample_id}

                if self.compute_r_stack:
                    T = I1.shape[0]
                    I1_exp = I1.unsqueeze(1)
                    I2_exp = I2.unsqueeze(0)
                    R = (I1_exp - I2_exp) / (I1_exp + I2_exp + EPS_RODDIER)
                    sample['R'] = R.reshape(T * T, *R.shape[2:])

                labels = torch.from_numpy(self._labels_mem[idx].astype(np.float32))
            else:
                # pair-expansion mode
                if self._temporal:
                    T = self.T
                    example_idx = idx // (T * T)
                    frame_i     = (idx // T) % T
                    frame_j     = idx % T
                    I1 = torch.from_numpy(
                        self._psfs_mem[example_idx, 0, frame_i].astype(np.float32)
                    ).unsqueeze(0)                                             # [1, H, W]
                    I2 = torch.from_numpy(
                        self._psfs_mem[example_idx, 1, frame_j].astype(np.float32)
                    ).unsqueeze(0)                                             # [1, H, W]
                    sample_id = int(self.indices[example_idx]) * (T * T) + (frame_i * T + frame_j)
                else:
                    example_idx = idx
                    I1 = torch.from_numpy(self._psfs_mem[example_idx, 0].astype(np.float32)).unsqueeze(0)
                    I2 = torch.from_numpy(self._psfs_mem[example_idx, 1].astype(np.float32)).unsqueeze(0)
                    sample_id = int(self.indices[example_idx])

                r = (I1 - I2) / (I1 + I2 + EPS_RODDIER)
                labels = torch.from_numpy(self._labels_mem[example_idx].astype(np.float32)) * self.label_scale
                sample = {'I1': I1, 'I2': I2, 'r': r, 'sample_id': sample_id}

            # shared label normalisation + mode selection
            if self.label_stats is not None:
                mean = torch.as_tensor(self.label_stats['mean'], dtype=torch.float32)
                std  = torch.as_tensor(self.label_stats['std'],  dtype=torch.float32)
                labels = (labels - mean) / (std + 1e-8)
            if self.mode_idx is not None:
                labels = labels[self.mode_idx]
            sample['labels'] = labels
            if self.transform is not None:
                sample = self.transform(sample)
            return sample

        # ── Lazy HDF5 path (preload=False) ─────────────────────────────
        if self._file is None:
            self._file = h5py.File(self.path, 'r')
            self._psfs_ds = cast(h5py.Dataset, self._file['psfs'])
            self._labels_ds = cast(h5py.Dataset, self._file['labels'])
        assert self._psfs_ds is not None
        assert self._labels_ds is not None

        if self.return_stacks:
            # ── stack mode: one item per example, returns all T frames ──
            i = int(self.indices[idx])
            psf_pair = self._psfs_ds[i].astype(np.float32)  # [2, T, H, W] or [2, H, W]
            if self._temporal:
                I1 = torch.from_numpy(psf_pair[0])                   # [T, H, W]
                I2 = torch.from_numpy(psf_pair[1])                   # [T, H, W]
            else:
                I1 = torch.from_numpy(psf_pair[0]).unsqueeze(0)      # [1, H, W]
                I2 = torch.from_numpy(psf_pair[1]).unsqueeze(0)      # [1, H, W]

            sample_id = i
            sample = {'I1': I1, 'I2': I2, 'sample_id': sample_id}

            # Optionally compute all T² Roddier combinations via broadcasting on CPU.
            if self.compute_r_stack:
                T = I1.shape[0]
                I1_exp = I1.unsqueeze(1)   # [T, 1, H, W]
                I2_exp = I2.unsqueeze(0)   # [1, T, H, W]
                R = (I1_exp - I2_exp) / (I1_exp + I2_exp + EPS_RODDIER)  # [T, T, H, W]
                R = R.reshape(T * T, *R.shape[2:])                        # [T², H, W]
                sample['R'] = R

            raw_labels = self._labels_ds[i]
            labels = torch.from_numpy(raw_labels.astype(np.float32)) * self.label_scale
            if self.label_stats is not None:
                mean = torch.as_tensor(self.label_stats['mean'], dtype=torch.float32)
                std  = torch.as_tensor(self.label_stats['std'],  dtype=torch.float32)
                labels = (labels - mean) / (std + 1e-8)
            # Select mode columns if specified (subset mode)
            if self.mode_idx is not None:
                labels = labels[self.mode_idx]
            sample['labels'] = labels
            if self.transform is not None:
                sample = self.transform(sample)
            return sample

        # ── pair-expansion mode: T² items per example ──
        if self._temporal:
            T = self.T
            example_idx = idx // (T * T)
            frame_i     = (idx // T) % T
            frame_j     = idx % T
            i = int(self.indices[example_idx])
            I1 = torch.from_numpy(
                self._psfs_ds[i, 0, frame_i].astype(np.float32)
            ).unsqueeze(0)                                      # [1, H, W]
            I2 = torch.from_numpy(
                self._psfs_ds[i, 1, frame_j].astype(np.float32)
            ).unsqueeze(0)                                      # [1, H, W]
            sample_id = i * (T * T) + (frame_i * T + frame_j)
        else:
            i = int(self.indices[idx])
            psf_pair = self._psfs_ds[i].astype(np.float32)     # [2, H, W]
            I1 = torch.from_numpy(psf_pair[0]).unsqueeze(0)    # [1, H, W]
            I2 = torch.from_numpy(psf_pair[1]).unsqueeze(0)    # [1, H, W]
            sample_id = i

        r  = (I1 - I2) / (I1 + I2 + EPS_RODDIER)              # [1, H, W]

        raw_labels = self._labels_ds[i]                        # [n_modes]
        labels = torch.from_numpy(raw_labels.astype(np.float32)) * self.label_scale

        if self.label_stats is not None:
            mean = torch.as_tensor(self.label_stats['mean'], dtype=torch.float32)
            std  = torch.as_tensor(self.label_stats['std'],  dtype=torch.float32)
            labels = (labels - mean) / (std + 1e-8)

        # Select mode columns if specified (subset mode) — before augmentation
        if self.mode_idx is not None:
            labels = labels[self.mode_idx]

        sample = {'I1': I1, 'I2': I2, 'r': r, 'labels': labels, 'sample_id': sample_id}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample



# ──────────────────────────────────────────────────────────────────────
# Dataset utilities
# ──────────────────────────────────────────────────────────────────────

def train_val_test_split(
    hdf5_path: str,
    ratios: tuple[float, float, float] = (0.80, 0.10, 0.10),
    seed: int = 42,
    amplitude_range_nm: Optional[Union[Tuple[float, float], List[float]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly partition the dataset into train / val / test index arrays.

    Parameters
    ----------
    hdf5_path : str
    ratios : tuple of 3 floats summing to 1.0
    seed : int
        Random seed for reproducibility.
    amplitude_range_nm : tuple/list of (min_nm, max_nm), optional
        If set, filters examples so only those whose RMS wavefront error across
        non-zero Zernike modes (Z4+) falls within [min_nm, max_nm] are retained.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray[int64]
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.6f}")

    with h5py.File(hdf5_path, 'r') as f:
        labels = f['labels'][:]
        valid_mask = np.max(np.abs(labels), axis=-1) > 0

        if amplitude_range_nm is not None:
            min_nm = float(amplitude_range_nm[0])
            max_nm = float(amplitude_range_nm[1])
            if min_nm > max_nm:
                raise ValueError(f"amplitude_range_nm min ({min_nm}) cannot exceed max ({max_nm})")
            # Determine label units: if max abs is small (<1e-3), it is in metres OPD (~1e-7m)
            is_metres = np.max(np.abs(labels)) < 1e-3
            scale = 1e9 if is_metres else 1.0
            col_start = 3 if labels.shape[1] > 3 else 0
            wfe_nm = np.sqrt(np.sum((labels[:, col_start:] * scale) ** 2, axis=-1))
            amp_mask = (wfe_nm >= min_nm) & (wfe_nm <= max_nm)
            valid_mask = valid_mask & amp_mask

        valid_indices = np.nonzero(valid_mask)[0].astype(np.int64)

    N = len(valid_indices)
    if N == 0:
        if amplitude_range_nm is not None:
            raise ValueError(f"No valid examples found in {hdf5_path} with RMS WFE in [{min_nm}, {max_nm}] nm.")
        raise ValueError(f"No valid (non-zero) examples found in {hdf5_path}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N).astype(np.int64)

    n_train = int(N * ratios[0])
    n_val   = int(N * ratios[1])

    train_idx = valid_indices[perm[:n_train]]
    val_idx   = valid_indices[perm[n_train : n_train + n_val]]
    test_idx  = valid_indices[perm[n_train + n_val :]]

    return train_idx, val_idx, test_idx


def subsample_indices(
    indices,
    ratio: Optional[float] = None,
    count: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Subsample a fraction or count of example indices without replacement.

    Parameters
    ----------
    indices : np.ndarray or array-like
        Array of dataset example indices.
    ratio : float, optional
        Fraction of examples to retain, in the range (0.0, 1.0].
    count : int, optional
        Exact count of examples to retain, in the range [1, len(indices)].
    seed : int
        Random seed for deterministic, reproducible selection across epochs/runs.

    Returns
    -------
    np.ndarray[int64]
        Subsampled indices, sorted ascending to preserve sequential HDF5 chunk locality.
        Uses permutation prefix to guarantee nested subsets across counts.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if count is not None:
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        n_samples = min(len(indices), int(count))
    elif ratio is not None:
        ratio = float(ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0.0, 1.0], got {ratio}")
        if ratio >= 1.0 or len(indices) == 0:
            return indices
        n_samples = max(1, int(round(len(indices) * ratio)))
    else:
        return indices

    if n_samples >= len(indices):
        return indices

    rng = np.random.default_rng(seed)
    chosen_pos = rng.permutation(len(indices))[:n_samples]
    return np.sort(indices[chosen_pos])


def get_n_modes(hdf5_path: str) -> int:
    """
    Read the number of Zernike modes from an HDF5 file produced by
    make_training_data.py.

    Uses the stored ``labels.attrs['n_modes']`` attribute when available;
    falls back to ``labels.shape[1]`` for files written without the attribute.
    Raises ValueError if the two values are present but disagree.
    """
    with h5py.File(hdf5_path, 'r') as f:
        n_from_shape = f['labels'].shape[1]
        n_from_attr  = f['labels'].attrs.get('n_modes', None)
    if n_from_attr is not None and int(n_from_attr) != n_from_shape:
        raise ValueError(
            f"{hdf5_path}: labels.attrs['n_modes']={n_from_attr} does not match "
            f"labels.shape[1]={n_from_shape}.  The HDF5 file may be corrupt."
        )
    return n_from_shape


def compute_label_stats(hdf5_path, train_indices):
    """
    Compute per-mode mean and standard deviation of Zernike labels over the
    training set.

    Reads all training labels in a single HDF5 call (~56 MB for 1 M × 14 float32).
    Indices are sorted before reading to maximise HDF5 read performance.

    Parameters
    ----------
    hdf5_path : str
    train_indices : array-like of int

    Returns
    -------
    dict with keys 'mean' and 'std', each an ndarray[14] of float32.
    """
    sorted_idx = np.sort(np.asarray(train_indices, dtype=np.int64))
    with h5py.File(hdf5_path, 'r') as f:
        labels = f['labels'][sorted_idx]          # [N_train, 14] float32
    mean = labels.mean(axis=0).astype(np.float32)
    std  = labels.std(axis=0).astype(np.float32)
    return {'mean': mean, 'std': std}

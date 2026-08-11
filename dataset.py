import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

# Roddier signal stabilisation constant
EPS_RODDIER = 1e-6


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
        {I1:[T,H,W], I2:[T,H,W], R:[T²,H,W], labels}.
        Required for input_mode='two_stream' or 'r_stack'.
    mode_columns : array-like of int or None
        If set, 0-based HDF5 label-column indices to select (subset mode
        training), in the given order — e.g. [4,5,6,7,8,9,10,11,12,13,14]
        selects Noll modes Z5..Z15.  Selection occurs after z-score
        normalisation but before the augmentation transform, ensuring
        D4Augment operates on correctly-sized, correctly-ordered labels.
    """

    def __init__(self, hdf5_path, indices, label_stats=None, transform=None, return_stacks=False, mode_columns=None):
        self.path = str(hdf5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_stats = label_stats
        self.transform = transform
        self.return_stacks = return_stacks
        # if set, select (and reorder) these 0-based label columns before augmentation
        self.mode_idx = torch.as_tensor(mode_columns, dtype=torch.long) if mode_columns is not None else None
        self._file = None          # opened lazily; one handle per DataLoader worker

        # Detect schema and store T (frames per stream).
        with h5py.File(self.path, 'r') as f:
            shape = f['psfs'].shape   # (N, 2, T, H, W) or (N, 2, H, W)
        self._temporal = (len(shape) == 5)
        self.T = int(shape[2]) if self._temporal else 1

    # ------------------------------------------------------------------
    # pickling: drop the open file handle so forked workers open fresh
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None
        return state

    def __len__(self):
        if self.return_stacks:
            return len(self.indices)
        return len(self.indices) * self.T * self.T

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.path, 'r')

        if self.return_stacks:
            # ── stack mode: one item per example, returns all T frames ──
            i = int(self.indices[idx])
            if self._temporal:
                I1 = torch.from_numpy(
                    self._file['psfs'][i, 0].astype(np.float32)   # [T, H, W]
                )
                I2 = torch.from_numpy(
                    self._file['psfs'][i, 1].astype(np.float32)   # [T, H, W]
                )
            else:
                psf_pair = self._file['psfs'][i].astype(np.float32)  # [2, H, W]
                I1 = torch.from_numpy(psf_pair[0]).unsqueeze(0)      # [1, H, W]
                I2 = torch.from_numpy(psf_pair[1]).unsqueeze(0)      # [1, H, W]

            # Compute all T² Roddier combinations via broadcasting.
            T = I1.shape[0]
            I1_exp = I1.unsqueeze(1)   # [T, 1, H, W]
            I2_exp = I2.unsqueeze(0)   # [1, T, H, W]
            R = (I1_exp - I2_exp) / (I1_exp + I2_exp + EPS_RODDIER)  # [T, T, H, W]
            R = R.reshape(T * T, *R.shape[2:])                        # [T², H, W]

            raw_labels = self._file['labels'][i]
            labels = torch.from_numpy(raw_labels.astype(np.float32))
            if self.label_stats is not None:
                mean = torch.as_tensor(self.label_stats['mean'], dtype=torch.float32)
                std  = torch.as_tensor(self.label_stats['std'],  dtype=torch.float32)
                labels = (labels - mean) / (std + 1e-8)
            # Select mode columns if specified (subset mode)
            if self.mode_idx is not None:
                labels = labels[self.mode_idx]
            sample = {'I1': I1, 'I2': I2, 'R': R, 'labels': labels}
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
                self._file['psfs'][i, 0, frame_i].astype(np.float32)
            ).unsqueeze(0)                                      # [1, H, W]
            I2 = torch.from_numpy(
                self._file['psfs'][i, 1, frame_j].astype(np.float32)
            ).unsqueeze(0)                                      # [1, H, W]
        else:
            i = int(self.indices[idx])
            psf_pair = self._file['psfs'][i].astype(np.float32) # [2, H, W]
            I1 = torch.from_numpy(psf_pair[0]).unsqueeze(0)    # [1, H, W]
            I2 = torch.from_numpy(psf_pair[1]).unsqueeze(0)    # [1, H, W]

        r  = (I1 - I2) / (I1 + I2 + EPS_RODDIER)              # [1, H, W]

        raw_labels = self._file['labels'][i]                    # [n_modes]
        labels = torch.from_numpy(raw_labels.astype(np.float32))

        if self.label_stats is not None:
            mean = torch.as_tensor(self.label_stats['mean'], dtype=torch.float32)
            std  = torch.as_tensor(self.label_stats['std'],  dtype=torch.float32)
            labels = (labels - mean) / (std + 1e-8)

        # Select mode columns if specified (subset mode) — before augmentation
        if self.mode_idx is not None:
            labels = labels[self.mode_idx]

        sample = {'I1': I1, 'I2': I2, 'r': r, 'labels': labels}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# ──────────────────────────────────────────────────────────────────────
# Dataset utilities
# ──────────────────────────────────────────────────────────────────────

def train_val_test_split(hdf5_path, ratios=(0.80, 0.10, 0.10), seed=42):
    """
    Randomly partition the dataset into train / val / test index arrays.

    Parameters
    ----------
    hdf5_path : str
    ratios : tuple of 3 floats summing to 1.0
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray[int64]
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.6f}")

    with h5py.File(hdf5_path, 'r') as f:
        valid_indices = np.array(
            [i for i, row in enumerate(f['labels']) if np.max(row) > 0],
            dtype=np.int64,
        )
    N = len(valid_indices)
    if N == 0:
        raise ValueError(f"No valid (non-zero) examples found in {hdf5_path}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N).astype(np.int64)

    n_train = int(N * ratios[0])
    n_val   = int(N * ratios[1])

    train_idx = valid_indices[perm[:n_train]]
    val_idx   = valid_indices[perm[n_train : n_train + n_val]]
    test_idx  = valid_indices[perm[n_train + n_val :]]

    return train_idx, val_idx, test_idx


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

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

# Roddier signal stabilisation constant
EPS_RODDIER = 1e-6


class CWFSDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset for Curvature WFS training data stored in HDF5.

    Expected HDF5 schema
    --------------------
    psfs   : float16  [N, 2, H, W]  — channel 0 = intra-focal, 1 = extra-focal
    labels : float32  [N, 14]       — Zernike coefficients Z2..Z15 (Noll ordering)

    The Roddier normalised-difference signal
        r = (I1 - I2) / (I1 + I2 + eps)
    is computed on-the-fly in __getitem__ and returned as a third tensor,
    so it does not need to be stored on disk.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file produced by make_training_data.py.
    indices : array-like of int
        Sample indices to expose.  Allows train / val / test views of the same
        file without copying data.  Use train_val_test_split() to generate these.
    label_stats : dict or None
        Optional {'mean': ndarray[14], 'std': ndarray[14]} for z-score
        normalisation of labels.  Compute once with compute_label_stats() and
        pass the result here.  If None, raw coefficient values are returned.
    transform : callable or None
        Optional callable applied to the sample dict after loading.
        Expected signature: transform(sample: dict) -> dict.
        See utils/augmentation.py for D4Augment.
    """

    def __init__(self, hdf5_path, indices, label_stats=None, transform=None):
        self.path = str(hdf5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_stats = label_stats
        self.transform = transform
        self._file = None          # opened lazily; one handle per DataLoader worker

    # ------------------------------------------------------------------
    # pickling: drop the open file handle so forked workers open fresh
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file'] = None
        return state

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.path, 'r')

        i = int(self.indices[idx])
        psf_pair = self._file['psfs'][i].astype(np.float32)   # [2, H, W]
        raw_labels = self._file['labels'][i]                    # [14]

        I1 = torch.from_numpy(psf_pair[0]).unsqueeze(0)        # [1, H, W]
        I2 = torch.from_numpy(psf_pair[1]).unsqueeze(0)        # [1, H, W]
        r  = (I1 - I2) / (I1 + I2 + EPS_RODDIER)              # [1, H, W]

        labels = torch.from_numpy(raw_labels.astype(np.float32))  # [14]

        if self.label_stats is not None:
            mean = torch.as_tensor(self.label_stats['mean'], dtype=torch.float32)
            std  = torch.as_tensor(self.label_stats['std'],  dtype=torch.float32)
            labels = (labels - mean) / (std + 1e-8)

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
        N = f['labels'].shape[0]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N).astype(np.int64)

    n_train = int(N * ratios[0])
    n_val   = int(N * ratios[1])

    train_idx = perm[:n_train]
    val_idx   = perm[n_train : n_train + n_val]
    test_idx  = perm[n_train + n_val :]

    return train_idx, val_idx, test_idx


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

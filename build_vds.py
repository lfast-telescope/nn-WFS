"""
build_vds.py — Zero-Copy HDF5 Virtual Dataset (VDS) Builder for nn_WFS

Scans multiple (potentially prematurely terminated) HDF5 simulation files,
detects populated non-zero sample ranges, and constructs an HDF5 Virtual Dataset (VDS)
that maps slices of the source files into a single unified dataset without copying
PSF tensors or duplicating disk storage.

Usage
-----
    # Combine all matching partial files into a VDS:
    python build_vds.py --inputs "data/cwfs_synthetic_*ex_8fr_SAVE*.h5" \
                        --output data/cwfs_combined_vds.h5 --test-read

    # Dry-run inspection without writing:
    python build_vds.py --inputs "data/*.h5" --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# File Scanner & Schema Validation
# ──────────────────────────────────────────────────────────────────────────────

def scan_hdf5_sources(
    patterns: List[str],
    min_valid_rows: int = 1,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Expand file patterns, open each HDF5 file, detect valid populated rows,
    and verify schema consistency across all files.

    Parameters
    ----------
    patterns : list of str
        File paths, directories, or glob patterns.
    min_valid_rows : int
        Minimum non-zero rows required to include a file.
    verbose : bool

    Returns
    -------
    source_infos : list of dict
        Metadata for each valid source file.
    schema : dict
        Common dataset schema (shapes, dtypes, n_modes, attributes).
    """
    # 1. Expand glob patterns and resolve unique files
    file_paths: List[Path] = []
    for pat in patterns:
        expanded = glob.glob(pat)
        if not expanded and os.path.exists(pat):
            expanded = [pat]
        for f in expanded:
            p = Path(f).resolve()
            if p.is_file() and p not in file_paths:
                file_paths.append(p)

    file_paths.sort()
    if not file_paths:
        raise FileNotFoundError(f"No HDF5 files matched the input pattern(s): {patterns}")

    if verbose:
        print(f"Scanning {len(file_paths)} candidate HDF5 files...")

    source_infos: List[Dict[str, Any]] = []
    common_schema: Optional[Dict[str, Any]] = None

    for p in file_paths:
        try:
            with h5py.File(p, "r") as hf:
                if "psfs" not in hf or "labels" not in hf:
                    if verbose:
                        print(f"  [SKIP] {p.name}: Missing 'psfs' or 'labels' dataset.")
                    continue

                psfs_shape = hf["psfs"].shape
                psfs_dtype = hf["psfs"].dtype
                labels_shape = hf["labels"].shape
                labels_dtype = hf["labels"].dtype

                # Fast vectorized detection of populated rows
                raw_labels = hf["labels"][:]
                valid_mask = np.max(np.abs(raw_labels), axis=-1) > 0
                n_valid = int(np.sum(valid_mask))
                n_alloc = labels_shape[0]

                if n_valid < min_valid_rows:
                    if verbose:
                        print(f"  [SKIP] {p.name}: No valid non-zero rows (alloc={n_alloc}).")
                    continue

                # Verify valid rows are contiguous from index 0
                first_invalid = int(np.argmin(valid_mask)) if not np.all(valid_mask) else n_alloc
                if first_invalid != n_valid:
                    if verbose:
                        print(f"  [WARN] {p.name}: Non-contiguous valid rows detected "
                              f"(found {n_valid} non-zero rows, but first zero at {first_invalid}). "
                              f"Truncating to contiguous prefix of {first_invalid} rows.")
                    n_valid = first_invalid

                label_units = hf["labels"].attrs.get("label_units", "metres_opd")
                if isinstance(label_units, bytes):
                    label_units = label_units.decode("utf-8")

                n_modes = int(hf["labels"].attrs.get("n_modes", labels_shape[1]))

                file_info = {
                    "path": p,
                    "filename": p.name,
                    "filesize_mb": os.path.getsize(p) / (1024 * 1024),
                    "n_alloc": n_alloc,
                    "n_valid": n_valid,
                    "psfs_shape": psfs_shape,
                    "psfs_dtype": psfs_dtype,
                    "labels_shape": labels_shape,
                    "labels_dtype": labels_dtype,
                    "n_modes": n_modes,
                    "label_units": str(label_units),
                    "temporal": (len(psfs_shape) == 5),
                    "t_frames": int(psfs_shape[2]) if len(psfs_shape) == 5 else 1,
                }

                # Validate consistency against common schema
                if common_schema is None:
                    common_schema = {
                        "psfs_shape_tail": psfs_shape[1:],  # (2, T, H, W) or (2, H, W)
                        "psfs_dtype": psfs_dtype,
                        "labels_dtype": labels_dtype,
                        "n_modes": n_modes,
                        "label_units": str(label_units),
                        "temporal": file_info["temporal"],
                        "t_frames": file_info["t_frames"],
                    }
                else:
                    if file_info["psfs_shape"][1:] != common_schema["psfs_shape_tail"]:
                        raise ValueError(
                            f"Schema mismatch in {p.name}: 'psfs' sample shape "
                            f"{file_info['psfs_shape'][1:]} does not match expected "
                            f"{common_schema['psfs_shape_tail']}"
                        )
                    if file_info["n_modes"] != common_schema["n_modes"]:
                        raise ValueError(
                            f"Schema mismatch in {p.name}: 'n_modes'={file_info['n_modes']} "
                            f"does not match expected {common_schema['n_modes']}"
                        )

                source_infos.append(file_info)

        except Exception as e:
            if verbose:
                print(f"  [ERROR] {p.name}: {e}")

    if not source_infos or common_schema is None:
        raise ValueError("No valid HDF5 sources found matching criteria.")

    return source_infos, common_schema


# ──────────────────────────────────────────────────────────────────────────────
# Virtual Dataset Construction
# ──────────────────────────────────────────────────────────────────────────────

def build_virtual_dataset(
    source_infos: List[Dict[str, Any]],
    schema: Dict[str, Any],
    output_path: str | Path,
    relative_paths: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Construct an HDF5 Virtual Dataset file mapping valid slices of source files.

    Parameters
    ----------
    source_infos : list of dict
        Metadata dicts for each valid source file (from scan_hdf5_sources).
    schema : dict
        Common dataset schema (from scan_hdf5_sources).
    output_path : str or Path
        Target .h5 file path.
    relative_paths : bool
        If True, store source file paths relative to output_path.parent.
    verbose : bool

    Returns
    -------
    Path to created VDS file.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    total_valid = sum(s["n_valid"] for s in source_infos)
    total_alloc = sum(s["n_alloc"] for s in source_infos)
    total_filesize_mb = sum(s["filesize_mb"] for s in source_infos)

    psfs_vshape = (total_valid, *schema["psfs_shape_tail"])
    labels_vshape = (total_valid, schema["n_modes"])

    layout_psfs = h5py.VirtualLayout(shape=psfs_vshape, dtype=schema["psfs_dtype"])
    layout_labels = h5py.VirtualLayout(shape=labels_vshape, dtype=schema["labels_dtype"])

    manifest: List[Dict[str, Any]] = []
    current_offset = 0

    for s in source_infos:
        n_valid = s["n_valid"]
        src_path = s["path"]

        # Determine path representation for virtual source
        if relative_paths:
            try:
                vsource_filename = os.path.relpath(src_path, out_p.parent)
            except ValueError:
                vsource_filename = str(src_path)
        else:
            vsource_filename = str(src_path)

        vsource_psfs = h5py.VirtualSource(
            vsource_filename, "psfs", shape=s["psfs_shape"]
        )[0:n_valid]
        vsource_labels = h5py.VirtualSource(
            vsource_filename, "labels", shape=s["labels_shape"]
        )[0:n_valid]

        layout_psfs[current_offset : current_offset + n_valid] = vsource_psfs
        layout_labels[current_offset : current_offset + n_valid] = vsource_labels

        manifest.append({
            "filename": s["filename"],
            "source_path": vsource_filename,
            "vds_offset_start": current_offset,
            "vds_offset_end": current_offset + n_valid,
            "n_valid": n_valid,
            "n_alloc": s["n_alloc"],
        })

        current_offset += n_valid

    # Write the Virtual Dataset file
    with h5py.File(out_p, "w", libver="latest") as f:
        f.create_virtual_dataset("psfs", layout_psfs)
        f.create_virtual_dataset("labels", layout_labels)

        # Standard metadata attributes expected by dataset.py & train.py
        f["labels"].attrs["n_modes"] = schema["n_modes"]
        f["labels"].attrs["label_units"] = schema["label_units"]
        f["labels"].attrs["vds_total_sources"] = len(source_infos)
        f["labels"].attrs["vds_manifest_json"] = json.dumps(manifest)

    vds_filesize_kb = os.path.getsize(out_p) / 1024.0

    if verbose:
        print(f"\nSuccessfully created Virtual Dataset: {out_p}")
        print(f"  Total valid examples : {total_valid:,} (from {total_alloc:,} allocated in {len(source_infos)} files)")
        print(f"  Virtual PSFs shape   : {psfs_vshape} [{schema['psfs_dtype']}]")
        print(f"  Virtual Labels shape : {labels_vshape} [{schema['labels_dtype']}]")
        print(f"  VDS metadata file size: {vds_filesize_kb:.1f} KB (references {total_filesize_mb:.1f} MB source data)")
        if schema["temporal"]:
            t = schema["t_frames"]
            print(f"  Effective frame pairs: {total_valid * t * t:,} ({total_valid:,} × {t}² combinations)")

    return out_p


# ──────────────────────────────────────────────────────────────────────────────
# Summary & Test Read Verification
# ──────────────────────────────────────────────────────────────────────────────

def print_source_summary_table(source_infos: List[Dict[str, Any]]) -> None:
    """Print formatted summary table of all scanned sources."""
    print("\n" + "─" * 86)
    print(f"{'Source File':<42} {'Valid / Alloc':<18} {'Populated':<12} {'Size':<10}")
    print("─" * 86)
    total_valid = 0
    total_alloc = 0
    total_mb = 0.0

    for s in source_infos:
        pct = (s["n_valid"] / s["n_alloc"]) * 100.0 if s["n_alloc"] > 0 else 0.0
        print(f"{s['filename']:<42} {s['n_valid']:>6,d} / {s['n_alloc']:<8,d} {pct:>8.1f}%  {s['filesize_mb']:>7.1f} MB")
        total_valid += s["n_valid"]
        total_alloc += s["n_alloc"]
        total_mb += s["filesize_mb"]

    total_pct = (total_valid / total_alloc) * 100.0 if total_alloc > 0 else 0.0
    print("─" * 86)
    print(f"{'TOTAL (' + str(len(source_infos)) + ' files)':<42} {total_valid:>6,d} / {total_alloc:<8,d} {total_pct:>8.1f}%  {total_mb:>7.1f} MB")
    print("─" * 86)


def test_read_vds(vds_path: str | Path, n_samples: int = 5) -> None:
    """
    Test reading slices from the Virtual Dataset across file boundaries
    using the CWFSDataset loader.
    """
    print(f"\nRunning test reads on {vds_path}...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from nn_WFS.dataset import CWFSDataset, train_val_test_split, get_n_modes

    n_modes = get_n_modes(str(vds_path))
    train_idx, val_idx, test_idx = train_val_test_split(str(vds_path), ratios=(0.8, 0.1, 0.1), seed=42)

    print(f"  train_val_test_split succeeded: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Test CWFSDataset in stack mode
    ds_stacks = CWFSDataset(str(vds_path), train_idx[:n_samples], return_stacks=True)
    sample0 = ds_stacks[0]
    print(f"  Stack mode read test:")
    print(f"    I1 shape    : {tuple(sample0['I1'].shape)}")
    print(f"    I2 shape    : {tuple(sample0['I2'].shape)}")
    print(f"    R shape     : {tuple(sample0['R'].shape)}")
    print(f"    Labels shape: {tuple(sample0['labels'].shape)} (non-zero: {torch_or_np_any(sample0['labels'])})")

    # Test boundary reads
    with h5py.File(vds_path, "r") as f:
        N = f["labels"].shape[0]
        # Read first, middle, last
        indices = [0, N // 2, N - 1]
        for idx in indices:
            p_slice = f["psfs"][idx]
            l_slice = f["labels"][idx]
            assert np.max(np.abs(l_slice)) > 0, f"Row {idx} has all zero labels!"
            assert not np.isnan(p_slice).any(), f"Row {idx} has NaNs in psfs!"

    print(f"  Boundary index checks (rows {indices}) verified non-zero and NaN-free.")
    print("  All VDS validation checks passed successfully!\n")


def torch_or_np_any(arr: Any) -> bool:
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return bool(np.any(arr != 0))


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an HDF5 Virtual Dataset (VDS) from multiple partial simulation runs."
    )
    parser.add_argument(
        "--inputs", "-i", nargs="+", required=True,
        help="One or more HDF5 file paths, directories, or glob patterns (e.g. 'data/*.h5')"
    )
    parser.add_argument(
        "--output", "-o", default="data/cwfs_combined_vds.h5",
        help="Target output Virtual Dataset .h5 file path (default: data/cwfs_combined_vds.h5)"
    )
    parser.add_argument(
        "--absolute-paths", action="store_true",
        help="Store absolute paths in the Virtual Dataset (default: relative paths)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and display source summary table without writing output file"
    )
    parser.add_argument(
        "--test-read", action="store_true",
        help="Perform test reads and dataset loader verification after writing VDS"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_infos, schema = scan_hdf5_sources(args.inputs, verbose=True)
    print_source_summary_table(source_infos)

    if args.dry_run:
        print("\nDry-run complete. No files written.")
        return

    vds_path = build_virtual_dataset(
        source_infos,
        schema,
        output_path=args.output,
        relative_paths=not args.absolute_paths,
        verbose=True,
    )

    if args.test_read:
        test_read_vds(vds_path)


if __name__ == "__main__":
    main()

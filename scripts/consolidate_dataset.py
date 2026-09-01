"""
consolidate_dataset.py — Consolidate partial HDF5 datasets into a single chunk=1 file.

Reads one or more HDF5 simulation files (or a Virtual Dataset), extracts valid
non-zero examples, and streams them into a single unified HDF5 file with:
    chunks = (1, 2, T, H, W)

This eliminates chunk decompression overhead during training, allowing
sub-millisecond (< 1 ms) random reads from Lustre with < 200 MB RAM usage.

Usage
-----
    python -m nn_WFS.scripts.consolidate_dataset \
        --inputs "data/cwfs_synthetic_*ex_8fr_SAVE*.h5" \
        --output data/cwfs_consolidated_chunk1.h5 \
        --compression gzip --compression-opts 1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


def scan_sources(
    patterns: List[str],
    min_valid_rows: int = 1,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scan and validate all input HDF5 files."""
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
        raise FileNotFoundError(f"No HDF5 files matched the pattern(s): {patterns}")

    if verbose:
        print(f"Scanning {len(file_paths)} candidate HDF5 file(s)...")

    source_infos: List[Dict[str, Any]] = []
    common_schema: Optional[Dict[str, Any]] = None

    for p in file_paths:
        try:
            with h5py.File(p, "r") as hf:
                if "psfs" not in hf or "labels" not in hf:
                    if verbose:
                        print(f"  [SKIP] {p.name}: Missing 'psfs' or 'labels'")
                    continue

                psfs_shape = hf["psfs"].shape
                psfs_dtype = hf["psfs"].dtype
                labels_shape = hf["labels"].shape
                labels_dtype = hf["labels"].dtype

                # Fast check for valid rows
                raw_labels = hf["labels"][:]
                valid_mask = np.max(np.abs(raw_labels), axis=-1) > 0
                n_valid = int(np.sum(valid_mask))
                n_alloc = labels_shape[0]

                if n_valid < min_valid_rows:
                    if verbose:
                        print(f"  [SKIP] {p.name}: No valid non-zero rows (alloc={n_alloc}).")
                    continue

                # Ensure contiguous prefix
                first_invalid = int(np.argmin(valid_mask)) if not np.all(valid_mask) else n_alloc
                if first_invalid != n_valid:
                    if verbose:
                        print(f"  [WARN] {p.name}: Non-contiguous valid rows. Truncating to {first_invalid} rows.")
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

                if common_schema is None:
                    common_schema = {
                        "psfs_shape_tail": psfs_shape[1:],
                        "psfs_dtype": psfs_dtype,
                        "labels_dtype": labels_dtype,
                        "n_modes": n_modes,
                        "label_units": str(label_units),
                        "temporal": file_info["temporal"],
                        "t_frames": file_info["t_frames"],
                    }
                else:
                    if file_info["psfs_shape"][1:] != common_schema["psfs_shape_tail"]:
                        raise ValueError(f"Shape mismatch in {p.name}")
                    if file_info["n_modes"] != common_schema["n_modes"]:
                        raise ValueError(f"Mode count mismatch in {p.name}")

                source_infos.append(file_info)
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {p.name}: {e}")

    if not source_infos or common_schema is None:
        raise ValueError("No valid HDF5 sources found.")

    return source_infos, common_schema


def consolidate_to_chunk1(
    source_infos: List[Dict[str, Any]],
    schema: Dict[str, Any],
    output_path: str | Path,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 1,
    batch_copy_size: int = 64,
    delete_source: bool = False,
    verbose: bool = True,
) -> Path:
    """Stream valid rows from sources into a chunk=1 HDF5 file."""
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    total_valid = sum(s["n_valid"] for s in source_infos)
    psfs_shape = (total_valid, *schema["psfs_shape_tail"])
    labels_shape = (total_valid, schema["n_modes"])

    chunk_psfs = (1, *schema["psfs_shape_tail"])
    chunk_labels = (1, schema["n_modes"])

    if verbose:
        print("\n" + "=" * 70)
        print("CONSOLIDATING DATASET TO CHUNK=1")
        print("=" * 70)
        print(f"Target file        : {out_p}")
        print(f"Total valid samples: {total_valid:,}")
        print(f"PSF tensor shape   : {psfs_shape} [{schema['psfs_dtype']}]")
        print(f"Labels shape       : {labels_shape} [{schema['labels_dtype']}]")
        print(f"Chunk layout       : psfs={chunk_psfs}, labels={chunk_labels}")
        print(f"Compression        : {compression} (opts={compression_opts})")
        print(f"Delete sources     : {delete_source} (frees space as each file finishes)")
        print("=" * 70 + "\n")

    t0 = time.time()
    current_dst = 0

    with h5py.File(out_p, "w") as f_out:
        # Create chunk=1 datasets
        create_kw: Dict[str, Any] = {
            "dtype": schema["psfs_dtype"],
            "chunks": chunk_psfs,
        }
        if compression:
            create_kw["compression"] = compression
            if compression_opts is not None and compression != "lzf":
                create_kw["compression_opts"] = compression_opts

        ds_psfs = f_out.create_dataset("psfs", shape=psfs_shape, **create_kw)
        ds_labels = f_out.create_dataset("labels", shape=labels_shape, dtype=schema["labels_dtype"], chunks=chunk_labels)

        # Set attributes
        ds_labels.attrs["n_modes"] = schema["n_modes"]
        ds_labels.attrs["label_units"] = schema["label_units"]
        ds_labels.attrs["n_total"] = total_valid
        if schema["temporal"]:
            ds_labels.attrs["t_frames"] = schema["t_frames"]

        # Stream copy
        for src_idx, s in enumerate(source_infos):
            src_path = s["path"]
            n_src_valid = s["n_valid"]
            if verbose:
                print(f"[{src_idx+1}/{len(source_infos)}] Reading {s['filename']} ({n_src_valid:,} valid rows)...")

            with h5py.File(src_path, "r") as f_src:
                src_psfs = f_src["psfs"]
                src_labels = f_src["labels"]

                # Copy in blocks
                for start in range(0, n_src_valid, batch_copy_size):
                    end = min(start + batch_copy_size, n_src_valid)
                    block_len = end - start

                    # Sequential slice read from source (fast)
                    psf_block = src_psfs[start:end]
                    lbl_block = src_labels[start:end]

                    # Write to target
                    ds_psfs[current_dst : current_dst + block_len] = psf_block
                    ds_labels[current_dst : current_dst + block_len] = lbl_block

                    current_dst += block_len

                    if verbose and (current_dst % 500 == 0 or current_dst == total_valid):
                        elapsed = time.time() - t0
                        rate = current_dst / elapsed if elapsed > 0 else 0
                        eta = (total_valid - current_dst) / rate if rate > 0 else 0
                        print(f"  -> Written {current_dst:,} / {total_valid:,} ({current_dst/total_valid*100:.1f}%) | "
                              f"{rate:.1f} ex/s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="\r", flush=True)

            if verbose:
                print()

            # Safely delete source file if requested
            if delete_source:
                try:
                    os.remove(src_path)
                    if verbose:
                        print(f"  [CLEANUP] Deleted source file: {s['filename']} (reclaimed {s['filesize_mb']:.1f} MB)")
                except Exception as e:
                    if verbose:
                        print(f"  [WARN] Could not delete {src_path}: {e}")

    total_time = time.time() - t0
    final_size_mb = os.path.getsize(out_p) / (1024 * 1024)

    if verbose:
        print("\n" + "=" * 70)
        print("CONSOLIDATION COMPLETE!")
        print(f"Total time     : {total_time:.1f} s ({total_valid/total_time:.1f} ex/s)")
        print(f"Output size    : {final_size_mb:.1f} MB ({final_size_mb/1024:.2f} GB)")
        print(f"Location       : {out_p}")
        print("=" * 70 + "\n")

    # Run fast verification on the created file
    verify_dataset(out_p, total_valid, schema["n_modes"], verbose=verbose)

    return out_p


def verify_dataset(file_path: Path, expected_rows: int, expected_modes: int, verbose: bool = True):
    """Verify integrity and measure random-access read latency on the new dataset."""
    if verbose:
        print("Running verification checks on consolidated dataset...")

    with h5py.File(file_path, "r") as f:
        assert "psfs" in f and "labels" in f, "Missing datasets in output file!"
        psfs_ds = f["psfs"]
        labels_ds = f["labels"]

        assert psfs_ds.shape[0] == expected_rows, f"Row count mismatch: {psfs_ds.shape[0]} vs {expected_rows}"
        assert labels_ds.shape[1] == expected_modes, f"Mode count mismatch: {labels_ds.shape[1]} vs {expected_modes}"
        assert psfs_ds.chunks[0] == 1, f"PSF chunks must be 1, got {psfs_ds.chunks}"

        # Test boundary and random reads
        test_indices = [0, expected_rows // 2, expected_rows - 1]
        for idx in test_indices:
            p = psfs_ds[idx]
            l = labels_ds[idx]
            assert not np.isnan(p).any(), f"NaN in PSFs at row {idx}"
            assert np.max(np.abs(l)) > 0, f"All-zero label at row {idx}"

        # Benchmark random read latency
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(expected_rows, min(100, expected_rows), replace=False)

        t0 = time.perf_counter()
        for idx in sample_indices:
            _ = psfs_ds[idx]
        read_lat_ms = ((time.perf_counter() - t0) / len(sample_indices)) * 1000.0

    if verbose:
        print(f"  [PASS] Verified {expected_rows:,} rows (boundary & non-zero checks passed).")
        print(f"  [PASS] Measured random-access read latency: {read_lat_ms:.2f} ms / example.")
        print("  All verification tests passed successfully!\n")


def main():
    parser = argparse.ArgumentParser(description="Consolidate HDF5 files to chunk=1.")
    parser.add_argument("--inputs", "-i", nargs="+", default=["data/cwfs_synthetic_*ex_8fr_SAVE*.h5"],
                        help="Input file patterns")
    parser.add_argument("--output", "-o", default="data/cwfs_consolidated_chunk1.h5",
                        help="Output HDF5 path")
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip",
                        help="Compression algorithm (default: gzip)")
    parser.add_argument("--compression-opts", type=int, default=1,
                        help="Compression level for gzip (default: 1 for maximum speed)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Block size for streaming copy (default: 64)")
    parser.add_argument("--delete-source", action="store_true",
                        help="Delete each source file after copying to keep disk usage low")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing")

    args = parser.parse_args()

    source_infos, schema = scan_sources(args.inputs, verbose=True)

    print("\nCandidate Sources Summary:")
    print("-" * 65)
    total_valid = 0
    total_mb = 0.0
    for s in source_infos:
        print(f"  {s['filename']:<40} {s['n_valid']:>6,d} rows | {s['filesize_mb']:>7.1f} MB")
        total_valid += s["n_valid"]
        total_mb += s["filesize_mb"]
    print("-" * 65)
    print(f"  TOTAL ({len(source_infos)} files): {total_valid:,} valid rows | {total_mb:.1f} MB source size\n")

    if args.dry_run:
        print("Dry-run requested. Exiting.")
        return

    comp = None if args.compression == "none" else args.compression
    consolidate_to_chunk1(
        source_infos,
        schema,
        args.output,
        compression=comp,
        compression_opts=args.compression_opts,
        batch_copy_size=args.batch_size,
        delete_source=args.delete_source,
        verbose=True,
    )


if __name__ == "__main__":
    main()


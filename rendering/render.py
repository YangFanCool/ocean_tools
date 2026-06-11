"""Ocean 2D map rendering — local adaptation of the remote ocean_render.py.

Reads .dat files (flattened valid-only, normalized to [-1,1]) and renders
layer 0 as a 3600x1800 equirectangular PNG using the pre-computed cell ID map.

Usage:
    # Render a single .dat file
    python3 render.py single path/to/0000.dat -o output.png

    # Render from raw .npy (applies mask + per-frame normalization internally)
    python3 render.py single path/to/0000.npy -o output.png --raw

    # Batch render a directory of .dat files
    python3 render.py batch path/to/ens_dir/ -o output_dir/
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from rasterizer import RasterRenderer

log = logging.getLogger("render")

NC_PATH = Path(__file__).resolve().parent.parent / "visualization" / "ocean.nc"
N_CELLS = 235160
N_LAYERS = 60
FILL_THRESHOLD = -1e30


def _load_valid_mask() -> np.ndarray:
    """Build valid-point mask from ocean.nc (or first .npy if available)."""
    import xarray as xr
    ds = xr.open_dataset(NC_PATH)
    temp = ds["temperature"].isel(Time=0).values  # (N_CELLS, N_LAYERS)
    return (temp.reshape(-1) > FILL_THRESHOLD)


def build_layer0_index() -> np.ndarray:
    valid_mask = _load_valid_mask()
    layer0_flat_indices = np.arange(0, N_CELLS * N_LAYERS, N_LAYERS)
    cumsum = np.cumsum(valid_mask)
    return (cumsum[layer0_flat_indices] - 1).astype(np.int64)


def load_layer0_from_dat(dat_path: str | Path, layer0_idx: np.ndarray) -> np.ndarray:
    dat = np.fromfile(dat_path, dtype=np.float32)
    return dat[layer0_idx]


def load_layer0_from_npy(npy_path: str | Path) -> np.ndarray:
    """Load raw .npy, apply mask, normalize to [-1,1], return layer 0."""
    raw = np.load(npy_path)  # (N_CELLS, N_LAYERS)
    layer0 = raw[:, 0].astype(np.float32)
    valid = layer0 > FILL_THRESHOLD
    fmin, fmax = layer0[valid].min(), layer0[valid].max()
    if fmax - fmin < 1e-10:
        layer0[valid] = 0.0
    else:
        layer0[valid] = (layer0[valid] - fmin) / (fmax - fmin) * 2.0 - 1.0
    layer0[~valid] = 0.0
    return layer0


def cmd_single(args):
    renderer = RasterRenderer()
    src = Path(args.input)

    if args.raw or src.suffix == ".npy":
        values = load_layer0_from_npy(src)
    else:
        layer0_idx = build_layer0_index()
        values = load_layer0_from_dat(src, layer0_idx)

    out = Path(args.output) if args.output else src.with_suffix(".png")
    renderer.render(values, out)
    log.info(f"Saved: {out}")


def cmd_batch(args):
    renderer = RasterRenderer()
    src_dir = Path(args.input_dir)
    out_dir = Path(args.output) if args.output else src_dir / "rendered"

    dat_files = sorted(src_dir.glob("*.dat"))
    npy_files = sorted(src_dir.glob("*.npy"))

    if dat_files:
        layer0_idx = build_layer0_index()
        for f in dat_files:
            out = out_dir / f"{f.stem}.png"
            if out.exists() and not args.force:
                continue
            values = load_layer0_from_dat(f, layer0_idx)
            renderer.render(values, out)
            log.info(f"Rendered: {out}")
    elif npy_files:
        for f in npy_files:
            out = out_dir / f"{f.stem}.png"
            if out.exists() and not args.force:
                continue
            values = load_layer0_from_npy(f)
            renderer.render(values, out)
            log.info(f"Rendered: {out}")
    else:
        log.warning(f"No .dat or .npy files found in {src_dir}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Ocean Voronoi cell renderer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="Render a single file")
    p_single.add_argument("input", help=".dat or .npy file")
    p_single.add_argument("-o", "--output", help="Output PNG path")
    p_single.add_argument("--raw", action="store_true", help="Input is raw .npy (not pre-normalized .dat)")

    p_batch = sub.add_parser("batch", help="Render all files in a directory")
    p_batch.add_argument("input_dir", help="Directory containing .dat or .npy files")
    p_batch.add_argument("-o", "--output", help="Output directory")
    p_batch.add_argument("--force", action="store_true", help="Overwrite existing PNGs")

    args = parser.parse_args()
    if args.cmd == "single":
        cmd_single(args)
    elif args.cmd == "batch":
        cmd_batch(args)


if __name__ == "__main__":
    main()

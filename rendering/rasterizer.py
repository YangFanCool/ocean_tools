"""Voronoi cell rasterizer for MPAS-Ocean data.

Uses a pre-computed cell ID map (pixel → cell index) for fast lookup rendering.
Each frame is a pure array operation + PIL save (~0.36s/frame on server).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
COLORMAP_DIR = HERE.parent / "visualization" / "colormap"


def load_kindlmann_lut(n: int = 256) -> np.ndarray:
    csv_path = COLORMAP_DIR / "kindlmann-table-float-1024.csv"
    color_data = pd.read_csv(
        csv_path, skiprows=1,
        names=["scalar", "RGB_r", "RGB_g", "RGB_b"],
        dtype=np.float64,
    )
    colors = color_data[["RGB_r", "RGB_g", "RGB_b"]].values
    idx = np.linspace(0, len(colors) - 1, n).astype(int)
    return (colors[idx] * 255).astype(np.uint8)


def load_cell_id_map(path: str | Path | None = None) -> np.ndarray:
    if path is None:
        path = HERE / "cell_id_map.npz"
    return np.load(path)["cell_id_map"]


class RasterRenderer:
    """Render per-cell scalar values onto an equirectangular raster via lookup."""

    def __init__(
        self,
        cell_id_map: np.ndarray | None = None,
        lut: np.ndarray | None = None,
    ):
        self.cell_id_map = cell_id_map if cell_id_map is not None else load_cell_id_map()
        self.lut = lut if lut is not None else load_kindlmann_lut()
        self.H, self.W = self.cell_id_map.shape
        self.mask = self.cell_id_map >= 0

    def render(self, values: np.ndarray, output_path: str | Path) -> None:
        """Render values (N_CELLS,) in [-1, 1] to a PNG.

        Land pixels (cell_id == -1) are white.
        """
        n = len(self.lut)
        color_idx = np.clip(((values + 1) / 2 * (n - 1)).astype(np.int32), 0, n - 1)
        cell_colors = self.lut[color_idx]

        canvas = np.full((self.H, self.W, 3), 255, dtype=np.uint8)
        canvas[self.mask] = cell_colors[self.cell_id_map[self.mask]]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(output_path)

    def render_to_array(self, values: np.ndarray) -> np.ndarray:
        """Same as render() but returns (H, W, 3) uint8 array instead of saving."""
        n = len(self.lut)
        color_idx = np.clip(((values + 1) / 2 * (n - 1)).astype(np.int32), 0, n - 1)
        cell_colors = self.lut[color_idx]

        canvas = np.full((self.H, self.W, 3), 255, dtype=np.uint8)
        canvas[self.mask] = cell_colors[self.cell_id_map[self.mask]]
        return canvas

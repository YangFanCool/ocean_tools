# MPAS-Ocean Visualization & Benchmarks

MPAS-Ocean (Voronoi mesh, 235,160 cells x 60 layers) visualization toolkit and DiT model benchmark results for the FANR paper.

## Structure

```
ocean/
├── visualization/          Exploratory vis of raw MPAS-Ocean NetCDF data
├── rendering/              Paper-quality Voronoi cell rasterization pipeline
├── benchmarks/             DiT architecture ablation charts (PSNR, time, size)
```

## visualization/

Matplotlib-based exploration of the raw `ocean.nc` dataset.

| Script | Purpose |
|--------|---------|
| `plot_ocean_layers.py` | Per-layer global temperature & salinity scatter plots (Kindlmann colormap) |
| `prepare_coords.py` | Normalize lon/lat/layer to [-1, 1] → `coord.npy` for neural network input |

Data files:
- `ocean.nc` — MPAS-Ocean output (1.8 GB, 235160 cells x 60 layers, temperature + salinity)
- `coord.npy` — Normalized coordinates (235160, 60, 3), order: [lon, lat, layer]
- `colormap/` — Kindlmann colormap tables (various resolutions)

```bash
cd visualization
python3 plot_ocean_layers.py    # outputs to figs/
python3 prepare_coords.py       # outputs coord.npy
```

## rendering/

Fast Voronoi cell rasterizer producing 3600x1800 equirectangular PNGs. Uses a pre-computed `cell_id_map.npz` (pixel → cell index lookup) so each frame is a pure array operation (~0.36s/frame).

| File | Purpose |
|------|---------|
| `rasterizer.py` | `RasterRenderer` class + Kindlmann LUT loader |
| `render.py` | CLI: render single .dat/.npy or batch a directory |
| `cell_id_map.npz` | Pre-computed (1800, 3600) int32 raster; -1 = land |

```bash
cd rendering

# Render a single pre-normalized .dat (values in [-1, 1])
python3 render.py single path/to/0042.dat -o output.png

# Render a raw .npy (applies mask + normalization internally)
python3 render.py single path/to/0042.npy -o output.png --raw

# Batch render all .dat files in a directory
python3 render.py batch path/to/ens_dir/ -o output_dir/
```

### Pipeline (on remote server)

The full paper-figure pipeline runs on the compute server:

```
ocean.nc (mesh topology)  →  cell_id_map.npz  (one-time precompute)
raw .npy (per-ensemble)   →  ocean_gt.py       →  .dat (masked + normalized)
.dat (per-method infer)   →  ocean_render.py   →  .png (3600x1800)
per-method .png           →  gen_compare.py    →  comparison grid
comparison grid           →  pick_compare.py   →  zoom boxes + crops
```

## benchmarks/

Ablation results comparing DiT configurations (plane_size x DiT_size x patch_size, each S/M/L).

| Script | Purpose |
|--------|---------|
| `plot_benchmarks.py` | Generate grouped bar charts from `results.csv` |

Output directories: `psnr/`, `train_time/`, `model_size/` — each contains overview + grouped-by-{patch, plane, dit} plots.

```bash
cd benchmarks
python3 plot_benchmarks.py
```

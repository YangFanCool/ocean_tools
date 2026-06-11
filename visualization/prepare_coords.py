import xarray as xr
import numpy as np

# --- Configuration ---
NC_FILE = "output.nc"
OUTPUT_FILE = "coord.npy"
N_LAYERS = 60

def main():
    print(f"Loading {NC_FILE}...")
    ds = xr.open_dataset(NC_FILE)

    # 1. 读取原始数据 (235160,)
    #    MPAS 输出通常是弧度 (Radians)
    lon = ds['lonCell'].values.astype(np.float32)
    lat = ds['latCell'].values.astype(np.float32)
    n_cells = lon.shape[0]

    print(f"Cells: {n_cells}, Layers: {N_LAYERS}")

    # 2. 物理归一化 (Physical Normalization)
    
    # --- Longitude: [0, 2pi] -> [-1, 1] ---
    # 公式: (lon / pi) - 1.0
    # 0 -> -1
    # pi -> 0
    # 2pi -> 1
    lon_norm = (lon / np.pi) - 1.0

    # --- Latitude: [-pi/2, pi/2] -> [-1, 1] ---
    # 公式: lat / (pi/2)
    # -pi/2 -> -1
    # 0 -> 0
    # pi/2 -> 1
    lat_norm = lat / (np.pi / 2.0)

    # --- Layer: [0, 59] -> [-1, 1] ---
    # 使用 linspace 生成从 -1 到 1 的均匀分布
    layer_norm = np.linspace(-1.0, 1.0, N_LAYERS, dtype=np.float32)

    # 3. 广播与堆叠 (Broadcasting & Stacking)
    # 目标 Shape: (N_cells, N_layers, 3)
    # Order: [Lon, Lat, Layer]

    # Expand Lon: (N,) -> (N, 60)
    lon_grid = np.tile(lon_norm[:, np.newaxis], (1, N_LAYERS))

    # Expand Lat: (N,) -> (N, 60)
    lat_grid = np.tile(lat_norm[:, np.newaxis], (1, N_LAYERS))

    # Expand Layer: (60,) -> (N, 60)
    layer_grid = np.tile(layer_norm[np.newaxis, :], (n_cells, 1))

    # Stack: axis=2
    coord_3d = np.stack([lon_grid, lat_grid, layer_grid], axis=2)

    # 4. 检查与保存
    print("\n--- Verification ---")
    print(f"Shape: {coord_3d.shape}")
    print(f"Order: [Lon, Lat, Layer]")
    
    # 检查物理边界是否溢出 (MPAS有时候会有极其微小的浮点误差导致 > 2pi)
    # 如果必须严格限制在 -1, 1，可以加上 clip
    coord_3d = np.clip(coord_3d, -1.0, 1.0)
    
    print(f"Data Min/Max: {coord_3d.min():.4f} / {coord_3d.max():.4f}")
    
    # 检查 Lat 南极点 (示例)
    # 如果你的数据最南端是 -1.37 (-78度), 归一化后应该是 -1.37/1.57 = -0.87 左右
    # 而不是 -1.0 (那是 -90度)
    print(f"Sample Lat min (normalized): {coord_3d[:,:,1].min():.4f} (Expected approx -0.87 for -78 deg)")

    np.save(OUTPUT_FILE, coord_3d)
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
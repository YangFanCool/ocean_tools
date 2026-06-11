import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --- 1. 配置和加载数据 ---
input_file = 'output.nc'
output_dir = 'figs'
# ❗ 确保此路径与您实际存储 CSV 文件的路径一致
KINDLMANN_CSV = 'colormap/kindlmann-table-float-1024.csv' 
FILL_VALUE = -1e+34 # 定义 Mask 填充值
TIME_STEP_TO_PLOT = 0 # 绘制第一个时间步的数据

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

print(f"Loading data from {input_file}...")
try:
    ds = xr.open_dataset(input_file)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading NetCDF file: {e}")
    exit()

# 获取必要变量
lon_cell_rad = ds['lonCell'].values
lat_cell_rad = ds['latCell'].values
temperature_data = ds['temperature'] # xarray DataArray
salinity_data = ds['salinity']     # xarray DataArray

# 修正：将弧度坐标转换为角度坐标
lon_cell_deg = np.rad2deg(lon_cell_rad)
lat_cell_deg = np.rad2deg(lat_cell_rad)
lon_cell_deg[lon_cell_deg > 180] -= 360 # 调整经度到 [-180, 180] 范围

lon_cell = lon_cell_deg
lat_cell = lat_cell_deg

num_vert_levels = temperature_data.shape[2]
print(f"Total vertical levels to plot: {num_vert_levels}")

# ------------------------------------------------------------------
# ❗ 修正：计算有效数据的 min/max
# ------------------------------------------------------------------
print("Calculating true color map ranges...")

# 临时将填充值替换为 NaN，以便计算 min/max
temp_masked = temperature_data.where(temperature_data != FILL_VALUE)
salt_masked = salinity_data.where(salinity_data != FILL_VALUE)

# 计算有效数据的 min/max
TEMP_CMAP_MIN = temp_masked.min().item()
TEMP_CMAP_MAX = temp_masked.max().item()

SALT_CMAP_MIN = salt_masked.min().item()
SALT_CMAP_MAX = salt_masked.max().item()

# 打印最终使用的范围
print(f"Temperature colormap range (Valid Data): {TEMP_CMAP_MIN} to {TEMP_CMAP_MAX}")
print(f"Salinity colormap range (Valid Data): {SALT_CMAP_MIN} to {SALT_CMAP_MAX}")
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 修正：从 CSV 文件创建 Kindlmann Colormap
# ------------------------------------------------------------------
print(f"Loading Kindlmann colormap from {KINDLMANN_CSV}...")
try:
    # 加载 CSV 文件，包含 scalar, RGB_r, RGB_g, RGB_b 列
    color_data = pd.read_csv(KINDLMANN_CSV, skiprows=1, 
                             names=['scalar', 'RGB_r', 'RGB_g', 'RGB_b'], 
                             dtype=np.float64)
    
    # 颜色列表：(R, G, B)
    kindlmann_colors = color_data[['RGB_r', 'RGB_g', 'RGB_b']].values
    
    # 创建 LinearSegmentedColormap
    TEMP_CMAP = LinearSegmentedColormap.from_list(
        'Kindlmann_Custom', 
        kindlmann_colors, 
        N=len(kindlmann_colors)
    )
    
    # 盐度色标仍使用 Matplotlib 默认色标
    SALT_CMAP = LinearSegmentedColormap.from_list(
        'Kindlmann_Custom', 
        kindlmann_colors, 
        N=len(kindlmann_colors)
    )
    print("Kindlmann Colormap created successfully from CSV data.")

except FileNotFoundError:
    print(f"Error: {KINDLMANN_CSV} not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error processing CSV file: {e}")
    exit()
# ------------------------------------------------------------------


# --- 2. 循环每一层并绘图 ---
print("Starting plotting process with pure Matplotlib and custom colormap...")

for level_idx in range(num_vert_levels):
    
    temp_layer_raw = temperature_data.isel(Time=TIME_STEP_TO_PLOT, nVertLevels=level_idx).values
    salt_layer_raw = salinity_data.isel(Time=TIME_STEP_TO_PLOT, nVertLevels=level_idx).values

    # 修正：将填充值替换为 NaN
    temp_layer = np.where(temp_layer_raw == FILL_VALUE, np.nan, temp_layer_raw)
    salt_layer = np.where(salt_layer_raw == FILL_VALUE, np.nan, salt_layer_raw)
    
    # 过滤掉 NaN 值及其对应的坐标
    valid_indices = ~np.isnan(temp_layer)
    
    if np.sum(valid_indices) == 0:
        print(f"Layer {level_idx:02d} has no valid data. Skipping.")
        continue

    lon_valid = lon_cell[valid_indices]
    lat_valid = lat_cell[valid_indices]
    temp_valid = temp_layer[valid_indices]
    salt_valid = salt_layer[valid_indices]
    
    # --- 创建 Figure 和 Axes ---
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(50, 10))
    fig.suptitle(f'Layer {level_idx} (Vertical Level {level_idx}) - Time Step {TIME_STEP_TO_PLOT}', fontsize=16)

    # --- 绘制温度图 (左侧) ---
    ax_temp = axes[0]
    ax_temp.set_title(r'Temperature ($\degree$C)')
    ax_temp.set_facecolor('black') # 陆地/Mask 区域显示为黑色
    
    # ❗ 使用修正后的范围和 Kindlmann 色标
    scat_temp = ax_temp.scatter(
        lon_valid, lat_valid, c=temp_valid, cmap=TEMP_CMAP, s=1,
        vmin=TEMP_CMAP_MIN, vmax=TEMP_CMAP_MAX
    )
    
    ax_temp.set_xlim([-180, 180])
    ax_temp.set_ylim([-90, 90])
    ax_temp.set_xlabel(r'Longitude ($\degree$)')
    ax_temp.set_ylabel(r'Latitude ($\degree$)')
    ax_temp.grid(linestyle=':', color='gray') 

    # 添加颜色条
    cbar_temp = fig.colorbar(scat_temp, ax=ax_temp, orientation='vertical', pad=0.05, shrink=0.7)
    cbar_temp.set_label(r'Temperature ($\degree$C)')

    # --- 绘制盐度图 (右侧) ---
    ax_salt = axes[1]
    ax_salt.set_title('Salinity (PSU)')
    ax_salt.set_facecolor('black')
    
    # ❗ 使用修正后的范围和默认色标
    scat_salt = ax_salt.scatter(
        lon_valid, lat_valid, c=salt_valid, cmap=SALT_CMAP, s=1,
        vmin=SALT_CMAP_MIN, vmax=SALT_CMAP_MAX
    )
    
    ax_salt.set_xlim([-180, 180])
    ax_salt.set_ylim([-90, 90])
    ax_salt.set_xlabel(r'Longitude ($\degree$)')
    ax_salt.set_ylabel(r'Latitude ($\degree$)')
    ax_salt.grid(linestyle=':', color='gray') 

    # 添加颜色条
    cbar_salt = fig.colorbar(scat_salt, ax=ax_salt, orientation='vertical', pad=0.05, shrink=0.7)
    cbar_salt.set_label('Salinity (PSU)') 
    
    # 存储图片到 figs 目录
    output_filename = os.path.join(output_dir, f'global_map_layer_{level_idx:02d}.png')
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig) 
    
    print(f"Layer {level_idx:02d} plot saved to {output_filename}")

print("\n🎉 All plots generated and saved to 'figs/' directory.")
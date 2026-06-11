import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置与预处理
# ==========================================
df = pd.read_csv('results.csv')

def time_to_minutes(t_str):
    h, m, s = map(int, t_str.split(':'))
    return h * 60 + m + s / 60

df['training_minutes'] = df['training time'].apply(time_to_minutes)
df['Config'] = df['plane_size'] + '_' + df['DiT_size']

size_order = ['S', 'M', 'L']

# 全局风格
sns.set_theme(style="whitegrid", context="notebook")

# ====== 自定义配色方案 (莫兰迪色) =======
morandi_colors = ["#afd4e3", "#b5d4be", "#f6cfaf"] # 浅蓝，浅橙，浅绿
current_palette = sns.color_palette(morandi_colors)
# ===========================

# ==========================================
# 2. 通用绘图函数
# ==========================================
# ====== min change (参数增加 label_fmt) =======
def generate_plots_for_metric(data, y_col, y_label, output_dir, palette, y_max=None, y_min=None, label_fmt='%.1f'):
# ============================================
    os.makedirs(output_dir, exist_ok=True)
    
    # ====== min change (添加数值标签辅助函数) =======
    def add_bar_labels(ax, fmt):
        for container in ax.containers:
            ax.bar_label(container, fmt=fmt, fontsize=8, padding=2)
    # ============================================

    # --- Fig 1: Overview ---
    plt.figure(figsize=(12, 6))
    
    ax1 = sns.barplot(
        data=data, x='Config', y=y_col, hue='patch_size', 
        palette=palette, errorbar=None 
    )
    
    ax1.xaxis.grid(False) 
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
    if y_max: ax1.set_ylim(y_min if y_min is not None else 0, y_max)
    
    # ====== min change (应用标签) =======
    add_bar_labels(ax1, label_fmt)
    # ===================================
    
    # Fig 1 图例横排
    sns.move_legend(ax1, "upper left", title='Patch Size', frameon=True, ncol=3)

    plt.title(f'Overview: {y_label}', fontsize=14)
    plt.ylabel(y_label)
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_overview.png', dpi=600)
    plt.close()

    # --- Fig 2: Split by Patch ---
    unique_patches = sorted(data['patch_size'].unique())
    # ====== min change (增加 wspace 间距以容纳中间Y轴) =======
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True, gridspec_kw={'wspace': 0.25})
    # =======================================================
    fig2.suptitle(f'{y_label} - Grouped by Patch Size', fontsize=16)

    for i, p_size in enumerate(unique_patches):
        if i >= 3: break 
        subset = data[data['patch_size'] == p_size]
        sns.barplot(data=subset, x='Config', y=y_col, ax=axes2[i], palette=palette, errorbar=None)
        
        # ====== min change (显式开启轴标签) =======
        axes2[i].tick_params(labelleft=True)             # 强制显示左轴
        if i == 2: axes2[i].tick_params(labelright=True) # 最后一个图显示右轴
        if i > 0: axes2[i].set_ylabel('')                # 隐藏中间图的ylabel文字但保留刻度
        add_bar_labels(axes2[i], label_fmt)              # 添加数值
        # ==========================================
        
        axes2[i].set_title(f'Patch Size = {p_size}')
        axes2[i].tick_params(axis='x', rotation=45)
        axes2[i].xaxis.grid(False)
        axes2[i].yaxis.grid(True, linestyle='--', alpha=0.5)
        if y_max: axes2[i].set_ylim(y_min if y_min is not None else 0, y_max)
             
    # plt.tight_layout() # tight_layout 会覆盖 wspace 设置，改用 subplots_adjust
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.08, right=0.92, wspace=0.25)
    plt.savefig(f'{output_dir}/2_by_patch.png', dpi=600)
    plt.close()

    # --- Fig 3: Split by Plane ---
    # ====== min change (增加 wspace) =======
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5), sharey=True, gridspec_kw={'wspace': 0.25})
    # =====================================
    fig3.suptitle(f'{y_label} - Impact of DiT Size (Fixed Plane)', fontsize=16)
    for i, plane in enumerate(size_order):
        subset = data[data['plane_size'] == plane]
        ax = sns.barplot(data=subset, x='DiT_size', y=y_col, hue='patch_size', 
                        order=size_order, ax=axes3[i], palette=palette, errorbar=None)
        
        # ====== min change (显式开启轴标签) =======
        ax.tick_params(labelleft=True)
        if i == 2: ax.tick_params(labelright=True)
        if i > 0: ax.set_ylabel('')
        add_bar_labels(ax, label_fmt)
        # ==========================================

        ax.set_title(f'Plane Size = {plane}')
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        if y_max: ax.set_ylim(y_min if y_min is not None else 0, y_max)
        
        # Fig 3 图例横排
        sns.move_legend(ax, "upper left", title='Patch Size', frameon=True, ncol=3)
        
    plt.subplots_adjust(top=0.85, wspace=0.25)
    plt.savefig(f'{output_dir}/3_by_plane.png', dpi=600)
    plt.close()

    # --- Fig 4: Split by DiT ---
    # ====== min change (增加 wspace) =======
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5), sharey=True, gridspec_kw={'wspace': 0.25})
    # =====================================
    fig4.suptitle(f'{y_label} - Impact of Plane Size (Fixed DiT)', fontsize=16)
    for i, dit in enumerate(size_order):
        subset = data[data['DiT_size'] == dit]
        ax = sns.barplot(data=subset, x='plane_size', y=y_col, hue='patch_size',
                    order=size_order, ax=axes4[i], palette=palette, errorbar=None)
        
        # ====== min change (显式开启轴标签) =======
        ax.tick_params(labelleft=True)
        if i == 2: ax.tick_params(labelright=True)
        if i > 0: ax.set_ylabel('')
        add_bar_labels(ax, label_fmt)
        # ==========================================

        ax.set_title(f'DiT Size = {dit}')
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        if y_max: ax.set_ylim(y_min if y_min is not None else 0, y_max)
        
        # Fig 4 图例横排
        sns.move_legend(ax, "upper left", title='Patch Size', frameon=True, ncol=3)
        
    plt.subplots_adjust(top=0.85, wspace=0.25)
    plt.savefig(f'{output_dir}/4_by_dit.png', dpi=600)
    plt.close()

# ==========================================
# 3. 执行逻辑
# ==========================================

print("Processing 1/3: Training Time (Morandi)...")
generate_plots_for_metric(
    df, y_col='training_minutes', y_label='Training Time (Minutes)', 
    output_dir='train_time', palette=current_palette, label_fmt='%.0f'
)

print("Processing 2/3: Model Size (Morandi)...")
generate_plots_for_metric(
    df, y_col='size_in_MB', y_label='Model Size (MB)', 
    output_dir='model_size', palette=current_palette, label_fmt='%.0f'
)

print("Processing 3/3: PSNR (Infer Only, Morandi)...")
generate_plots_for_metric(
    df, y_col='infer_PSNR', y_label='Inference PSNR (dB)', 
    output_dir='psnr', palette=current_palette, y_max=55, y_min=20, label_fmt='%.1f'
)

print("All plots generated with Horizontal Legend!")
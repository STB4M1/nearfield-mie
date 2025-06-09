import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib import font_manager

# ✅=== フォント設定（Times New Roman + Computer Modern）===✅
font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"  # システムによって変更必要
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()

mpl.rcParams.update({
    "font.family": font_name,
    "mathtext.fontset": "cm",
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 300
})

print(f"✅ 使用フォント名: {font_name}")

# === データ読み込み ===
def load_intensity_data(dat_path):
    data = np.loadtxt(dat_path)
    x = data[:, 0]
    y = data[:, 1]
    intensity = data[:, 2]
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    intensity_grid = intensity.reshape(len(y_unique), len(x_unique))
    X, Y = np.meshgrid(x_unique, y_unique)
    return X, Y, intensity_grid, x_unique, y_unique

# === プロット関数 ===
def save_contour_plot(X, Y, intensity, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, filename)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, intensity, levels=60, cmap='viridis')
    cbar = plt.colorbar(cp, label="Intensity [a.u.]", pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel(r"$x$ [μm]")
    plt.ylabel(r"$y$ [μm]")
    plt.title("Fresnel Diffraction Intensity")
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(full_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved contour plot: {full_path}")

def save_heatmap(intensity, x_min, x_max, y_min, y_max, out_dir, filename, interpolation="nearest"):
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, filename)

    plt.figure(figsize=(8, 6))
    plt.imshow(intensity, extent=[x_min, x_max, y_min, y_max],
               origin='lower', cmap='viridis',
               interpolation=interpolation, aspect='equal')
    cbar = plt.colorbar(label="Intensity [a.u.]", pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel(r"$x$ [μm]")
    plt.ylabel(r"$y$ [μm]")
    title_str = "Fresnel Intensity (smooth)" if interpolation != "nearest" else "Fresnel Intensity (pixel)"
    plt.title(title_str)
    plt.tight_layout()
    plt.savefig(full_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved heatmap ({interpolation}): {full_path}")

# === メイン処理 ===
def main():
    dat_path = "../00_ASM/data/fresnel_20um_opaque_particle_05umres_z10um.dat"  # 必要に応じて変更
    out_dir = "../00_ASM/figs"  # 出力先

    # ファイル名の設定
    contour_file = "ASM_contour_20um_opaque_particle_05umres_z10um.svg"
    heatmap_file = "ASM_heatmap_20um_opaque_particle_05umres_z10um.svg"
    smooth_file = "ASM_smooth_20um_opaque_particle_05umres_z10um.svg"

    # データ読み込み
    X, Y, intensity_grid, x_vals, y_vals = load_intensity_data(dat_path)

    # プロット実行
    save_contour_plot(X, Y, intensity_grid, out_dir, contour_file)
    save_heatmap(intensity_grid, x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max(),
                 out_dir, heatmap_file, interpolation="nearest")
    save_heatmap(intensity_grid, x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max(),
                 out_dir, smooth_file, interpolation="bicubic")

main()

import os
import numpy as np
from PIL import Image
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import json  # JSONはまだ使っていませんが、Juliaコードにあるので一応インポート
import matplotlib.pyplot as plt

width = 256 # pixel
height = 256 # pixel
el = 0.5  # µm, ピクセル間隔（= サンプリング間隔）
          # サンプリングに基づくナイキスト周波数: f_N = 1 / (2 * el) = 1.0 [cycles/µm]
          # Fresnel伝搬における空間周波数条件: f_x^2 + f_y^2 ≤ (1/λ)^2 = (1/0.6943)^2 ≈ 2.07 [cycles/µm]^2
          # 離散サンプリングによる制限: f_x^2 + f_y^2 ≤ (1/2el)^2 = 1.0^2 = 1.0 [cycles/µm]^2
          # → 実際に使用可能な周波数領域は上記の min() で決まる
          #   = f_x^2 + f_y^2 ≤ 1.0 ⇒ radicand = 1 - (λf_x)^2 - (λf_y)^2 ≥ 0 を保証
dx = el
dy = el
z0 = 10.0  # 伝搬距離 [μm]
# Dz = 15.0 * 1e3
λ = 0.6943  # 波長 [μm]
dp = 20.0

def make_dirs(*dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"Checked/created directory: {dir}")

def trans_radicand(width, height, dx, dy, λ):
    radicand = np.empty((height, width), dtype=np.float32)
    
    for j in range(height):
        for i in range(width):
            x = i - width / 2.0
            y = j - height / 2.0
            radicand[j, i] = 1.0 - (x * λ / (height * dx))**2 - (y * λ / (width * dy))**2

    return radicand

# def trans_radicand_vec(width, height, dx, dy, λ):
#     x = np.arange(width) - width / 2.0
#     y = np.arange(height) - height / 2.0
#     X, Y = np.meshgrid(x, y)
#     radicand = 1.0 - (X * λ / (height * dx))**2 - (Y * λ / (width * dy))**2
#     return radicand.astype(np.float32)

def trans_expi_phi(radicand, width, height, z0, λ):
    expi_phi = np.empty((height, width), dtype=np.complex64)
    # 2πz0/λ * sqrt(radicand) を指数関数の虚数部に
    expi_phi[:, :] = np.exp(1j * 2 * np.pi * z0 / λ * np.sqrt(radicand))
    return expi_phi

def zero_padding(img):
    """
    画像を2倍サイズにpadding（背景は平均輝度値）して返す
    """
    height, width = img.shape
    mean_brightness = np.mean(img)  # 平均輝度（0〜1とか0〜255）
    
    padded_img = np.full((height * 2, width * 2), mean_brightness, dtype=img.dtype)

    start_y = height // 2
    start_x = width // 2

    padded_img[start_y:start_y + height, start_x:start_x + width] = img

    return padded_img


def save_intensity_data(intensity, x_range, y_range, dat_path):
    """
    強度マップ（2D）を .dat ファイルに保存する関数。
    x, y は μm単位。
    """
    Nx, Ny = intensity.shape
    x = np.linspace(0, x_range, Nx)
    y = np.linspace(0, y_range, Ny)

    with open(dat_path, "w") as f:
        for j in range(Ny):
            for i in range(Nx):
                f.write(f"{x[i]:.6f} {y[j]:.6f} {intensity[j, i]:.8e}\n")

    print(f"💾 Saved intensity data to: {dat_path}")

def save_intensity_plots(intensity, x_range, y_range, fig_path_contour, fig_path_heatmap):
    """
    等高線付き＆ヒートマップ画像を保存する関数
    """
    Nx, Ny = intensity.shape
    x = np.linspace(0, x_range, Nx)
    y = np.linspace(0, y_range, Ny)
    X, Y = np.meshgrid(x, y)

    # 等高線付き
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, intensity, levels=60, cmap='viridis')
    plt.colorbar(cp, label="Intensity [a.u.]")
    plt.xlabel("x [µm]")
    plt.ylabel("y [µm]")
    plt.title("Fresnel Diffraction Intensity (contour)")
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(fig_path_contour)
    plt.close()
    print(f"📉 Saved contour plot to: {fig_path_contour}")

    # ヒートマップ（imshow）
    plt.figure(figsize=(8, 6))
    im = plt.imshow(intensity, extent=[0, x_range, 0, y_range], origin='lower', cmap='viridis', aspect='equal')
    plt.colorbar(im, label="Intensity [a.u.]")
    plt.xlabel("x [µm]")
    plt.ylabel("y [µm]")
    plt.title("Fresnel Diffraction Intensity (heatmap)")
    plt.tight_layout()
    plt.savefig(fig_path_heatmap)
    plt.close()
    print(f"🌈 Saved heatmap plot to: {fig_path_heatmap}")

def save_intensity_smooth(intensity, x_range, y_range, fig_path_smooth,
                          interpolation="bicubic",
                          vmin=None, vmax=None):
    """
    補間付きヒートマップ（滑らか版）を保存
    vmin, vmax を与えるとカラーマップの範囲を固定できる
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(intensity,
               extent=[0, x_range, 0, y_range],
               origin='lower',
               cmap='viridis',
               interpolation=interpolation,
               aspect='equal',
               vmin=vmin, vmax=vmax)      # ★ 追加
    plt.colorbar(label="Intensity [a.u.]")
    plt.xlabel("x [µm]")
    plt.ylabel("y [µm]")
    plt.title(f"Fresnel Intensity (smooth: {interpolation})")
    plt.tight_layout()
    plt.savefig(fig_path_smooth)
    plt.close()
    print(f"🖼️  Saved smooth heatmap to: {fig_path_smooth}")


def main():
    out_img_dir = "../00_ASM"
    out_img_path1 = os.path.join(out_img_dir, "Obj.png")
    out_img_path2 = os.path.join(out_img_dir, "Holo1.png")

    make_dirs(out_img_dir)

    # Step 1: 初期画像生成
    img = np.full((height, width), 60, dtype=np.uint8)

    cx = width // 2
    cy = height // 2

    # Step 2: sqrtと複素初期化
    img = np.sqrt(img.astype(np.float32))
    psi = img.astype(np.complex64) + 0.0j

    # Step 3: 中心の円形粒子を除去
    for j in range(height):
        for i in range(width):
            r = np.sqrt((i - cx)**2 + (j - cy)**2)
            if r * el < dp / 2:
                psi[j, i] = 0.0 + 0.0j
    
    # 粒子作製後の強度を計算
    obj_intensity = np.real(psi * np.conj(psi))  # = abs(psi)^2
    img_uint8 = np.clip(obj_intensity, 0, 255).astype(np.uint8) # 最大値255に合わせてスケーリング（正規化ではない）
    Image.fromarray(img_uint8).save(out_img_path1)

    # Step 3.5: パディング（平均輝度で2倍サイズに拡張）
    psi_padded = zero_padding(psi)
    padded_height, padded_width = psi_padded.shape

    # Step 4: 光伝搬（パディング込み）
    radicand = trans_radicand(padded_width, padded_height, dx, dy, λ)
    expi_phi = trans_expi_phi(radicand, padded_width, padded_height, z0, λ)

    psi_fft = fft2(psi_padded)
    psi_fft = fftshift(psi_fft) * expi_phi
    psi_propagated = ifft2(ifftshift(psi_fft))

    # Step 4.5: クロップ（中央領域を抽出）
    offset_y = padded_height // 4
    offset_x = padded_width // 4
    psi = psi_propagated[offset_y:offset_y + height, offset_x:offset_x + width]

    # Step 5: 強度計算と保存
    intensity = np.real(psi * np.conj(psi))  # = abs(psi)^2
    img_uint8 = np.clip(intensity, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(out_img_path2)

    # Step 4.6: 出力用に中心から右上方向の領域を切り出し
    extract_size = 40  # pixel
    center_y = padded_height // 2
    center_x = padded_width // 2

    # 中心から x, y の正方向に extract_size 分だけ取り出す
    psi_extracted = psi_propagated[center_y:center_y + extract_size, center_x:center_x + extract_size]
    intensity_extracted = np.real(psi_extracted * np.conj(psi_extracted))

    # 出力サイズ（実空間）を計算
    extract_width_um = extract_size * dx
    extract_height_um = extract_size * dy

    # Step 6: データ保存＆プロット
    data_dir = "../00_record/data"
    os.makedirs(data_dir, exist_ok=True)
    dat_path = os.path.join(data_dir, "ASM_20um_opaque_particle_05umres_z10um.dat")
    save_intensity_data(intensity_extracted, extract_width_um, extract_height_um, dat_path)

    # fig_dir = "../00_record/figs"
    # os.makedirs(fig_dir, exist_ok=True)
    # fig_contour = os.path.join(fig_dir, "fresnel_contour_20um_opaque_particle_05umres_z10um.svg")
    # fig_heatmap = os.path.join(fig_dir, "fresnel_heatmap_20um_opaque_particle_05umres_z10um.svg")
    # fig_smooth = os.path.join(fig_dir, "fresnel_smooth_20um_opaque_particle_05umres_z10um.svg")  # ←追加

    # save_intensity_plots(intensity_extracted, extract_width_um, extract_height_um, fig_contour, fig_heatmap)
    # save_intensity_smooth(intensity_extracted, extract_width_um, extract_height_um, fig_smooth, interpolation="bicubic")     # お好みで調整


if __name__ == "__main__":
    main()
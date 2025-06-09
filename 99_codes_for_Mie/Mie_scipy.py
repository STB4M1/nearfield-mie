import numpy as np
from scipy.special import jv, lpmv
from numpy import sqrt, pi

import matplotlib.pyplot as plt
from tqdm import tqdm

# === パラメータ設定 ===
λ = 0.6943e-6
d = 20e-6
m = 1.5 - 100j
k = 2 * np.pi / λ
α = np.pi * d / λ
β = m * α
N = round(α + 4 * α**(1/3))
E0 = 1.0
ϵ = 8.854e-12
μ = 4 * np.pi * 1e-7

# === Eq. 10: ψ_n(x)
def ψ(n, x):
    return sqrt(pi * x / 2) * jv(n + 0.5, x)

def ψdiff(n, x):
    return sqrt(pi / (2 * x)) * ((n + 1) * jv(n + 0.5, x) - x * jv(n + 1.5, x))

def ψdiff2(n, x):
    return (1 / (4 * x**(1.5))) * sqrt(pi / 2) * (
        x**2 * jv(n - 1.5, x) - 2 * x**2 * jv(n + 0.5, x) + x**2 * jv(n + 2.5, x)
        + 2 * x * jv(n - 0.5, x) - 2 * x * jv(n + 1.5, x) - jv(n + 0.5, x)
    )

# === Eq. 11: χ_n(x)
def χ(n, x):
    return (-1)**n * sqrt(pi * x / 2) * jv(-n - 0.5, x)

def χdiff(n, x):
    return 0.5 * (-1)**n * sqrt(pi / (2 * x)) * (
        x * jv(-n - 1.5, x) + jv(-n - 0.5, x) - x * jv(-n + 0.5, x)
    )

def χdiff2(n, x):
    return (1 / (4 * x**(1.5))) * sqrt(pi / 2) * (-1)**n * (
        x**2 * jv(-n - 2.5, x) - 2 * x**2 * jv(-n - 0.5, x) + x**2 * jv(-n + 1.5, x)
        + 2 * x * jv(-n - 1.5, x) - 2 * x
    )

# === Eq. 9: ξ_n(x)
def ξ(n, x):
    return ψ(n, x) + 1j * χ(n, x)

def ξdiff(n, x):
    return ψdiff(n, x) + 1j * χdiff(n, x)

def ξdiff2(n, x):
    return ψdiff2(n, x) + 1j * χdiff2(n, x)

# === Eq. 12, 13: a_n, b_n
def a(n, α, β, m):
    num = ψ(n, α) * ψdiff(n, β) - m * ψdiff(n, α) * ψ(n, β)
    den = ξ(n, α) * ψdiff(n, β) - m * ξdiff(n, α) * ψ(n, β)
    return num / den

def b(n, α, β, m):
    num = m * ψ(n, α) * ψdiff(n, β) - ψdiff(n, α) * ψ(n, β)
    den = m * ξ(n, α) * ψdiff(n, β) - ξdiff(n, α) * ψ(n, β)
    return num / den

# === Eq. 20, 21: pifunc, τ
def pifunc(n, θ):
    sθ = np.sin(θ)
    cθ = np.cos(θ)
    return 0.5 * n * (n + 1) if abs(sθ) < 1e-12 else lpmv(1, n, cθ) / sθ

def dnPl(cθ, n, m=1, h=1e-6):
    # 中央差分による数値微分（d/dx P^m_n(x)）
    return (lpmv(m, n, cθ + h) - lpmv(m, n, cθ - h)) / (2 * h)

def τ(n, θ):
    sθ = np.sin(θ)
    cθ = np.cos(θ)
    return 0.0 if abs(sθ) < 1e-12 else -sθ * dnPl(cθ, n, 1)

from numpy import sin, cos, exp, sqrt, pi
from numpy import conj, real

# === Eq. 22: dξ/dr ===
def dξdr(n, r):
    return k * ξdiff(n, k * r)

# === Eq. 23: d²ξ/dr² ===
def d2ξdr2(n, r):
    return k**2 * ξdiff2(n, k * r)

# === Eq. 24: ωとH0（必要なら定数として使用）
c = 3e8
ω = 2 * pi * c / λ
H0 = -k / (ω * μ) * E0  # 単純に電場から磁場への換算係数

# === Eq. 33: Etr ===
def Etr(r, θ, ϕ):
    sum_term = 0
    for n in range(1, N + 1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * a(n, α, β, m) * (ξdiff2(n, k*r) + ξ(n, k*r)) * lpmv(1, n, cos(θ))
        sum_term += term
    return E0 * cos(ϕ) * (sin(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

# === Eq. 34: Etθ ===
def Etθ(r, θ, ϕ):
    sum_term = 0
    for n in range(1, N + 1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * (
            a(n, α, β, m) * ξdiff(n, k*r) * τ(n, θ)
            - 1j * b(n, α, β, m) * ξ(n, k*r) * pifunc(n, θ)
        )
        sum_term += term
    return E0 * cos(ϕ) / (k * r) * (k * r * cos(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

# === Eq. 35: Etϕ ===
def Etϕ(r, θ, ϕ):
    sum_term = 0
    for n in range(1, N + 1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * (
            a(n, α, β, m) * ξdiff(n, k*r) * pifunc(n, θ)
            - 1j * b(n, α, β, m) * ξ(n, k*r) * τ(n, θ)
        )
        sum_term += term
    return -E0 * sin(ϕ) / (k * r) * (k * r * exp(-1j * k * r * cos(θ)) + sum_term)

# === Eq. 36: Htr ===
def Htr(r, θ, ϕ):
    sum_term = 0
    for n in range(1, N + 1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * b(n, α, β, m) * (ξdiff2(n, k*r) + ξ(n, k*r)) * lpmv(1, n, cos(θ))
        sum_term += term
    return E0 * sqrt(ϵ / μ) * sin(ϕ) * (sin(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

# === Eq. 37: Htθ ===
def Htθ(r, θ, ϕ):
    sum_term = 0
    for n in range(1, N + 1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * (
            -1j * a(n, α, β, m) * ξ(n, k*r) * pifunc(n, θ)
            + b(n, α, β, m) * ξdiff(n, k*r) * τ(n, θ)
        )
        sum_term += term
    return E0 / (k * r) * sqrt(ϵ / μ) * sin(ϕ) * (k * r * cos(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

# === Eq. 38: Htϕ ===
def Htϕ(r, θ, ϕ):
    sum_term = 0
    for n in range(1, N + 1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * (
            -1j * a(n, α, β, m) * ξ(n, k*r) * τ(n, θ)
            + b(n, α, β, m) * ξdiff(n, k*r) * pifunc(n, θ)
        )
        sum_term += term
    return E0 / (k * r) * sqrt(ϵ / μ) * cos(ϕ) * (k * r * exp(-1j * k * r * cos(θ)) + sum_term)

# === Eq. 40: 放射強度 S(r, θ, φ)
def S(r, θ, ϕ):
    etθ = Etθ(r, θ, ϕ)
    etϕ = Etϕ(r, θ, ϕ)
    etr = Etr(r, θ, ϕ)
    htθ = Htθ(r, θ, ϕ)
    htϕ = Htϕ(r, θ, ϕ)
    htr = Htr(r, θ, ϕ)

    term1 = cos(θ) * (etθ * conj(htϕ) - etϕ * conj(htθ))
    term2 = -sin(θ) * (etϕ * conj(htr) - etr * conj(htϕ))
    return 0.5 * real(term1 + term2)







# === グリッド設定 ===
x = np.linspace(0.0, 20.0e-6, 60)  # [m]
y = np.linspace(0.0, 20.0e-6, 60)  # [m]
z = 10.0e-6                        # [m]
Nx = len(x)
Ny = len(y)

# === 結果格納用配列 ===
result_arr = np.zeros((Ny, Nx))

# === 散乱強度 S(r, θ, φ) の評価 ===
for j in tqdm(range(Ny), desc="評価中"):
    for i in range(Nx):
        xi = x[i]
        yj = y[j]
        r = np.sqrt(xi**2 + yj**2 + z**2)
        θ = np.arccos(z / r)
        ϕ = 0.0 if (xi == 0 and yj == 0) else np.arctan2(yj, xi)
        result_arr[j, i] = S(r, θ, ϕ)  # Eq. 40 のS()

# === プロット（µmに変換）===
x_um = x * 1e6
y_um = y * 1e6
X, Y = np.meshgrid(x_um, y_um)

plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, result_arr, levels=60, cmap='viridis', vmin=0.0, vmax=0.0025)
plt.colorbar(cp, label="S [W/m²]")
plt.xlabel("x [µm]")
plt.ylabel("y [µm]")
plt.title("Isophote behind 20-µm Transparent Particle (Fig.3)")
plt.gca().set_aspect('equal')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.tight_layout()
plt.savefig("scipy_isophote_close_behind_a_20um_transparent_particle_100.svg")
plt.show()

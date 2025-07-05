using SpecialFunctions, LegendrePolynomials, ProgressMeter
using ForwardDiff
# using Unitful
using Plots


# === 設定値 ===
λ = 0.6943e-6      # 波長 [m]
d = 20e-6          # 粒子径 [m]
m = 1.5 - 0.0im    # 粒子の屈折率（複素数）[-]（無次元）# transparent particle
k = 2π / λ         # 波数 [1/m]
α = π * d / λ      # サイズパラメータ（無次元）[-]
β = m * α          # 複素サイズパラメータ（無次元）[-]
N = round(Int, α + 4α^(1/3))  # 計算に使う最大次数 [-]
E0 = 1.0           # 入射電場の振幅（規格化）[V/m]
ϵ = 8.854e-12      # 真空の誘電率 [F/m]（ファラッド毎メートル）
μ = 4π * 1e-7      # 真空の透磁率 [H/m]（ヘンリー毎メートル）

# x = 0.0:0.2e-6:20.0e-6  # [m] ← 0〜20 µm を m単位で
# y = 0.0:0.2e-6:20.0e-6  # [m]
x = range(0.0, stop=20.0e-6, length=60)  # [m]
y = range(0.0, stop=20.0e-6, length=60)  # [m]
z = 10.0e-6             # [m] ← 粒子の直後

Nx = length(x)
Ny = length(y)

# Eq. 10
ψ(n, x) = sqrt(π*x/2) * besselj(n+1/2, x)
ψdiff(n, x) = sqrt(π/2/x) * ((n+1)*besselj(n+1/2, x) - x * besselj(n+3/2, x))
ψdiff2(n, x) = 1/(4* x^(3/2)) * sqrt(π/2) * (x^2 * besselj(n-3/2,x) - 2*x^2 * besselj(n+1/2,x) + x^2 * besselj(n+5/2,x) + 2*x * besselj(n-1/2,x) - 2*x * besselj(n+3/2,x) - besselj(n+1/2,x))

# Eq. 11
χ(n, x) = (-1)^n * sqrt(π * x / 2) * besselj(-n - 1/2, x)
χdiff(n, x) = 0.5 * (-1)^n * sqrt(π / (2x)) * (
    x * besselj(-n - 3/2, x) + besselj(-n - 1/2, x) - x * besselj(-n + 1/2, x)
)
χdiff2(n, x) = 1/(4* x^(3/2)) * sqrt(π/2) * (-1)^n * (x^2 * besselj(-n-5/2,x) - 2*x^2 * besselj(-n-1/2,x) + x^2 * besselj(-n+3/2,x) + 2*x * besselj(-n-3/2,x) - 2*x)

# Eq. 8 is LegendrePolynomials.Plm

# Eq. 9
ξ(n, x) = ψ(n, x) + im * χ(n, x)
ξdiff(n, x) = ψdiff(n, x) + im * χdiff(n, x)
ξdiff2(n, x) = ψdiff2(n, x) + im * χdiff2(n, x)

# Eq. 12 and 13
function a(n, α, β, m)
    num = ψ(n,α)*ψdiff(n,β) - m*ψdiff(n,α)*ψ(n,β)
    den = ξ(n,α)*ψdiff(n,β) - m*ξdiff(n,α)*ψ(n,β)
    return num / den
end
function b(n, α, β, m)
    num = m * ψ(n,α) * ψdiff(n,β) - ψdiff(n,α) * ψ(n,β)
    den = m * ξ(n,α) * ψdiff(n,β) - ξdiff(n,α) * ψ(n,β)
    return num / den
end

# Eq. 20, 21
# pifunc(n, θ) = Plm(cos(θ), n, 1) / sin(θ)
# τ(n, θ) = -sin(θ) * dnPl(cos(θ), n, 1)

# Eq. 20, 21 修正版
pifunc(n, θ) = abs(sin(θ)) < 1e-12 ? 0.5 * n * (n + 1) : Plm(cos(θ), n, 1) / sin(θ)
τ(n, θ) = abs(sin(θ)) < 1e-12 ? 0.0 : -sin(θ) * dnPl(cos(θ), n, 1)

# Eq. 22
dξdr(n, r) = k * ξdiff(n, k * r)

# Eq. 23
d2ξdr2(n, r) = k^2 * ξdiff2(n, k * r)

# Eq. 24
ω = 2π * 3e8 / λ         # または ω = 2π * f, f = c / λ
H0 = -k / (ω * μ) * E0   # これで十分

# Eq. 33
Etr(r, θ, ϕ) = E0 * cos(ϕ) * (
    sin(θ) * exp(-im * k * r * cos(θ)) +
    sum(im^(n+1) * (-1)^n * (2n+1)/(n*(n+1)) * a(n, α, β, m) *
        (ξdiff2(n, k*r) + ξ(n, k*r)) * Plm(cos(θ), n, 1) for n in 1:N)
)

# Eq. 34
Etθ(r, θ, ϕ) = E0 * cos(ϕ) / (k * r) * (
    k * r * cos(θ) * exp(-im * k * r * cos(θ)) +
    sum(
        im^(n+1) * (-1)^n * (2n + 1) / (n * (n + 1)) *
        (a(n, α, β, m) * ξdiff(n, k*r) * τ(n, θ) - im * b(n, α, β, m) * ξ(n, k*r) * pifunc(n, θ))
        for n in 1:N
    )
)

# Eq. 35
Etϕ(r, θ, ϕ) = -E0 * sin(ϕ) / (k * r) * (
    k * r * exp(-im * k * r * cos(θ)) +
    sum(
        im^(n + 1) * (-1)^n * (2n + 1) / (n * (n + 1)) *
        (a(n, α, β, m) * ξdiff(n, k * r) * pifunc(n, θ) - im * b(n, α, β, m) * ξ(n, k * r) * τ(n, θ))
        for n in 1:N
    )
)

# Eq. 36
Htr(r, θ, ϕ) = E0 * sqrt(ϵ/μ) * sin(ϕ) * (
    sin(θ) * exp(-im * k * r * cos(θ)) +
    sum(
        im^(n+1) * (-1)^n * (2n+1)/(n*(n+1)) *
        b(n, α, β, m) * (ξdiff2(n,k*r)+ξ(n,k*r)) * Plm(cos(θ), n, 1)
        for n in 1:N
    )
)

# Eq. 37
Htθ(r, θ, ϕ) = E0/(k*r) * sqrt(ϵ/μ) * sin(ϕ) * (
    k * r * cos(θ) * exp(-im * k * r * cos(θ)) +
    sum(
        im^(n+1) * (-1)^n * (2n+1)/(n*(n+1)) *
        (-im*a(n, α, β, m) * ξ(n,k*r) * pifunc(n, θ) + b(n, α, β, m) * ξdiff(n,k*r) * τ(n, θ))
        for n in 1:N
    )
)

# Eq. 38
Htϕ(r, θ, ϕ) = E0/(k*r) * sqrt(ϵ/μ) * cos(ϕ) * (
    k * r * exp(-im * k * r * cos(θ)) +
    sum(
        im^(n+1) * (-1)^n * (2n+1)/(n*(n+1)) *
        (-im*a(n, α, β, m) * ξ(n,k*r) * τ(n, θ) + b(n, α, β, m) * ξdiff(n,k*r) * pifunc(n, θ))
        for n in 1:N
    )
)

# Eq. 40
S(r, θ, ϕ) = 1/2 * real(
    cos(θ) * (Etθ(r, θ, ϕ) * conj(Htϕ(r, θ, ϕ)) - Etϕ(r, θ, ϕ) * conj(Htθ(r, θ, ϕ))) -
    sin(θ) * (Etϕ(r, θ, ϕ) * conj(Htr(r, θ, ϕ)) - Etr(r, θ, ϕ) * conj(Htϕ(r, θ, ϕ)))
)



#==============================================================#
# # Eq. 14
# Er_s = E0 * cos(ϕ) * sum(
#     (im)^(n + 1) * (-1)^n * (2n + 1) / (n * (n + 1)) * 
#     a(n, α, β, m) * (ξdiff2(n, k*r) + ξ(n, k*r)) * Plm(cos(θ), n, 1)
#     for n in 1:N
# )

# # Eq. 15
# Eθ_s = E0 / (k * r) * cos(ϕ) * sum(
#     (im)^(n + 1) * (-1)^n * (2n + 1) / (n * (n + 1)) * 
#     (a(n, α, β, m) * ξdiff(n, k*r) * τ(n, θ) - im * b(n, α, β, m) * ξ(n, k*r) * pifunc(n, θ))
#     for n in 1:N
# )

# # Eq. 16
# Esϕ = -E0 / (k * r) * sin(ϕ) * sum(
#     im^(n + 1) * (-1)^n * (2n + 1) / (n * (n + 1)) *
#     (a(n, α, β, m) * ξdiff(n, k*r) * pifunc(n, θ) - im * b(n, α, β, m) * ξ(n, k*r) * τ(n, θ))
#     for n in 1:N
# )

# # Eq. 17
# Hsr = E0 * sqrt(ϵ / μ) * sin(ϕ) * sum(
#     im^(n + 1) * (-1)^n * (2n + 1) / (n * (n + 1)) *
#     b(n, α, β, m) * (ξdiff2(n, k*r) + ξdiff(n, k*r)) * Plm(cos(θ), n, 1)
#     for n in 1:N
# )

# # Eq. 18
# Hsθ = E0 / (k * r) * sqrt(ϵ / μ) * sin(ϕ) * sum(
#     im^(n + 1) * (-1)^n * (2n + 1) / (n * (n + 1)) *
#     (-im * a(n, α, β, m) * ξ(n, k * r) * pifunc(n, θ) +
#      b(n, α, β, m) * ξdiff(n, k * r) * τ(n, θ))
#     for n in 1:N
# )


# # Eq. 19
# Hsϕ = E0 / (k * r) * sqrt(ϵ / μ) * cos(ϕ) * sum(
#     im^(n + 1) * (-1)^n * (2n + 1) / (n * (n + 1)) *
#     (-im * a(n, α, β, m) * ξ(n, k * r) * τ(n, θ) +
#      b(n, α, β, m) * ξdiff(n, k * r) * pifunc(n, θ))
#     for n in 1:N
# )

# # # Eq. 26
# # # E_phi_TE(theta) = i * ω * μ / r * ∂U_TE / ∂θ
# # function Eϕ_TE(U_TE, θ_grid, r, ω, μ)
# #     dθ = θ_grid[2] - θ_grid[1]
# #     dUdθ = gradient(U_TE, dθ)  # または finite difference 等で ∂U_TE/∂θ を計算
# #     return im * ω * μ / r .* dUdθ
# # end

# # Eq. 27
# Ei_r = E0 * cos(ϕ) * sin(θ) * exp(-im * k * r * cos(θ))

# # Eq. 28
# Ei_θ = E0 * cos(ϕ) * cos(θ) * exp(-im * k * r * cos(θ))

# # Eq. 29
# Ei_ϕ = -E0 * sin(ϕ) * exp(-im * k * r * cos(θ))

# # Eq. 30
# Hi_r = E0 * sqrt(ε / μ) * sin(ϕ) * sin(θ) * exp(-im * k * r * cos(θ))

# # Eq. 31
# Hi_θ = E0 * sqrt(ε / μ) * sin(ϕ) * cos(θ) * exp(-im * k * r * cos(θ))

# # Eq. 32
# Hi_ϕ = E0 * sqrt(ε / μ) * cos(ϕ) * exp(-im * k * r * cos(θ))

# # Eq. 39 (ポインティングベクトル S の z成分を返す関数)
# S_z = 0.5 * real(Eθ * conj(Hϕ) - Eϕ * conj(Hθ))


#=====================================================================#
# === 散乱強度（S）を評価 ===
p = Progress(Ny, "Computing in parallel...")

result_arr = zeros(Float64, Ny, Nx)

Threads.@threads for j in 1:Ny
    for i in 1:Nx
        r = sqrt(x[i]^2 + y[j]^2 + z^2)
        θ = acos(z / r)
        ϕ = (x[i] == 0 && y[j] == 0) ? 0.0 : atan(y[j], x[i])

        result_arr[j, i] = real(S(r, θ, ϕ))
    end
    next!(p)
end

finish!(p)

# === 出力先の設定 ===
output_dir = "../data"  # ここで自由にフォルダ名を指定
mkpath(output_dir)  # フォルダがなければ自動作成

# 出力ファイルパス
output_file = joinpath(output_dir, "mie_isophote.dat")

# === ファイル保存処理 ===
open(output_file, "w") do io
    for j in 1:Ny
        for i in 1:Nx
            xval = x[i] * 1e6  # [m] → [μm]
            yval = y[j] * 1e6
            sval = result_arr[j, i]
            println(io, "$(xval) $(yval) $(sval)")
        end
    end
end



# # === プロット ===
# using Plots
# contour(
#     x .* 1e6, y .* 1e6, result_arr,         # [m] → [µm]
#     xlabel = "x [µm]",
#     ylabel = "y [µm]",
#     title = "Isophote behind 20-µm Transparent Particle (Fig.3)",
#     fill = true,
#     levels = 60,                         # 等高線を細かく
#     clim = (0.0, 0.002),                 # 明るい部分の saturate を防ぐ
#     aspect_ratio = :equal,         # 縦横1:1 (同義)
#     xlims = (0, 20),               # 明示的に範囲指定
#     ylims = (0, 20),
#     colorbar = true,
#     colorbar_title = "S [W/m²]"
# )
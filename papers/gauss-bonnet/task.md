# Gauss-Bonnet 格点收敛性测试：任务记录

本文档总结本次收敛性测试的背景、文章内容、使用的代码、已完成的计算、结果以及后续计算更大 $N$ 时需要知道的事项。目标是让后续接手的人可以直接在此基础上添加 $N=256,320,400$ 等算例，而不必重新梳理代码和远端流程。

---

## 1. 任务背景

文章 `papers/gauss-bonnet/main-v2.tex` 使用格点模拟研究 scalar Gauss-Bonnet (SGB) inflation 中的 ultra-slow-roll (USR) 阶段，并将格点功率谱用于 scalar-induced gravitational waves (SIGW) 和 PTA 拟合。

审稿意见 `papers/gauss-bonnet/report-1.txt` 的第 2 条指出：文章没有给出格点收敛性测试，需要检查

- 分辨率（$\Delta x$）
- 盒子大小（$L$）
- 时间步长（$\Delta t$）
- 初始/结束时间

至少要覆盖振幅最大、真正驱动文章结论的那些 $\alpha$。

本次只处理第 2 条。第 1 条（Bayesian 分析中的插值/外推）和第 3 条（SIGW 的非高斯性）不在本任务范围内。

本次任务先选取振幅最大的

$$
\alpha = 1.13412\times 10^7
$$

并完成了：

- $N=64$，$\Delta t=2.0$（本地）
- $N=128$，$\Delta t=2.0$（`reex-zerotier`）
- $N=300$，$\Delta t=2.0$（`reex-zerotier`）
- $N=64$，$\Delta t=1.0$（`reex-zerotier`）
- $N=400$，$\Delta t=2.0$（`reex-zerotier`，分辨率）
- $N=400$，$k_0=10^{-9}$ Hz（`reex-zerotier`，固定 $\Delta x$、$L=2L_{\rm base}$ 的**盒子/UV 对照**，2026-11-12 启动，进行中）

> 2026-11-12 补：新增 `box_n400_k0_1e-9` 作为 §4.2.1 的判别算例，见 §4.2.4；
> 同时补入文献 arXiv:2610.26669（Braglia–Franciolini–Ijaz–Peloso）对该机制的独立诊断与
> 绝热减除方案，见 §4.2.2。

N=200 的生产结果直接复用 `inflat/out/gauss_bonnet.bak/case1-final.output.npz`，没有重跑，也没有覆盖旧数据。

---

## 2. 文章与模型要点

### 2.1 SGB inflation

文章采用

$$
\mathcal L = \frac12 R - \frac12\partial_\mu\phi\partial^\mu\phi - V(\phi)
- \frac{\xi(\phi)}{16}\mathcal L_{\rm GB},
$$

$$
\mathcal L_{\rm GB}=R^2-4R_{\mu\nu}R^{\mu\nu}+R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}.
$$

背景方程为

$$
3H^2=\frac12\dot\phi^2+V(\phi)+\frac32 H^3\xi'(\phi)\dot\phi,
$$

$$
\ddot\phi+3H\dot\phi+\frac{\partial V_{\rm eff}}{\partial\phi}(\phi,H)=0,
$$

$$
V_{\rm eff}=V(\phi)+\frac32 H^2(\dot H+H^2)\xi(\phi).
$$

文章取

$$
\xi(\phi)=\alpha\tanh[\xi_1(\phi-\phi_c)],
\qquad
V(\phi)=\Lambda^4\left(1+\cos\frac{\phi}{f}\right),
$$

其中 $\phi_c$ 取非平庸不动点 $\phi_*$。在 $\phi_c$ 附近背景进入 USR，功率谱在特定频率范围被放大。

文章 Table I 给出的生产参数为

| 参数 | 值 |
|---|---:|
| $f$ | 7.0 |
| $\Lambda$ | 0.0065 |
| $\xi_1$ | 9.6 |
| $\phi_c$ | 30.0 |

> **注意：代码中的参数顺序**
>
> 生产代码 `inflat/src/bin/gauss_bonnet.rs` 的 `param_set1` 形参顺序是
> `(phi_c, xi0, xi1)`，生产调用是
> `param_set1(9.6, xi0, 30.0)`。
> 因此代码实际使用的是 `phi_c=9.6`、`xi1=30.0`，与文章 Table I 的
> $\xi_1=9.6$、$\phi_c=30.0$ 表面上是交换的。
>
> 这可能是文章表格标签或代码形参命名的问题。**新算例必须沿用生产调用
> `param_set1(9.6, xi0, 30.0)`**，否则无法与现有结果比较。若以后要修正，
> 需要同时核对文章和全部已算结果。

### 2.2 格点方法

空间离散为 $N^3$ 个格点，物理盒子长度 $L$，格距

$$
\Delta x=\frac{L}{N}.
$$

动量空间为

$$
\vec k=\frac{2\pi\vec n}{L}.
$$

格点 Laplacian 给出修正色散关系

$$
k_{\rm eff}=\frac{2}{\Delta x}
\sqrt{\sum_i\sin^2\frac{\pi n_i}{N}}.
$$

频率范围为

$$
k_{\rm eff,min}\approx \frac{2\pi}{L},
\qquad
k_{\rm eff,max}=\frac{2\sqrt3}{\Delta x},
\qquad
\frac{k_{\rm max}}{k_{\rm min}}=\frac{\sqrt3 N}{\pi}.
$$

生产设置：

- 起始时间：$k_{\rm min}=1.3\,aH$
- 结束时间：$k_{\rm max}=aH/200$
- $N=200$

### 2.3 功率谱与 $\zeta$ 的提取

格点谱与连续谱的关系为

$$
\mathcal P_f(k)
=
\frac{k_{\rm eff}^3}{2\pi^2}
\left(\frac{\Delta x}{N}\right)^3
\tilde\Delta_f(\vec k).
$$

初始条件使用 Bunch-Davies 真空和 truncated Wigner approximation。最后使用 $\delta N$ 形式

$$
\zeta(\vec x)=\ln a(\vec x)-\overline{\ln a(\vec x)}
$$

从每个格点的最终 $a$ 提取曲率扰动。

### 2.4 当前文章结果与 $\alpha$ 网格

文章在

$$
\alpha=1.133\times10^7+a_i\times10^3,
$$

$$
a_i=\{1,2,\dots,10,10.5,11,11.1,11.2\}
$$

上计算了 14 个 $\alpha$。$\alpha-\alpha_0$ 越大，功率谱峰值越高；峰值达到 $10^{-2}$ 量级时格点修正变得显著。

本次收敛性测试选择最大振幅的

$$
\alpha=1.13412\times10^7
\quad(\alpha-\alpha_0=10200).
$$

baseline 对应 `case1-final.output.npz` 中 `params == 1.13412e7` 的行。注意 `np.isclose` 的默认相对容差会把 `1.13411e7` 也选进来，**必须用绝对容差**：

```python
idx = np.where(np.abs(params - 1.13412e7) < 1.0)[0]
```

---

## 3. 代码结构

### 3.1 文件清单

| 文件 | 作用 |
|---|---|
| `inflat/src/bin/gauss_bonnet.rs` | 原生产程序，包含 14 个 $\alpha$ 的扫描和旧输出逻辑。**不要随意运行，避免覆盖旧数据。** |
| `inflat/src/scalar.rs` | `LatticeInput`、`LatticeOutputData`、`from_background_and_k_normalized`、`run`、`spectrum`、$\delta N$ 提取等核心库。 |
| `inflat/src/gauss_bonnet.rs` | SGB 背景、场演化、RK4、$V_{\rm eff}$、度规扰动等。 |
| `inflat/src/igw.rs` | SIGW 积分 `sigw_2_spectrum`。 |
| `inflat/src/util.rs` | `ParamRange`、插值、bincode 缓存等。 |
| `inflat/src/bin/gauss_bonnet_convergence.rs` | 本次新增的收敛测试程序。 |
| `inflat/run_convergence_remote.sh` | 远端顺序运行脚本。 |
| `papers/gauss-bonnet/plot_convergence.py` | 从 NPZ 数据生成论文图。 |
| `papers/gauss-bonnet/fig/convergence-resolution.pdf/png` | $N=64,128$ vs baseline 图。 |
| `papers/gauss-bonnet/fig/convergence-dt.pdf/png` | $N=64$ $\Delta t=1$ vs $2$ 图。 |

项目规则：

- `inflat/out/` 不在 git 中，**不要随便删除**。
- `inflat/out/gauss_bonnet.bak/` 是旧结果备份，**不要动**。
- 实际可执行程序在 `inflat/src/bin/`，不是 `inflat/bin/`。

### 3.2 核心参数化：`LatticeInput`

`inflat/src/scalar.rs` 中的

```rust
LatticeInput::<3, GaussBonnetBackgroundState>::from_background_and_k_normalized(
    background,
    input,
    k_min,
    subhorizon_tolerance,
    superhorizon_tolerance,
    lattice_size,
    dt,
)
```

实际接收：

- `k_min`：物理 $k_{\min}$（在代码单位中，通过 `k0 / k_star` 传入）
- `subhorizon_tolerance`：起始条件 $k_{\min}=$ `tolerance` $\times aH$
- `superhorizon_tolerance`：结束条件 $k_{\max}=aH/$ `tolerance`
- `lattice_size`：$N$
- `dt`：时间步长（代码时间，起始 $a=1$）

内部：

```text
starting_horizon = k_min / subhorizon_tolerance
end_k = k_min * sqrt(3) / pi * N
end_horizon = end_k * superhorizon_tolerance

start_state      = first background state with aH >= starting_horizon
start_k_state    = first background state with aH >= k_min
end_state        = first background state with aH >= end_horizon

normalized_start_k = start_k_state.aH / start_state.a
dx = 2*pi / normalized_start_k / N
k_unit = k_min / normalized_start_k
L = N * dx
```

因此：

- 固定 `k_min`、改变 $N$：$L$ 固定，$\Delta x=L/N$ 改变，$k_{\max}\propto N$。这是分辨率测试。
- 固定 $N$、只改 `k_min`：$L$ 和 $\Delta x$ 同时按比例改变，只是平移频率窗口。
- 固定 $\Delta x$、改变 $L$：需要保持 $N k_{\min}$ 不变（因为 $N k_{\min}\propto k_{\max}$），即 $N\propto 1/k_{\min}$。这是干净的盒子大小测试。

`k_min()`、`k_max()`、`orignal_final_a()` 可用于报告实际范围。

### 3.3 生产程序的 baseline 设置

`inflat/src/bin/gauss_bonnet.rs` 的 `run_scan_case1` 使用：

```text
k0      = 2e-9
dt      = 2.0
size    = 200
sub     = 1.3
super   = 200
seed    = 10
```

对应的 baseline：

| 量 | 值 |
|---|---|
| $k_{\min}$ 目标 | $2\times10^{-9}\,\mathrm{Hz}$ |
| 首个 bin 中心 | $\approx 2.55\times10^{-9}\,\mathrm{Hz}$（shell binning 导致） |
| $k_{\max}$ | $\approx 2.205\times10^{-7}\,\mathrm{Hz}$ |
| $L$ | $\approx 4.86\,\mathrm{pc}$ |
| $\Delta x$ | $\approx 0.0243\,\mathrm{pc}$ |

> 首个 bin 的中心不是 $k_{\min}$，因为 `spectrum_with_scratch` 按
> `round(sqrt(nx^2+ny^2+nz^2))` 分 bin，最低 shell 包含多个不同 $k_{\rm eff}$ 的模式。
> 不要用首个 bin 反推 $k_{\min}$。

### 3.4 收敛程序 `gauss_bonnet_convergence.rs`

#### 用途

独立于生产程序，避免覆盖旧数据。默认只打印计划；显式 `--run` 才计算。

#### 当前配置表

| name | group | N | $k_0$ [Hz] | dt | sub | super | seed |
|---|---|---:|---:|---:|---:|---:|---:|
| `n64` | resolution | 64 | 2e-9 | 2.0 | 1.3 | 200 | 10 |
| `n128` | resolution | 128 | 2e-9 | 2.0 | 1.3 | 200 | 10 |
| `n300` | resolution | 300 | 2e-9 | 2.0 | 1.3 | 200 | 10 |
| `n400` | resolution | 400 | 2e-9 | 2.0 | 1.3 | 200 | 10 |
| `n512` | resolution | 512 | 2e-9 | 2.0 | 1.3 | 200 | 10 |
| `box_n128_k0_3p125e-9` | box | 128 | 3.125e-9 | 2.0 | 1.3 | 200 | 10 |
| `box_n64_k0_6p25e-9` | box | 64 | 6.25e-9 | 2.0 | 1.3 | 200 | 10 |
| `box_n400_k0_1e-9` | box | 400 | 1e-9 | 2.0 | 1.3 | 200 | 10 |
| `dt1_n64` | timestep | 64 | 2e-9 | 1.0 | 1.3 | 200 | 10 |
| `dt4_n128` | timestep | 128 | 2e-9 | 4.0 | 1.3 | 200 | 10 |
| `dt8_n128` | timestep | 128 | 2e-9 | 8.0 | 1.3 | 200 | 10 |
| `sub2_n128` | times | 128 | 2e-9 | 2.0 | 2.0 | 200 | 10 |
| `super500_n128` | times | 128 | 2e-9 | 2.0 | 1.3 | 500 | 10 |
| `seed11_n128` | seed | 128 | 2e-9 | 2.0 | 1.3 | 200 | 11 |
| `seed12_n128` | seed | 128 | 2e-9 | 2.0 | 1.3 | 200 | 12 |

`box_n128_k0_3p125e-9` 和 `box_n64_k0_6p25e-9` 保持 $Nk_0=4\times10^{-7}$，即固定 $\Delta x$，盒子分别只有 baseline 的 $1/1.5625$ 和 $1/3.125$。它们测试的是更小的盒子，不是更大的盒子。

`box_n400_k0_1e-9` 同样保持 $Nk_0=4\times10^{-7}$，但方向相反：$N=400$、$k_0=10^{-9}$ Hz，
给出 $L=9.720$ pc $=2L_{\rm base}$、$\Delta x=0.02430$ pc $=\Delta x_{\rm base}$。这是目前唯一一个
"更大盒子 + 与 baseline 相同 $\Delta x$" 的算例，用途见 §4.2.4。

`n512` 已加入配置但尚未运行（外推约 9 天）。

#### CLI

```bash
# 只看计划
../target/release/gauss_bonnet_convergence --list

# 默认只跑 resolution 组（n64, n128）
../target/release/gauss_bonnet_convergence --run

# 指定单个算例
../target/release/gauss_bonnet_convergence --run --config n64
../target/release/gauss_bonnet_convergence --run --config n128

# 按组
../target/release/gauss_bonnet_convergence --run --group box
../target/release/gauss_bonnet_convergence --run --group timestep
../target/release/gauss_bonnet_convergence --run --group times
../target/release/gauss_bonnet_convergence --run --group seed

# 已有 bincode，只重新出图/汇总
../target/release/gauss_bonnet_convergence --plot-only --config n64 --config n128 --config dt1_n64
```

其他参数：

- `--xi0 <value>`：默认 `1.13412e7`
- `--baseline <path>`：默认 `inflat/out/gauss_bonnet.bak/case1-final.output.npz`
- `--group all`
- `--help`

#### 输出

输出目录：

```text
inflat/out/gauss_bonnet_convergence/alpha_1.13412e7/
```

每个算例：

| 文件 | 内容 |
|---|---|
| `<case>.lattice.bincode` | `LatticeOutputData`，完整原始结果，可续算 |
| `<case>.lattice.checkpoint.bincode` | 每 600 s 保存的 checkpoint |
| `<case>.lattice.zeta.checkpoint.bincode` | $\delta N$ 重建过程中的 checkpoint |
| `<case>.final_field.npz` | 最终 $\phi(\vec x)$, $\dot\phi(\vec x)$, $\zeta(\vec x)$, $a$, $\dot a$ |
| `<case>.summary.npz` | $k$, $\mathcal P_\zeta$, $\mathcal P_{\zeta,\rm lin}$, SIGW, 中途谱, 降采样背景量 |
| `<case>.sigw.bincode` | SIGW 积分结果（本任务不使用） |

`summary.npz` 的键：

```text
k, p, p_lin, sigw, spectrum_k, spectrum_ln_a, spectrums,
measurable_a, measurable_v_a, measurable_phi, measurable_v_phi,
measurable_metric_pert_a, measurable_metric_pert_b,
measurable_hubble_constraint, final_a, final_v_a,
N, k0, dt, subhorizon_tolerance, superhorizon_tolerance, seed
```

注意：

- 总演化区间被分成 `SPECTRUM_COUNT = 10` 段，按段保存中途功率谱（外加初始谱）。
- `measurables` 在 bincode 中仍是每一步都保存；`summary.npz` 里降采样到最多约 1 万点。
- 最终格点场在 NPZ 中是 $(N,N,N)$ 数组，C-order（最后一个索引最快）。
- `.lattice.bincode` 和 `.final_field.npz` 都会保存最终场，磁盘占用约为最终场的两倍。

#### 相对差异与比较窗口

Rust 和 Python 都使用：

```text
COMPARISON_K_MIN        = 1e-8
COMPARISON_K_MAX_FRACTION = 0.7
```

即相对差异只在

$$
k\in[10^{-8},\,0.7\,k_{\max}]
$$

内计算，并去掉首尾 bin。更高 $k$ 受 UV 截断影响，只在上面板显示并标为 `UV-affected`。

`rel_curve` 内部会：

1. 去掉首尾 bin；
2. 在公共 $k$ 区间内取 300 个对数点；
3. 对 $\mathcal P_\zeta$ 做 log-log 插值；
4. 返回 `a/b - 1`。

#### baseline 读取

`load_baseline` 使用 `ndarray_npy::NpzReader` 读取：

- `params`
- `linear_k`, `linear_spectrums`
- `lat_k`, `lat_spectrums`
- `lat_sigw_omega`

选择行时使用

```rust
(p - xi0).abs() < 1.0
```

而不是 `np.isclose`。`k_star` 由

```rust
let k_star = linear_k[0] / 1e-2;
```

得到，因为 `linear_k` 是从 `ParamRange::new(1e-2, 1e25, 1000)` 乘以 `k_star` 生成的。如果 baseline 不存在，程序会用线性谱 pivot 重新计算 `k_star`，但不会重跑 N=200 格子。

### 3.5 绘图脚本 `plot_convergence.py`

```bash
python3 papers/gauss-bonnet/plot_convergence.py
```

读取：

- `inflat/out/gauss_bonnet.bak/case1-final.output.npz` 的 $\alpha=1.13412e7$ 行
- `inflat/out/gauss_bonnet_convergence/alpha_1.13412e7/n64.summary.npz`
- `.../n128.summary.npz`
- `.../dt1_n64.summary.npz`

生成：

- `papers/gauss-bonnet/fig/convergence-resolution.pdf/png`
- `papers/gauss-bonnet/fig/convergence-dt.pdf/png`

图中：

- 上面板：$\mathcal P_\zeta(k)$，线性理论虚线，baseline 黑色，N=128 红色，N=64 蓝色。
- 下面板：相对 baseline 或相对 $\Delta t=2$ 的差异，百分比坐标。
- 灰色 `UV-affected` 区域：$k>0.7k_{\max}$。
- 首尾 bin 去掉。

脚本中同样使用绝对容差选 baseline 行，避免 `np.isclose` 误选。

### 3.6 远端脚本

`inflat/run_convergence_remote.sh`：

```sh
#!/bin/sh
set -e

export PATH="/usr/local/bin:$HOME/.cargo/bin:$PATH"
cd "$HOME/computations-cfy"

mkdir -p out/gauss_bonnet_convergence

# 服务器网络慢，所有 crate 已在本地 cargo cache；plotly 手动上传
cargo build --release --offline --bin gauss_bonnet_convergence \
    > out/gauss_bonnet_convergence/build.log 2>&1

BIN=./target/release/gauss_bonnet_convergence
BASE=baseline/case1-final.output.npz

# 严格顺序运行，不并发
$BIN --baseline "$BASE" --run --config dt1_n64 \
    > out/gauss_bonnet_convergence/dt1_n64.run.log 2>&1
$BIN --baseline "$BASE" --run --config n128 \
    > out/gauss_bonnet_convergence/n128.run.log 2>&1

# 两者都完成后，再做一次后处理
$BIN --baseline "$BASE" --plot-only --group timestep \
    > out/gauss_bonnet_convergence/timestep.plot.log 2>&1
```

远端路径：

```text
reex-zerotier:~/computations-cfy/
```

baseline 单独放在：

```text
~/computations-cfy/baseline/case1-final.output.npz
```

服务器缺 `plotly-0.12.1.crate` 时，从本地 cargo cache 上传到：

```text
/root/.cargo/registry/cache/index.crates.io-1949cf8c6b5b557f/plotly-0.12.1.crate
```

然后使用 `--offline` 构建。

`screen`：

```bash
ssh reex-zerotier 'screen -ls'
ssh -t reex-zerotier 'screen -r gb_conv'
# detach: Ctrl-A D
```

日志：

```text
~/computations-cfy/out/gauss_bonnet_convergence/build.log
~/computations-cfy/out/gauss_bonnet_convergence/dt1_n64.run.log
~/computations-cfy/out/gauss_bonnet_convergence/n128.run.log
~/computations-cfy/out/gauss_bonnet_convergence/timestep.plot.log
```

> 服务器上的 `ssh_connect` 工具无法处理 passphrase；本次全部使用 shell `ssh` / `rsync`，并以非沙箱方式访问 ssh-agent。

---

## 4. 已完成的算例与结果

### 4.1 算例表

| 算例 | 运行位置 | N | $k_0$ [Hz] | dt | sub | super | seed | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 现有 `case1-final.output.npz` | 200 | 2e-9 | 2.0 | 1.3 | 200 | 10 | 复用 |
| `n64` | 本地 | 64 | 2e-9 | 2.0 | 1.3 | 200 | 10 | 作废，待重算 |
| `n128` | `reex-zerotier` | 128 | 2e-9 | 2.0 | 1.3 | 200 | 10 | 完成 |
| `n300` | `reex-zerotier` | 300 | 2e-9 | 2.0 | 1.3 | 200 | 10 | 完成 |
| `n400` | `reex-zerotier` | 400 | 2e-9 | 2.0 | 1.3 | 200 | 10 | 完成（未收敛，见 4.2） |
| `n400_super500` | `reex-zerotier` | 400 | 2e-9 | 2.0 | 1.3 | 500 | 10 | 完成（仅远端） |
| `dt1_n64` | `reex-zerotier` | 64 | 2e-9 | 1.0 | 1.3 | 200 | 10 | 完成 |
| `box_n400_k0_1e-9` | `reex-zerotier` | 400 | 1e-9 | 2.0 | 1.3 | 200 | 10 | **运行中**（2026-11-12 启动，见 4.2.3/4.2.4） |

> `n64` 使用旧 $\delta N$ 算法，数据已删除，需重新计算；表中 n64 相关数值作废。

本次实测耗时（仅供粗略参考，实际以日志里的 ETA 为准）：

| 算例 | 机器 | 大致墙钟时间 |
|---|---|---:|
| `n64` (dt=2) | 本地 Mac | 约 2.5 h |
| `dt1_n64` | `reex-zerotier` | 约 3.3 h |
| `n128` | `reex-zerotier` | 约 4.5 h |
| `n300` | `reex-zerotier` | 约 43 h |
| `n400` | `reex-zerotier` | 约 91 h（约 3.8 天） |
| `n400_super500` | `reex-zerotier` | 约 100 h |
| `box_n400_k0_1e-9` | `reex-zerotier` | 约 92 h（日志 ETA，2026-11-12 启动） |

N=64 的开销有较多固定/并行效率因素，不建议直接用 $N^3$ 外推到更大 N；提交后先看 `[lattice] eta remaining` 的实际估计。

峰值信息（$\alpha=1.13412\times10^7$）：

| 算例 | $k_{\rm peak}$ [Hz] | $\mathcal P_{\zeta,\rm peak}$ | $\mathcal P_{\rm peak}/\mathcal P_{\rm lin}$ |
|---|---:|---:|---:|
| baseline N=200 | $\approx 2.37\times10^{-8}$ | $\approx 8.00\times10^{-2}$ | $\approx 1.38$ |
| `n64` (dt=2) | $2.503\times10^{-8}$ | $8.208\times10^{-2}$ | $1.434$ |
| `dt1_n64` | $2.503\times10^{-8}$ | $8.060\times10^{-2}$ | $1.408$ |
| `n128` | $2.385\times10^{-8}$ | $7.654\times10^{-2}$ | $1.323$ |
| `n300` | $2.805\times10^{-8}$ | $7.739\times10^{-2}$ | $1.450$ |
| `n400` | $2.017\times10^{-8}$ | $6.650\times10^{-2}$ | $1.217$ |
| `n400_super500` | $2.017\times10^{-8}$ | $6.499\times10^{-2}$ | — |
| `box_n400_k0_1e-9` | 运行中（预计 $\approx1.0\times10^{-8}$） | — | — |

`n400_super500`（super 从 200 提到 500，末态 $\ln a$ 从 10.98 到 11.91）与 `n400` 相比
峰值只变 $-2.3\%$、**峰位完全不动**（$2.017\times10^{-8}$ Hz）。这独立确认了
"N=400 偏低不是演化时间不够"。注意该算例只在远端，本地只下载了 `summary.npz`。

### 4.2 收敛指标

共同窗口：

$$
k\in[10^{-8},\,4.931\times10^{-8}]\,\mathrm{Hz}
=
[10^{-8},\,0.7\,k_{\max}(N=64)].
$$

| 比较 | 中位 $\lvert\Delta P/P\rvert$ | 最大 $\lvert\Delta P/P\rvert$ |
|---|---:|---:|
| N=64 vs baseline | 8.58% | 42.7% |
| N=128 vs baseline | 2.11% | 8.62% |
| N=300 vs baseline | 2.81% | 10.79% |
| N=400 vs baseline | 10.41% | 17.35% |
| N=64 dt=1 vs dt=2 | 2.33% | 11.99% |

解释：

- N=128 与 N=300 相对 baseline 的中位差异都在 2–3%，明显优于 N=64 的约 9%。
- N=300 的中位/最大差异没有单调优于 N=128（2.81% vs 2.11%），但两者在该窗口内已经非常接近；剩余差异与随机实现/其他系统误差同量级，不能简单解读为 N=300 更差。
- N=64 的偏差在窗口高 $k$ 端明显增大。
- N=64 的 $\Delta t=1$ 与 $\Delta t=2$ 在大部分区间只差 1–3%，仅在接近 UV 截断处增大到约 12%，说明 $\Delta t=2$ 对 N=64 已基本收敛。
- **N=400 没有继续收敛**：共同窗口内中位 10.41%、最大 17.35%，明显差于 N=128/N=300 的 2–3%；峰值也最低（$6.65\times10^{-2}$），峰位移到 $2.02\times10^{-8}$ Hz。
- N=400 的演化时间反而最长（末态 $\ln a=10.98$，N=300 为 10.69，N=128 为 9.84），所以整体偏低不是「演化时间不够」造成的；更可能是随机实现或尚未识别的系统误差。
- `n400_super500`（2026-11-12 补齐）把 $\zeta$ 冻结判据从 $k_{\max}=aH/200$ 放宽到 $aH/500$、末态
  $\ln a$ 推到 11.91，峰值只再降 2.3%、峰位不动，进一步排除了"末态未冻结"。详见 §4.1。
- 若把 $\rho_{\rm vac}/\rho_{\rm tot}\propto N^4$ 的线性响应当作模型，用 N=300/400 两点定系数会给出
  $P_\infty\simeq8.24\times10^{-2}$、$c\simeq7.6\times10^{-12}$（$P(N)=P_\infty(1-cN^4)$），
  则 N=200 预言 $8.14\times10^{-2}$（实测 8.00，差 1.8%，可接受）、
  **N=128 预言 $8.23\times10^{-2}$（实测 7.65，差 7%，不可接受）**。
  即 $N^4$ 通道能解释 N=300→400 的跳变，但解释不了 N=128 已经偏低 3–4%。
  由于四个算例 seed 都是 10（同一随机实现，不能提供散度），这说明至少还有第二条通道，
  或者响应不是线性的。详见 §4.2.3。
- 用 `plot_evolution.py` 检查 `spectrums`：窗口内模式在 $\ln a\approx1.9$–$3.5$ 出视界，末态 $k/(aH)\sim10^{-4}$–$10^{-3}$；但线性化估计在最后两三个 e-fold 仍上漂约 20%（N=300 同样如此），因此不能仅凭中期快照证明 $\zeta$ 已冻结。
- 图中分辨率对比已加入 N=300 与 N=400；为了不让 N=400 更长的尾部压缩比较窗口，上面板横轴截到 $2\times$ 共同窗口（约 $9.9\times10^{-8}$ Hz），灰色 UV 区域只展示到该处。

#### 4.2.1 N=400 偏低的候选机制：未重整化的 UV 真空能改变初始背景

理论上，当前固定 $L$、增大 $N$ 的分辨率测试同时也在提高初始 BD 真空的 UV 截断：

$$
\Lambda_{\rm UV}\sim k_{\max}\propto N.
$$

TWA/Wigner 真空包含每个模式的零点能。对高频模式 $E_{\mathbf k}\simeq k_{\rm eff}/2$，所以未减除的真空能密度具有标准的三维 quartic UV 发散：

$$
\rho_{\rm vac}(\Lambda_{\rm UV})
\simeq \frac12\int^{|k|<\Lambda_{\rm UV}}
\frac{d^3k}{(2\pi)^3}\,k
=\frac{\Lambda_{\rm UV}^4}{16\pi^2}
\propto N^4.
$$

代码中的 `populate_noise` 按 BD 方差生成 $\phi$ 和 $\dot\phi$ 噪声，随后 `solve_hubble` 将包含这些噪声的梯度、动能和势能代入完整 SGB Hamilton 约束，重新确定初始 $H$。这里没有相应的真空减除或随截断变化的反项。因此，不同 $N$ 不只是空间离散精度不同，也对应不同的裸初始总能量和背景膨胀历史。

现有日志支持这一渠道确实存在：

- N=300 和 N=400 的初始梯度量分别为 $9.42853\times10^{-12}$ 和 $2.97906\times10^{-11}$，比值为 $3.16$，几乎等于 $(400/300)^4=3.1605$。
- 无噪声背景为 $H_{\rm bg}=2.7006884\times10^{-5}$；N=300 和 N=400 的初始 $H$ 分别为 $2.7064997\times10^{-5}$ 和 $2.7190104\times10^{-5}$，即分别偏高约 $0.215\%$ 和 $0.679\%$。在当前背景仍占主导时，$\delta H/H\propto\rho_{\rm vac}\propto N^4$；若继续取裸 $N\to\infty$，真空项最终占主导，$H_{\rm ini}$ 会渐近按 $N^2$ 发散。

这可能通过敏感的 USR 背景轨迹造成宽频段的谱振幅和峰位偏移，因此是 N=400 整体偏低的一个候选系统误差。但目前尚未证明它能产生约 10% 的压低：真空梯度能会红移，且仅凭初始 $H$ 增大不能确定最终谱变化的方向和大小。需要区分的是，**没有有限连续极限的是当前未重整化的裸 $H_{\rm ini}(N)$，不必然是重整化后的物理模型或 IR 功率谱**。在控制这一效应前，N=400 的结果不能被视为纯粹的空间分辨率收敛测试。

#### 4.2.2 外部证据：arXiv:2610.26669 的绝热减除与格点对账

> Braglia, Franciolini, Ijaz, Peloso, *Mean-Field Backreaction in Single-Field Inflation*,
> arXiv:2610.26669（2026-10-19，CERN-TH-2026-239）。作者之一 Franciolini 也是 §3.4
> 所用格点方法（Caravano–Franciolini–Renaux-Petel, arXiv:2410.23942）的作者，
> 因此这是格点这一派对自己结果的修正，而不是外部质疑。

文章内容（与本文直接相关的部分）：

1. 在 decoupling limit 下做 homogeneous（mean-field）backreaction，1-loop 阶保留
   $\rho_{\delta\phi},p_{\delta\phi}$ 对 $H^2$、$\dot H$ 和 Klein–Gordon 的反馈；
2. 指出裸 Bunch–Davies 真空在格点上**没有被减除**：裸被积函数在 UV 端
   $|B_\phi|\propto k^2$、$B_H\propto k^4$，因此 $\int d\ln k$ 出的真空能
   $\propto\Lambda_{\rm UV}^4$，与截断强相关；
3. 给出标准修法：WKB/绝热减除（adiabatic subtraction）到 2 阶与 4 阶，
   显式 counterterm 见其式 (3.10)、(3.17)–(3.18)，并加 $S_0,S_2,S_4$ filter
   避免在 WKB 失效区引入假 IR 贡献；
4. **关键结论**：不做减除的 mean-field 计算可以高精度重现 arXiv:2410.23942 的
   三组 USR 格点谱，但需要把 effective cutoff 从名义的 $45k_{\rm in}$ 调到
   $16.6k_{\rm in}$（同一个值对三组都对上）；减除之后 USR 的 backreaction 掉到
   $O(10^{-3})$（$P_{\zeta,\max}\le6\times10^{-3}$ 时峰值只变 0.03–0.17%）。
   即：那批格点里看到的 backreaction 有相当一部分来自未减除的 UV 真空。

对我们有用的三个数字/事实：

- **符号**：裸（未减除）backreaction 在 USR 里压制峰值，与 N=400 偏低方向一致；
- **量级**：其 Fig. 5 给出裸格点的峰值压制随振幅增长
  $P_{\zeta,\max}^{\rm tree}=10^{-4},10^{-3},10^{-2},10^{-1},1$
  对应 $2.0\%,6.0\%,17.4\%,42.3\%,75.0\%$。本文 $\mathcal P_{\zeta,\rm peak}\approx8\times10^{-2}$
  落在最后两档之间，对数插值约 35–40%，所以 N=400 的 10–17% 在这个机制的量级之内；
- **陷阱**：用连续估计 $\Lambda_{\rm UV}^4/16\pi^2$ 去对格点，cutoff 必须缩到
  $16.6k_{\rm in}$ 才吻合（$\Lambda^4$ 上差约 54 倍）。所以做减除时**必须用格点求和**
  $\frac{1}{2V}\sum_{\mathbf k}\omega_{\mathbf k}$，不能用解析式。

不能直接搬过来的地方：该文是 GR + 单场 + decoupling limit + 反推的 $\eta_H$ 三段势，
没有 $\xi(\phi)\mathcal G$ 项，也没有做 cutoff 扫描（只做了"名义 vs 有效 cutoff"的对账），
并且明确说减除后的有限部分是 scheme 依赖的（量级可差 $O(1)$；$d^3V/d\phi^3$ 大的变体在
最大振幅下残余可达 $-20.2\%$ 与 $+36.2\%$，符号可正可负）。

引文措辞建议（写文章时）：不要说"标准做法是重整化"；说
"lattice USR simulations without adiabatic subtraction are known to be UV-cutoff
dependent [Braglia et al. 2026]；我们用 $N$ 扫描和固定 $\Delta x$ 的盒子对照显式检验这一点"。

#### 4.2.3 本代码的参数化事实：真空能占比只由 $N$ 决定

2026-11-12 核对 `LatticeInput::from_background_and_k_normalized` 后发现一个此前没注意到的
结构性事实。由

```text
dx       = 2*pi / (sub*H*N)          // H 取起始态的 H
Lambda   = 2*sqrt(3)/dx = 0.5513*sub*N*H
```

可得

$$
\frac{\Lambda_{\rm UV}}{H}=0.5513\,{\rm sub}\,N ,
$$

**与 $L$ 无关，与 $k_0$ 只在 $H$ 随场值缓慢变化的精度内无关**（$k_0$ 主要改变盒子与 $k$ 轴的
标号；实测 `n400` 与 `box_n400_k0_1e-9` 的 $\Delta x$ 差 0.44%，即源于起始 $H$ 的 0.44% 差别）。
由于 USR 前后 $H\simeq2.70\times10^{-5}$ 几乎不变，真空能占比只随 ${\rm sub}\cdot N$ 变化。
用现有日志标定（`sub=1.3`）：

$$
\frac{\rho_{\rm vac}}{\rho_{\rm tot}}\simeq 5.3\times10^{-13}\,N^4,
\qquad
\frac{\delta H}{H}\simeq\frac12\frac{\rho_{\rm vac}}{\rho_{\rm tot}} .
$$

| $N$ | $\langle(\nabla\phi)^2\rangle/a^2$ | $\rho_{\rm vac}/\rho_{\rm tot}$ | $\delta H/H$ |
|---:|---:|---:|---:|
| 64 | $1.955\times10^{-14}$ | $8.9\times10^{-6}$ | $4.5\times10^{-6}$ |
| 128 | $3.128\times10^{-13}$ | $1.43\times10^{-4}$ | $7.1\times10^{-5}$ |
| 200（外推） | $1.86\times10^{-12}$ | $8.5\times10^{-4}$ | $4.2\times10^{-4}$ |
| 300 | $9.429\times10^{-12}$ | $4.31\times10^{-3}$ | $2.15\times10^{-3}$ |
| 400 | $2.979\times10^{-11}$ | $1.36\times10^{-2}$ | $6.78\times10^{-3}$ |

比值实测为 16.00 / 30.14 / 3.160，对应 $(N_2/N_1)^4=$ 16.00 / 30.20 / 3.160，
即 $N^4$ 精确成立（0.2% 以内）。2026-11-12 启动的 `box_n400_k0_1e-9` 也验证了这一点：
它的打印值是 $3.032\times10^{-11}$，与 `n400` 的 $2.979\times10^{-11}$ 只差 1.8%
（来自起始 $H$ 的 0.4% 差别），**因为 $N$ 没变**。

三条推论（很重要，会改变测试设计）：

1. **任何"固定 $\Delta x$、改 $L$"的 box 测试都不能降低真空能占比**——$N$ 没变，
   $\Lambda/H$ 就没变。所以 lit-survey 里"方案 C（固定物理 UV 截断）"的原始设想不成立：
   它固定的是 $\rho_{\rm vac}$ 的绝对值，不是 $\rho_{\rm vac}/\rho_{\rm tot}$。
2. 想降低占比只能减小 $N$ 或减小 `sub`；而 `sub` 是初始条件 $k_{\min}={\rm sub}\cdot aH$ 的
   容差，必须 $>1$，减小它意味着最低 $k$ 在起始时已经在视界外，BB 初始条件不成立。
   所以**在物理可接受的参数范围内，$\rho_{\rm vac}/\rho_{\rm tot}\propto N^4$ 无法回避**。
3. 因此 **$N$ 扫描本身无法把"空间分辨率"和"零点能"分开**，唯一干净的解法是显式减除
   （lit-survey 的方案 A），或者做一个把 $\Lambda_{\rm UV}$ 与 $N$ 解耦的改动。

#### 4.2.4 `box_n400_k0_1e-9` 能判别什么、不能判别什么

`box_n400_k0_1e-9` 的实测参数：$N=400$、$L=9.720$ pc（$2L_{\rm base}$）、
$\Delta x=0.02430$ pc（$=\Delta x_{\rm base}$）、$\Lambda_{\rm UV}=2.205\times10^{-7}$ Hz
（$=\Lambda_{\rm UV,base}$）、起始 $H=2.71265\times10^{-5}$、峰位预计在
$k\simeq1.0\times10^{-8}$ Hz。

三者对照：

| 算例 | $\Delta x$ | $L$ | $\Lambda/H$ | $\rho_{\rm vac}/\rho_{\rm tot}$ |
|---|---:|---:|---:|---:|
| baseline N=200 | 0.0243 | 4.86 | 143 | $8.5\times10^{-4}$ |
| `n400` | 0.01215 | 4.86 | 287 | $1.36\times10^{-2}$ |
| `box_n400_k0_1e-9` | 0.0243 | 9.72 | 287 | $1.36\times10^{-2}$ |

所以它与 `n400` 的真空能占比**相同**，与 baseline 的 $\Delta x$ 和 $\Lambda_{\rm UV}$ **相同**。
判别逻辑：

- 若 `box_n400` 在共同窗口内**回到 baseline 的 2–3%**（尽管真空能占比和 `n400` 一样高）
  $\Rightarrow$ 压低来自 $\Delta x$/色散通道，§4.2.1 的零点能猜想可以降级；
- 若 `box_n400` 仍然**偏低 10–17%** $\Rightarrow$ 压低与 $\Delta x$ 无关，只与"$N$ 决定的那些量"
  （真空能占比，以及可能的 IR/盒子）有关，§4.2.1 升级为已定位的系统误差；
- 若两者都不像（中间值），说明有两条以上通道叠加，必须做显式减除才能分开。

注意它**不**改变真空能占比，所以它不是"减除对照"，真正的减除对照仍是方案 A。
另外，由于峰值落在 $k\simeq10^{-8}$ Hz，正好在 `COMPARISON_K_MIN = 1e-8` 的左边缘，
统计窗口应放宽到 $[2.6\times10^{-9},0.7k_{\max}]$ 再用 Python 重算，
不要直接用 `summary.csv` 里的 `rel_baseline_*`。

### 4.3 图

- `papers/gauss-bonnet/fig/convergence-resolution.pdf`
- `papers/gauss-bonnet/fig/convergence-resolution.png`
  - 包含 N=64、N=128、N=300、N=400 与 N=200 baseline。
- `papers/gauss-bonnet/fig/convergence-dt.pdf`
- `papers/gauss-bonnet/fig/convergence-dt.png`
  - N=64 $\Delta t=1$ vs $\Delta t=2$。
- `papers/gauss-bonnet/fig/convergence-evolution-n400.pdf`
- `papers/gauss-bonnet/fig/convergence-evolution-n400.png`
  - N=400 的 `spectrums` 中期谱快照、晚期冻结曲线和估计量因子 $H/\langle\dot\phi\rangle$（`plot_evolution.py`）。
- `papers/gauss-bonnet/fig/convergence-evolution-n300.pdf/png`
  - 同上，作为 N=300 对照。

### 4.4 数据位置

本地合并后的工作目录：

```text
inflat/out/gauss_bonnet_convergence/alpha_1.13412e7/
```

包含：

```text
n128.summary.npz
n128.final_field.npz
n128.lattice.bincode
dt1_n64.summary.npz
dt1_n64.final_field.npz
dt1_n64.lattice.bincode
n300.summary.npz
n300.final_field.npz
n300.lattice.bincode
n400.summary.npz
n400.final_field.npz
n400.lattice.bincode
n400.lattice.checkpoint.bincode
n400.lattice.zeta.checkpoint.bincode
n400_super500.summary.npz          # 2026-11-12 补下载（仅 summary，完整结果在远端）
dt4_n128.*, dt8_n128.*             # 结果可疑，见 §7 第 15 条
summary.csv
comparison.npz
resolution.*.html
timestep.*.html
all.*.html
```

2026-11-12 起 `box_n400_k0_1e-9` 正在远端运行，本地暂时只有日志；预计产出
`box_n400_k0_1e-9.{lattice.bincode,final_field.npz,summary.npz}`，单个算例约 4.4 GB
（远端根分区当时剩 80 GB，够用）。

远端全部原始输出仍保留在：

```text
reex-zerotier:~/computations-cfy/out/gauss_bonnet_convergence/
```

本地曾用 `inflat/out/gauss_bonnet_convergence_remote*` 保存下载归档；这些归档已在算例合并后删除（释放约 7.6 GB）。原始运行日志已复制到 `inflat/out/gauss_bonnet_convergence/` 顶层，与远端结构一致：

```text
build.log
dt1_n64.run.log
n128.run.log
n300.run.log
n400.run.log
n512.run.log
timestep.plot.log
```

（`n64.run.log` 是本地运行产生的，远端没有；`n512.run.log` 是早期一次中止的 N=512 尝试，只有 29 步，没有留下数据。）

清理后本地与远端仍有的差异：远端多 `n300.lattice.checkpoint.bincode`、`n300.lattice.zeta.checkpoint.bincode` 和 `backup_before_n300/`；本地多 `n64.*`。checkpoint 在 `.lattice.bincode` 存在时不会被读取，不影响复现。

注意：远端 `timestep.*` 仍是旧版 `--plot-only` 生成的（当时只有 `dt1_n64` 和 `n128`），不能作为纯 dt 对比；本地工作目录中的 `timestep.*` 已用修正后的同 N 逻辑重新生成。

---

## 5. SIGW 的处理

本次**没有**使用 SIGW 结果，也没有把 $\Omega_{\rm GW}$ 画进收敛图。

原因：`inflat/src/igw.rs` 中的 `sigw_2_spectrum` 在积分时会对超出输入 $k$ 范围的点做截断（`interpolate_1d(..., extrapolate=false)` 会夹到边界值）。N=64 的采样范围比 N=200 小，截断值对 $\Omega_{\rm GW}$ 的影响很大，因此 N=64 的 SIGW 不能作为收敛性判据。

数据中的以下内容仅作为原始输出保留，不应直接用于文章结论：

- `*.sigw.bincode`
- `summary.npz` 中的 `sigw`
- `summary.csv` 中的 `sigw_rel_baseline_*`
- `*.sigw.html`

如果以后要研究 SIGW 收敛性，需要先修改 `sigw_2_spectrum` 的边界处理（例如外推、或只在共同 $k$ 范围内积分），再重新计算。

---

## 6. 后续计算更大 $N$

### 6.1 推荐算例顺序

N=300 与 N=400 分辨率算例均已完成（结果见第 4 节）。N=400 在共同窗口内相对 baseline 偏中位约 10%，没有继续收敛（见 4.2），因此下一步优先做 N=400 的对照，而不是直接上更大的 N。

先做分辨率收敛，再做盒子、dt、seed 等：

1. ~~`N=256`，`k0=2e-9`，`dt=2.0`~~（已由 N=300 覆盖，跳过）
2. `N=256`，`k0=1.5625e-9`，`dt=2.0`（固定 $\Delta x$，盒子 $1.28L$）
3. `N=320`，`k0=2e-9`（可选，分辨率）
4. `N=320`，`k0=1.25e-9`（固定 $\Delta x$，盒子 $1.6L$）
5. `N=400`，`k0=1e-9`（固定 $\Delta x$，盒子 $2L$）——**2026-11-12 已启动**，
   `box_n400_k0_1e-9`，ETA 约 92 h，判别逻辑见 §4.2.4
6. **优先**：在 N=400 上补 `seed=11,12` 和 `super=500`，确认 4.2 中约 10% 的偏差是随机实现还是系统误差
   —— `super=500` 已完成（`n400_super500`，峰值只变 $-2.3\%$、峰位不动，见 §4.1）；
   `seed=11,12` 目前只有 N=128 的配置，N=400 的还没加
7. `N=512`，`k0=2e-9`，`dt=2.0`（分辨率；实测外推约 9 天）
8. **新增（2026-11-12）**：实现 lit-survey 方案 A（格点零点能减除）后重跑 N=200/N=400。
   这是唯一能把"空间分辨率"和"UV 真空能"分开的测试，理由见 §4.2.3。
   论文 arXiv:2610.26669 给了连续极限的绝热 counterterm 形式（其式 3.10、3.17–3.18），
   但格点上应直接减 $\frac{1}{2V}\sum_{\mathbf k}\omega_{\mathbf k}(t)$（含 $V''$ 质量项），
   并把 $\rho$ 与 $p$ 一起减（$p=\rho/3$）。

> **顺序说明**：第 5 项之所以排在第 8 项之前，是因为它不需要改代码、只花机时，
> 且能立刻给出"$\Delta x$ 通道 vs $N$ 通道"的第一刀；但它**不能**替代第 8 项，
> 因为固定 $\Delta x$ 的 box 测试并不改变 $\rho_{\rm vac}/\rho_{\rm tot}$（见 §4.2.3）。

对应频率范围：

| N | $k_{\max}$ [Hz] | $\Delta x$ [pc] |
|---:|---:|---:|
| 200 | $2.205\times10^{-7}$ | 0.02430 |
| 256 | $2.823\times10^{-7}$ | 0.01898 |
| 320 | $3.529\times10^{-7}$ | 0.01519 |
| 400 | $4.411\times10^{-7}$ | 0.01215 |

固定 $\Delta x$ 时：

| N | $k_0$ [Hz] | $L$ [pc] |
|---:|---:|---:|
| 200 | $2.000\times10^{-9}$ | 4.860 |
| 256 | $1.5625\times10^{-9}$ | 6.221 |
| 320 | $1.250\times10^{-9}$ | 7.776 |
| 400 | $1.000\times10^{-9}$ | 9.720 |

### 6.2 添加新配置

在 `inflat/src/bin/gauss_bonnet_convergence.rs` 的 `configs()` 中加入，例如：

```rust
Config {
    name: "n256",
    group: "resolution",
    n: 256,
    k0: BASELINE_K0,
    dt: BASELINE_DT,
    sub: BASELINE_SUB,
    sup: BASELINE_SUPER,
    seed: BASELINE_SEED,
},
Config {
    name: "box_n256_k0_1p5625e-9",
    group: "box",
    n: 256,
    k0: 1.5625e-9,
    dt: BASELINE_DT,
    sub: BASELINE_SUB,
    sup: BASELINE_SUPER,
    seed: BASELINE_SEED,
},
```

并在 `color_for()` 中加颜色，例如：

```rust
"n256" => "#ff7f0e",
"box_n256_k0_1p5625e-9" => "#17becf",
```

### 6.3 本地构建与检查

```bash
cd inflat
cargo build --release --offline --bin gauss_bonnet_convergence
../target/release/gauss_bonnet_convergence --list
```

### 6.4 上传到远端

只上传 `inflat/` 的内容，排除 `out/`：

```bash
cd /Users/hadroncfy/Documents/Academic/work/inflation-2025
rsync -avz --exclude 'out/' --exclude '.DS_Store' \
    -e 'ssh -o BatchMode=yes' \
    inflat/ reex-zerotier:~/computations-cfy/
```

baseline 不需要重复上传（远端已有 `~/computations-cfy/baseline/case1-final.output.npz`）。如果远端 cache 缺包，参考第 3.6 节手动上传 crate。

### 6.5 远端构建

```bash
ssh -o BatchMode=yes reex-zerotier \
    'bash -lc "cd ~/computations-cfy && cargo build --release --offline --bin gauss_bonnet_convergence"'
```

如果 `--offline` 报缺 crate，先在本地找到对应的 `.crate`，上传到远端 `$CARGO_HOME/registry/cache/index.crates.io-*/`，再重试。

### 6.6 用 screen 顺序运行

不要同时跑多个算例。建议每次只跑一个，或者写一个 `set -e` 的顺序脚本。

例如跑 `n256`：

```bash
ssh -o BatchMode=yes reex-zerotier \
    'cd ~/computations-cfy && mkdir -p out/gauss_bonnet_convergence && \
     screen -dmS gb_n256 bash -lc \
     "./target/release/gauss_bonnet_convergence \
        --baseline baseline/case1-final.output.npz \
        --run --config n256 \
        > out/gauss_bonnet_convergence/n256.run.log 2>&1"'
```

检查：

```bash
ssh reex-zerotier 'screen -ls'
ssh reex-zerotier 'tail -5 ~/computations-cfy/out/gauss_bonnet_convergence/n256.run.log'
```

中断后续算：重新执行同一命令即可，`lazy_file_opt` 会优先读取 checkpoint。

### 6.7 下载与合并

下载到新目录，避免覆盖已有结果：

```bash
rsync -avz -e 'ssh -o BatchMode=yes' \
    reex-zerotier:~/computations-cfy/out/gauss_bonnet_convergence/ \
    inflat/out/gauss_bonnet_convergence_remote_n256/
```

然后只把新算例的文件复制到工作目录：

```bash
cp -a inflat/out/gauss_bonnet_convergence_remote_n256/alpha_1.13412e7/n256.* \
      inflat/out/gauss_bonnet_convergence/alpha_1.13412e7/
```

不要直接用远端的 `summary.csv` / `comparison.npz` / `all.*` 覆盖本地，因为远端可能没有本地已有的 `n64`。

### 6.8 重新生成图和表

```bash
cd inflat
../target/release/gauss_bonnet_convergence --plot-only \
    --config n64 --config n128 --config n256
```

然后修改 `papers/gauss-bonnet/plot_convergence.py`：

- 在 `main()` 中 `load_case("n256")`
- 在 `make_resolution_plot` 中加入 N=256 的曲线、颜色和相对差异
- 更新图例和输出文件名（例如 `convergence-resolution.pdf` 保持同名即可）

运行：

```bash
python3 papers/gauss-bonnet/plot_convergence.py
```

---

## 7. 已知坑与注意事项

1. **不要运行旧生产程序覆盖数据**
   - `inflat/src/bin/gauss_bonnet.rs` 的 `run_scan_case1` 会写入 `inflat/out/gauss_bonnet/`。
   - `inflat/out/gauss_bonnet.bak/` 是备份，不要动。
   - 新算例只写入 `inflat/out/gauss_bonnet_convergence/`。

2. **`--run` 不会覆盖已有文件**
   - `LatticeInput::run` 使用 `lazy_file_opt`：文件存在时直接读取。
   - 要重算某个算例，需要先手动删除对应的 `.lattice.bincode` 和 `.checkpoint.bincode`。

3. **checkpoint**
   - 每 600 s 写一次 `.checkpoint.bincode`。
   - 中断后重新运行同一命令会从 checkpoint 继续。
   - `.zeta.checkpoint.bincode` 是 $\delta N$ 重建阶段的 checkpoint。

4. **baseline 行选择**
   - 必须用 `abs(params - xi0) < 1.0`，不能用 `np.isclose` 默认参数。
   - `1.13411e7` 和 `1.13412e7` 相差 100，默认 `rtol=1e-5` 会同时匹配。

5. **`param_set1` 参数顺序**
   - 形参 `(phi_c, xi0, xi1)`，生产调用 `(9.6, xi0, 30.0)`。
   - 新算例沿用生产调用；不要在收敛测试里改成 `(30.0, xi0, 9.6)`，否则曲线对不上。

6. **`k_min` 与首个 bin 中心不同**
   - 首个 bin 的中心约 $2.55\times10^{-9}$ Hz（对 $k_0=2\times10^{-9}$），这是 shell binning 的结果。
   - 报告频率范围时用 `k_min()` / `k_max()`，不要用 `lat_k[0]` 反推 $k_{\min}$。

7. **UV 区域**
   - 首尾 bin 和 $k>0.7k_{\max}$ 的区域受有限格距/截断影响。
   - 相对差异统计使用 $[10^{-8},0.7k_{\max}]$；图中灰色标为 `UV-affected`。
   - 如果不剔除，N=128 末端会出现 >100% 的假差异。

8. **盒子测试方向**
   - 现有 `box_n64`/`box_n128` 是更小的盒子（固定 $\Delta x$）。
   - 更大的盒子需要 $N>200$：$1.28L$ 用 N=256，$1.6L$ 用 N=320，$2L$ 用 N=400。

9. **磁盘**
   - `final_field.npz` 和 `.lattice.bincode` 都保存最终场。
   - N=256 最终场约 400 MB（未压缩）；压缩 NPZ 会小一些，但总量仍可观。
   - N=400 实测：`n400.final_field.npz` 1.36 GB、`n400.lattice.bincode` 1.55 GB、checkpoint 1.02 GB、zeta checkpoint 0.51 GB，单个算例约 4.4 GB。
   - N=512 会更大，注意磁盘空间。

10. **SIGW**
    - `sigw_2_spectrum` 会截断采样范围外的点。
    - N=64 的 SIGW 不可用于收敛性判断。
    - 本任务的两张图完全不含 SIGW。

11. **远端的 `timestep.*`**
    - 远端最后一次 `--plot-only --group timestep` 只有 `dt1_n64` 和 `n128`，不是纯 dt 对比。
    - 本地已用修正后的同 N 逻辑重新生成 `timestep.*`（`dt1_n64` vs `n64`）。
    - 论文图以 `plot_convergence.py` 的 PDF/PNG 为准。

12. **CSV 的 `rel_baseline` 列**
    - `summary.csv` 里的 `rel_baseline_*` 是按每个算例自己的 overlap 计算的。
    - N=128 的 overlap 可以延伸到 $0.7k_{\max}(N=128)\approx 9.9\times10^{-8}$；N=300 可以延伸到约 $2.3\times10^{-7}$。因此它们的 `max` 会比图里共同窗口的 8.62%/10.79% 大。
    - 图里的数字使用所有算例共同的 N=64 窗口，便于公平比较。

13. **`spectrums` 不能直接当作 $\zeta$ 的冻结判据**
    - `summary.npz` 的 `spectrums` 是 $\mathcal P_\phi(H/\langle\dot\phi\rangle)^2$ 的线性化估计，不是最终的非线性 $\delta N$ 谱。
    - USR 期间 $H/\langle\dot\phi\rangle$ 会先暴涨约 $7\times10^2$ 倍再回落，所以中期快照的绝对幅度不能直接解读为 $\zeta$ 演化。
    - 用 `plot_evolution.py` 时应同时看第三个面板的估计量因子。

14. **N=400 尚未收敛**
    - 共同窗口内相对 baseline 中位 10.41%、最大 17.35%，差于 N=128/N=300 的 2–3%。
    - 谱整体偏低、峰位更低（$2.02\times10^{-8}$ Hz），而 N=400 的演化反而最长，因此不是「时间不够」。
    - `n400_super500` 已排除"末态未冻结"（峰值只再降 2.3%、峰位不动）。
    - 在 seed 对照和 UV 真空能减除（§4.2.3/§4.2.4）完成前，不要把 N=400 当作分辨率收敛的证据。

15. **`dt4_n128` 与 `dt8_n128` 的结果可疑（待核实，2026-11-12 发现）**
    - 两个算例的最终场一致到 $10^{-13}$（`zeta` 差 $6.1\times10^{-6}$，`a` 差 $3\times10^{-9}$，
      末态 $\ln a$ 都是 9.8353057），而 `n128`(dt=2) 与 `dt4_n128` 的 `zeta` 差 $1.4\times10^{-2}$。
    - 二阶格式下 dt=4→8 的差应该比 dt=2→4 **大** 4 倍，现在却小了约 $2\times10^{3}$ 倍；
      这不像是收敛，更像是 dt 被截断在 4，或 dt8 复用了 dt4 的 checkpoint。
    - 在查清之前，**"$\Delta t=2$ 已收敛"只能依据 `dt1_n64` 那条（N=64, dt=1 vs 2）**，
      不要引用 dt4/dt8 的结果做收敛性论证。
    - 最快的判据：把 `dt` 提到 16 再跑一次 N=128，看是否还是同一个结果。

16. **`box_n400_k0_1e-9` 的峰值落在比较窗口左边缘**
    - 由于 $k$ 轴整体减半，预计峰位在 $k\simeq1.0\times10^{-8}$ Hz，
      正好等于 `COMPARISON_K_MIN = 1e-8`。
    - 分析时把窗口放宽到 $[2.6\times10^{-9},0.7k_{\max}]$ 用 Python 重算，
      不要直接用 `summary.csv` 的 `rel_baseline_*`。

---

## 8. 快速命令清单

```bash
# 本地构建
cd inflat
cargo build --release --offline --bin gauss_bonnet_convergence

# 查看计划
../target/release/gauss_bonnet_convergence --list

# 本地跑/读取已有结果后出图
../target/release/gauss_bonnet_convergence --plot-only \
    --config n64 --config n128 --config n300 --config n400 --config dt1_n64

# 画论文图
cd ..
python3 papers/gauss-bonnet/plot_convergence.py

# 画 N=400 的中期谱演化诊断
python3 papers/gauss-bonnet/plot_evolution.py n400

# 上传源码到远端
rsync -avz --exclude 'out/' --exclude '.DS_Store' \
    -e 'ssh -o BatchMode=yes' \
    inflat/ reex-zerotier:~/computations-cfy/

# 远端离线构建
ssh -o BatchMode=yes reex-zerotier \
    'bash -lc "cd ~/computations-cfy && cargo build --release --offline --bin gauss_bonnet_convergence"'

# 远端 screen 运行单个算例
ssh -o BatchMode=yes reex-zerotier \
    'cd ~/computations-cfy && screen -dmS gb_n400 bash -lc \
     "./target/release/gauss_bonnet_convergence \
        --baseline baseline/case1-final.output.npz \
        --run --config n400 \
        > out/gauss_bonnet_convergence/n400.run.log 2>&1"'

# 查看 screen / 日志
ssh reex-zerotier 'screen -ls'
ssh reex-zerotier 'tail -5 ~/computations-cfy/out/gauss_bonnet_convergence/n400.run.log'

# 下载远端结果
rsync -avz -e 'ssh -o BatchMode=yes' \
    reex-zerotier:~/computations-cfy/out/gauss_bonnet_convergence/ \
    inflat/out/gauss_bonnet_convergence_remote_n400/

# ---- 2026-11-12: box_n400_k0_1e-9（正在跑，screen 名 gb_box400）----

# 启动（已执行，勿重复启动；中断后重跑同一命令会从 checkpoint 续算）
ssh -o BatchMode=yes reex-zerotier \
    'cd ~/computations-cfy && screen -dmS gb_box400 bash -lc \
     "./target/release/gauss_bonnet_convergence \
        --baseline baseline/case1-final.output.npz \
        --run --config box_n400_k0_1e-9 \
        > out/gauss_bonnet_convergence/box_n400_k0_1e-9.run.log 2>&1"'

# 监控（ETA 在日志里；预计约 92 h）
ssh reex-zerotier 'screen -ls'
ssh reex-zerotier 'tail -3 ~/computations-cfy/out/gauss_bonnet_convergence/box_n400_k0_1e-9.run.log'
ssh reex-zerotier 'df -h /root | tail -1; du -sh ~/computations-cfy/out/gauss_bonnet_convergence'

# 完成后只下载 summary 先看数（final_field/bincode 约 4.4 GB）
rsync -avz -e 'ssh -o BatchMode=yes' \
    reex-zerotier:~/computations-cfy/out/gauss_bonnet_convergence/alpha_1.13412e7/box_n400_k0_1e-9.summary.npz \
    inflat/out/gauss_bonnet_convergence/alpha_1.13412e7/
```

---

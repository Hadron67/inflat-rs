# 宇宙学格点方法中的重整化：文献调研

调研日期：2026-11-06
目的：为 `task.md` §4.2.1（N=400 谱偏低，怀疑未重整化的 UV 真空能抬高初始 $H$）寻找
"宇宙学格点方法里是否有人做过重整化"的文献依据。

---

## 0. 结论

**基本没有。**

主流宇宙学格点程序（CosmoLattice、LATTICEEASY、PSpectRe、DEFROST、STOLAS、
InflationEasy、PSpectCosmo，以及近几年 USR / axion / SGB 的格点文章）**都不引入
counterterm、不做真空能减除、也不讨论格点的 UV 重整化**。它们把格点当作一个
UV 正规化，然后就停在那里。

个别工作识别到了"格点零点能发散"这一问题，但处理方式是**回避**（换个定义、砍掉高
$k$、只算过视界模），不是重整化。真正做了重整化的是**相邻领域**：量子实时格点
（2PI / Kadanoff–Baym）和弯曲时空格点 QFT。这些可以作为 §4.2.1 的方法学模板。

对文章写作的含义：**不能说"标准做法是重整化"**。正确的表述是"这是一个被广泛回避、
尚未被系统处理的问题，我们在本文中显式处理它"。

---

## 1. 直接证据：关键词统计

方法：从 arXiv 下载 PDF，`pdftotext` 转文本，`grep -ic` 统计词干
（`renormali` 覆盖 renormalization / renormalized / renormalizable / renormalisation）。
这样能区分"正文里真的讨论过"和"只在参考文献标题里出现"。

| arXiv | 内容 | renormali | divergen | counter-term | vacuum energy |
|---|---|---:|---:|---:|---:|
| 2102.01031 | CosmoLattice（主程序论文） | **0** | 0 | 0 | **0** |
| 2006.15122 | The art of simulating the early Universe I | 1 | 1 | 0 | 1 |
| 2512.15627 | The art of simulating the early Universe II | 3 | 0 | 1 | 0 |
| hep-ph/0011159 | LATTICEEASY | **0** | 0 | 0 | 0 |
| 1005.1921 | PSpectRe | **0** | 0 | 0 | 0 |
| 0809.4904 | DEFROST | **0** | 2 | 0 | 0 |
| 2405.10692 | STOLAS | 4 | 0 | 0 | 0 |
| 2506.11797 | InflationEasy | **0** | 0 | 0 | 0 |
| 2410.23942 | Ultra-Slow-Roll on the Lattice I | **0** | 0 | 0 | **0** |
| 2506.11795 | Ultra-Slow-Roll on the Lattice II | 1 | 7 | 0 | **0** |
| 2204.12874 | Lattice sim. of Axion-U(1) Inflation | **0** | 0 | 0 | 0 |
| 1812.03132 | Lattice formulation of axion inflation | **0** | 0 | 0 | 0 |
| 2608.20219 | SGB on the lattice（本文） | **0** | 0 | 0 | 0 |
| 2411.17658 | PSpectCosmo | **0** | 2 | 0 | 1 |
| 2312.08167 | Quantum tachyonic preheating (2PI) | **14** | 6 | 0 | — |

注：USR II 用 HTML 全文统计（`arxiv.org/html/2506.11795v2`），其余用 PDF + `pdftotext`。

逐条核对命中的上下文，全部是**别的意思**，没有一条是格点 vacuum-energy 重整化：

- **2006.15122**：`renormalizable part of the SM`（粒子物理常规用语）、
  `divergence of a vector`（矢量场散度）、`vacuum energy of a Higgs-like field`（暴胀势）。
- **2512.15627**：`ξRφ² ... in order to renormalize the theory`（连续理论里非最小耦合的
  必要性）；另一处是展望里说将来做 2PI 量子模拟时 `要引入重整化程序和 counter-term`，
  即明确把重整化划归"超出经典格点"的范畴。
- **2405.10692 (STOLAS)**：`renormalised noise` —— 随机暴涨噪声的重定义，与真空能无关。
- **2506.11795 (USR II)**：唯一一处 `renormalization` 是
  "`a renormalization of the scale factor`" —— 把 $\zeta$ 的一个常数平移吸收进背景
  演化，属于**背景/标度因子的重整化**（IR），与 UV counterterm 无关。这一点和
  Kristiano–Yokoyama 的"Inflationary background renormalization"（arXiv:2504.18514）
  是同一类想法，可以借用这个词汇，但要明确区分。7 处 `divergen` 全是 $\zeta_{\rm lin}$
  的对数发散，不是 UV。
- **2411.17658 (PSpectCosmo)**：`divergen` 指真空能发散，但处理方式见 §2(c)。
- **2112.08388 / 2512.15627**：`ξ` 的重整化是连续理论层面的 RG 跑动，**不是**格点截断重整化。
  写文章时不要混用这两个"renormalization"。

---

## 2. 他们实际怎么做：四种"回避"策略

### (a) 初态 $H$ 只用齐次背景算

CosmoLattice 源码（2102.01031 附录，第 122 行附近的注释）明确写：

> `// Here we call this function to compute the value of the potential on the homogeneous
> initial condition (useful to set the initial Hubble rate).`

也就是说初始 $H$ 由**均匀背景**给出，真空涨落的能量不进入 $H_{\rm ini}$。之后演化中
标度因子才用格点体积平均能量自洽演化。这与本文代码 `solve_hubble` 把噪声的梯度、
动能、势能一起代入约束的做法**不同**，也是本文 §4.2.1 现象之所以出现的原因。

### (b) 给初始涨落加一个硬截断

CosmoLattice 有 `kCutOff` 参数（同上附录）：

> `If specified, the given cutoff (in program units) is imposed in the spectrum of initial
> fluctuations for all scalar fields: the amplitude of the field modes at larger momenta is
> set to zero up to machine precision. Not specifying kCutOff implies not having an initial
> cut-off, whereas kCutOff = 0 implies initially vanishing fluctuations.`

这是一个纯数值旋钮，没有对应的 counterterm；改变 `kCutOff` 就改变了裸量。

### (c) 只把过视界模计入能量密度

PSpectCosmo（2411.17658）是**唯一一篇把"暴胀起始时能量密度发散"当作核心问题写进
摘要的格点文章**：

> `During the initial stages of inflation, lattice modes are sub-horizon and represent quantum
> fluctuations, whose kinetic and gradient energies should be excluded from the total energy
> density. However, as these modes cross the horizon, they transition from quantum to classical
> behavior and must be incorporated into the energy density calculations.`
> （为此引入一个跟踪 horizon exit 的参数 `R`。）

这是"减除"，但减除条件是**时间依赖的、按视界交叉定义的**，不是一个可微的
counterterm，也没有给出连续极限的收敛性论证。可以作为"这个问题是被承认的"的引文，
但不宜作为方案模板。

### (d) 干脆不提

USR-on-the-lattice I/II（Caravano–Franciolini–Renaux-Petel）、axion-U(1)、axion
inflation preheating、以及本文所用的 SGB 格点，全部是标准 TWA：

> `fluctuations are introduced in the discrete Bunch-Davies vacuum in Fourier space:
> δφ(κ,τ) = 1/√(2ω_κ) e^{−iω_κ τ}`，`ω_κ` 用格点色散关系
> `ω = (2N/L)√(Σ sin²(πm_i/N))`。

与本文 §2.2/§2.3 的构造完全相同，但**没有人讨论这个离散真空的零点能如何影响背景，
也没有人做 $N$ 变化的收敛性检查**。换句话说，本文 §4.2.1 提出的问题在这一支文献里
是空白。

---

## 2bis. InflationEasy（Caravano 组）的源代码级核对

Caravano–Franciolini–Renaux-Petel 的 USR 两篇文章用的是 **InflationEasy**
（arXiv:2506.11797，代码 `refs/inflation-easy`，v1.1.3）。直接读源码比读论文更能说明
问题，结论是：**它和本文的代码在这一点上几乎一模一样，同样没有任何重整化。**

### 初始化：`src/initialize.cpp`

1. `set_mode()`（第 80–126 行）逐模填充 BD 真空：
   - `rms_amplitude = norm / sqrt(2*omega)`，`norm = rescale_B * pow(L/dx^2, 1.5)`；
   - `omega = sqrt(k_eff^2 (+ m_init^2))`，**`k_eff` 用的是格点修正色散关系**
     （`spatial::effective_momentum_squared`），与本文 §2.2 的 $k_{\rm eff}$ 一致；
   - `deriv = omega*Im φ + hbterm*Re φ`，`hbterm = -hubble_init`，即
     $\phi' = (-i\omega - H)\phi$（共形时间下的 $\dot\phi$ 换算），与本文
     `populate_noise` 里的 `v_phi/a² − φ·v_a/a²` 是同一件事；
   - 高/低模截断 `high_cutoff_index` / `low_cutoff_index`，**默认为 0 即关闭**
     （`params.txt` 里就是 `0.0`），所以生产跑里所有模（含 Nyquist）都被填充。

2. `initializef()`（第 238–247 行）算初始 Hubble：

```cpp
double vard = 0.0;
LOOP vard += pw2(fd[idx(i,j,k)]);
double deriv_energy_in = 0.5 * vard / gridsize;

const double total_initial_energy =
    deriv_energy_in + potential_energy() + gradient_energy();
hubble_init = checked_hubble(total_initial_energy, "the initialized lattice energy");
ad = hubble_init;
```

即 $3H_{\rm ini}^2=\langle\dot\phi^2\rangle/2+\langle V\rangle+\langle(\nabla\phi)^2\rangle/2$，
**动能、势能、梯度三项全部包含真空涨落**，没有任何减除。这与本文
`GaussBonnetField::populate_noise` → `solve_hubble` 的做法完全对应。

### 演化：`src/evolution/inflation.cpp`

标度因子方程是

```
a'' = a^(3-2s) (2 ρ_grad/3 + <V~>) - (s+1) a'^2/a
```

（用 Friedmann 约束消掉了动能项），其中 `gradient_energy()` 和 `potential_energy()`
都是**全格点**平均，所以真空能在整个演化过程中持续驱动膨胀，而不只是初态。

### 论文里的原话（2506.11797 §3.2.1）

这是整段调研里最接近 §4.2.1 的表述，值得直接引用：

> `After generating the fluctuations, the initial value of ∂_τ̃ a is updated to include the
> contribution from the gradient energy, computed as a lattice average of (∇̃φ)²/2.
> **While quantum vacuum fluctuations should not source the background equations, their
> inclusion here reflects the semi-classical approximation used in the lattice approach.
> Importantly, omitting this gradient term would introduce a residual curvature term into
> the second Friedmann equation, compromising the consistency of the evolution.** Including
> it also enables accurate monitoring of energy conservation during the simulation.`

即：他们**知道**真空涨落原则上不该驱动背景，但**故意保留**，理由是不保留会让
Friedmann 约束出现残差。这是一个"选约束自洽、不选重整化"的显式取舍，而且是
CosmoLattice 的相反选择。

### 量级（实测）

在 `/tmp` 里复制一份代码，只把 `const int N` 从 128 改成 64，用仓库自带的默认
`params.txt`（$V_0=3\times10^{-9}$、$L=10$、$\bar\phi_{\rm in}=2.918$、
$\dot{\bar\phi}_{\rm in}=-0.0673$、`n=64`）跑 2 步，直接读 `results/energy.dat`
第一行（$t=0$，即初始化后、演化前）：

| $N$ | $E_{\rm kin}$ | $E_{\rm grad}$ | $E_{\rm pot}$ | $E_{\rm grad}/E_{\rm pot}$ |
|---:|---|---|---|---:|
| 64 | $6.7982\times10^{-12}$ | $9.0003\times10^{-15}$ | $2.97063\times10^{-9}$ | $3.03\times10^{-6}$ |
| 128 | $6.9335\times10^{-12}$ | $1.4427\times10^{-13}$ | $2.97063\times10^{-9}$ | $4.86\times10^{-5}$ |

- $E_{\rm grad}$ 从 64 到 128 正好涨 **16 倍** = $(2)^4$，$N^4$ 标度坐实。
- $E_{\rm kin}$ 的大头是背景 $\frac12\dot{\bar\phi}^2$；把两行的动能相减，
  $6.9335-6.7982=1.35\times10^{-13}$，正好等于 $E_{\rm grad}(N{=}128)$，
  说明**真空动能 = 真空梯度能**（equipartition），所以总真空能 $\simeq 2E_{\rm grad}$。
- 外推到他们生产用的 $N=512$：
  $$E_{\rm grad}(512)=1.4427\times10^{-13}\times4^4=3.69\times10^{-11}$$
  $$\frac{\rho_{\rm vac}}{\rho}\simeq\frac{2E_{\rm grad}}{E_{\rm pot}}=2.5\%,\qquad
  \frac{\delta H_{\rm ini}}{H}\simeq1.2\%.$$

也就是说，**Caravano 组 $N=512$ 的生产跑，初态里未减除的 BD 真空能约占背景的
2.5%，把 $H_{\rm ini}$ 抬高约 1.2%**——比本文 $N=400$ 的 1.36% / 0.68% 还大一倍。
论文里只说"free of lattice artifacts, such as finite-size effects"，没有单独讨论
这一项；**若他们的分辨率扫描包含"固定 $L$、改变 `Npts`"（也就是本文的做法），这个
效应就会和真正的空间分辨率效应混在一起。**

注意这只是**仓库默认算例**（$V_0=3\times10^{-9}$、$L=10$）的数；他们 USR 文章用的
$V_0$ 和 $L$ 不同（$L$ 由 $k\in[0.22,45]k_{\rm ref}$ 定），数值会有出入，但 $N^4$
标度和量级是稳健的。要对他们真实参数取值，改 `params.txt` 后重跑一次、读
`results/energy.dat` 第一行即可。

### 结论

- 代码里 `grep -i "renorm|counter|subtract"` **零命中**；
- 初始 $H$ 与整个膨胀历史都被裸的 BD 真空能污染，且随 $N^4$ 变化；
- 他们做了分辨率检查（$512^3$，并"changing lattice parameters Npts and L"），但
  论文只提"finite-size effects"，没有把 UV 真空能这一项单独拿出来讨论；
- 因此**不能把 Caravano 的结果当成"已处理重整化"的先例**；反过来，这正好说明
  本文 §4.2.1 指出的问题是这一类主流代码共有的、尚未被量化的系统误差。

---

## 3. 真正做了重整化的文献（可借用的方法学）

### (1) Tranberg & Ungersbäck, 2312.08167 —— 最接近的模板

"Quantum tachyonic preheating, revisited"（量子实时格点，2PI / Kadanoff–Baym，暴胀后
tachyonic preheating）。它有整整一节 `3.2.2 Renormalisation`，开头一句几乎就是本文
需要的表述：

> `The evolution equations are UV divergent, and although the lattice provides a
> regularization, a substantial cut-off dependence is present, calling for renormalisation.`

具体做法（可直接照搬的形式）：

1. **质量 counterterm**：$-(a\mu_0)^2 = -(a\mu)^2 - \delta(a\mu)^2$，其中
   $$\delta(a\mu)^2 = \lambda\frac{N+2}{6N}\frac{1}{N_x^3}\sum_{\vec k}\frac{1}{2\sqrt{(ak)^2_{\rm lat}+(a\mu)^2}}\Big|_{\rm lat}$$
   即在格点上求和得到的 tadpole，系数 $0.212\lambda$（LO 时前因子改为 $1/6$）。
2. **能量/压力 counterterm**：减除质量发散后仍有剩余发散，用一个常数 counterterm
   吸收，重整化条件取"初始重整化能量 = $+V_0$、压力 = $-V_0$"。

这两条正好对应本文需要的两件事：一个 $\propto\Lambda_{\rm UV}^4$ 的能量减除，
以及一个可由重整化条件固定的常数。

### (2) Epelbaum, Gelis, Wu, 1402.0115 —— 为什么经典格点"没有"重整化方案

"Non-renormalizability of the classical statistical approximation"：
经典统计近似虽然 UV 幂次计数与量子理论相同，但**不可重整**（Weinberg 定理在该近似
下失效），残余发散无法用有限个 counterterm 消掉，只能系统性地部分减除。

这条很重要，有两个用途：

- 解释了为什么 CosmoLattice 一系的程序里找不到重整化章节——不是疏忽，而是这个近似
  本身没有严格的重整化方案；
- 说明**残余 UV 依赖是内禀的**，因此必须显式做 $N$（或 $\Delta x$）收敛性检查，
  而不能指望"取连续极限自动变好"。这正好是本文审稿意见第 2 条的立足点。

**但它不能用来解释本文的 $\Lambda_{\rm UV}^4$ 真空能问题。** 详细分析见 §3bis。

---

## 3bis. 1402.0115 是否适用于本文的格点计算：不直接适用

### 论文做了什么

4 维 $\phi^4$ 理论（$g^2\phi^4/4!$）+ 外源 $j$，初态是 $x^0\to-\infty$ 的**微扰真空**
（零温）。CSA 的定义：在 retarded–advanced 基下丢掉所有含三个 1 指标的顶点
$\Gamma^{2111}$（式 17–19），等价于对初条件做高斯平均

$$\overline{c_k c^*_l}=(2\pi)^3E_k\delta^3(\vec k-\vec l)\ \Longrightarrow\ \text{每模 }1/2$$
（式 28–29）——**正是 TWA 的 BD 真空初始化**，和本文 §2.3 用的是同一套东西。

### 病根在哪

| 步骤 | 结果 |
|---|---|
| 表观 UV 幂次计数（式 32） | $\omega=4-E-N_\phi$，与全理论**完全相同** |
| 但 $\Gamma^{1112}$ 被丢掉 | 被积函数变成 $\delta(K^2)\delta((P+K)^2)$，第二个传播子只贡献 $-1$ 而非 $-2$，且**无法 Wick 转动** |
| 后果（式 42） | 1-loop $\Gamma^{1122}$ 出现**线性**发散 $g^4\Lambda_{\rm UV}/64\pi\cdot[\theta(-T)/|p_1+p_3|+\dots]$，系数**非多项式**依赖外动量 |
| 需要什么 counterterm | 需要 $\phi_1^2\phi_2^2$ 算符，而它**根本不在拉氏量里** |

补上被丢掉的那些图（式 44）后，两边的 $\pm\Lambda_{\rm UV}$ 精确抵消，得到式 (45)
的有限结果——这就是全理论可重整的体现。所以：**CSA 的不可重整性 = "有两类图的
紫外发散互相抵消，而 CSA 只保留其中一类"**。

此外 2-loop 的 $\Sigma^{11}$ 有 $\Lambda^2$ 发散（需要 $\phi_1^2$ 算符），
$\mathrm{Im}\,\Sigma^{12}$ 需要**虚** counterterm（破坏厄米性）。

### 与本文问题的四点关键区别

**区别一：它对 $g\to0$ 消失，$\Lambda^4$ 真空能不会。**
污染的量级是 $g^4\Lambda_{\rm UV}/Q$（式 49），判据 $\Lambda_{\rm UV}\ll16\pi Q/g^2$
（式 52）。论文里 $\mathcal L\supset-\frac{g^2}{4!}\phi^4$，所以 $g^2=V''''$。
本文模型

$$g^2=|V''''|=\frac{\Lambda^4}{f^4}=\frac{(0.0065)^4}{7^4}\approx 7.4\times10^{-13},$$

于是污染 $\sim g^4\Lambda_{\rm UV}/Q\sim 5.5\times10^{-25}(\Lambda_{\rm UV}/Q)$，
判据放宽到 $\Lambda_{\rm UV}\ll 6.8\times10^{13}Q$，完全无关。
**而 $\frac12\sum_k\omega_k$ 的 $\Lambda^4$ 在自由场（$g=0$）上就是最大的。
两个问题正交。**

**区别二：它是 NNLO（1-loop）效应，$\Lambda^4$ 是 LO 的初态效应。**
论文自己说"CSA 在 LO 和 NLO 与全理论完全相同，从 NNLO 开始才有差别"。
BD 真空的零点能是初态高斯两点的**零阶**性质，不是圈图修正，不在这套幂次分析的
管辖范围内。

**区别三：它是平坦时空、无引力、无 Friedmann 约束。**
在 Minkowski 里，零点能只是一个不可观测的常数平移（能量守恒，绝对零点无意义），
所以 $\Gamma^{1122}$ 的发散与"真空能"完全无关。**在宇宙学里，引力耦合的是绝对
能量密度**，零点能直接进 $3H^2=\rho$，改的是整个背景轨迹——本文的 $N^4$ 问题
正是这样来的。这一点是 1402.0115 框架里根本没有的东西（它连 $a(t)$ 都没有）。

**区别四（技术层面）：$\Lambda^4$ 是可减除的局域 counterterm，$\Gamma^{1122}$ 不是。**
$\int d^4x\sqrt{-g}\,\delta\Lambda$ 是局域的，能吸收任意常数能量密度；这也正是
Tranberg–Ungersbäck（2312.08167）在做的事。所以"CSA 不可重整"**不构成**
"所以不要尝试减除真空能"的理由。

### 那它能用来干什么

三条，都是真的：

1. **权威表述**："格点提供了一个正规化，但经典统计近似本身不可重整，残余
   $\Lambda_{\rm UV}$ 依赖是内禀的"——支持"必须做截断扫描"这一方法论，
   也就是审稿意见第 2 条的立足点。
2. **§5.2 的时间累积**：污染随时间增长，$Qt_*\sim2048\pi^3g^{-4}(Q/\Lambda)^2$，
   等效于一个 $\propto g^4\Lambda^2_{\rm UV}/Q$ 的假散射率；且**渐近态只由守恒量
   和 $\Lambda_{\rm UV}$ 决定**，所以即使 $g$ 很小，晚期结果仍会依赖截断。
   对本文：把 $g^2\simeq7.4\times10^{-13}$ 代进去得到
   $Qt_*\sim1.1\times10^{29}(Q/\Lambda)^2$，所以 11 个 e-fold 内不可能有问题——
   但这句话值得写进文章，说明长时演化是安全的。
3. **一个反面警告**：它提醒"减除之后不能就认为没事了"，剩余还要有
   $\Lambda_{\rm UV}$ 无关性检查。这与方案 A（减除 + 扫描）是一致的。

### 结论

> **不能**把 1402.0115 当作"所以本文的 $N^4$ 真空能问题无法通过重整化解决"的依据；
> 也**不能**反过来用它论证"格点结果必然不可信"。
>
> **正确用法**：引用它来说明"用格点做 UV 正规化、而不做重整化，在这类经典统计模拟里
> 是标准做法但原则上不完备；残余截断依赖是内禀的，因此必须显式做 $N$ 扫描"，
> 然后紧接着说明本文的 $\Lambda_{\rm UV}^4$ 项**是**一个可被局域 counterterm 吸收的
> 效应，所以本文选择显式减除并验证收敛性——这比"承认它不可重整然后不管"要强。

### (3) Lewis & Vidal, 1911.12978 —— 弯曲时空格点 QFT 的完整流程

"Classical Simulations of QFT in Curved Spacetime I"：在格点上做**逐点分裂
（point-splitting）正规化 + Hadamard 重整化**，得到有限的重整化二次算子期望值，
并明确包含 **de Sitter 的 Bunch-Davies 真空**这一算例。虽然对象是 2D 自由费米子、
技术路线是张量网络，但它是"格点 → 连续极限 → 减除发散 → 有限物理量"这套流程在
弯曲时空里的完整示范，可作为方法学引用。

### (4) 其他相关但不同的

- **Kainulainen, Nurmi, Väisänen, 2406.17468**：经典格点 vs 量子 2PI（Hartree）在
  reheating 暗物质产生上的对比，可说明"经典格点与需要重整化的量子方案之间差 O(1)"。
- **Barker, Gladwyn, Zell, 2606.14853**（"The limits of lattice inflation: a cautionary
  tale"）：不是重整化，但同样是"格点宇宙学的系统性误差"，讲 FLRW 近似导致超视界模
  不冻结；可作为同类"格点方法有未受控系统误差"的引文，且它在 CosmoLattice 里实现了
  一阶度规扰动。

---

## 3ter. 要不要也给 $V(\phi)$ 的参数做重整化？（对照 2312.08167 的质量 counterterm）

### 2312.08167 的质量 counterterm 是什么

对 $O(N)$ 标量理论，单圈 tadpole 给出

$$\delta m^2=\frac{\lambda(N+2)}{6N}\langle\phi^2\rangle,\qquad
\langle\phi^2\rangle=\frac{1}{N_x^3}\sum_{\vec k}\frac{1}{2\omega_{\vec k}}
\simeq\frac{\Lambda_{\rm UV}^2}{4\pi^2}.$$

即 $\delta m^2\propto\lambda\Lambda_{\rm UV}^2$。他们**必须**做，因为 $\lambda$ 是
$O(1)$（文中 $\delta(a\mu)^2\simeq0.212\lambda$，对 $\mu^2$ 是百分之几十的修正）。

### 一般公式：tadpole 一次性重整化整个 $V''$

锐截断下的一圈有效势

$$V_{\rm 1\text{-}loop}=\frac12\int^{\Lambda_{\rm UV}}\frac{d^3k}{(2\pi)^3}\ln\!\left(k^2+V''(\phi)\right)
=\text{const}+\frac{\Lambda_{\rm UV}^2}{8\pi^2}V''(\phi)+\dots$$

所以"$V(\phi)$ 的参数要不要重整化"的答案是：**形式上要，但所有参数的修正被同一个
$\frac{\Lambda_{\rm UV}^2}{8\pi^2}V''(\phi)$ 一次性给出**，量级由 $\Lambda_{\rm UV}^2$
控制，而不是 $\Lambda_{\rm UV}^4$。

### 套到余弦势

$$V=\Lambda^4\left(1+\cos\frac{\phi}{f}\right)\ \Longrightarrow\
\frac{\delta\Lambda^4}{\Lambda^4}\bigg|_{\rm tadpole}
=-\frac{\langle\delta\phi^2\rangle}{2f^2}
=-\frac{\Lambda_{\rm UV}^2}{8\pi^2f^2}.$$

用本文 $N=400$ 的实际数值（$\rho_{\rm grad}=2.979\times10^{-11}$，$\rho_{\rm bg}=2.19\times10^{-9}$）：

| 量 | 值 |
|---|---|
| $3H^2=\rho_{\rm bg}$ | $2.188\times10^{-9}$ |
| $\rho_{\rm vac}=2\rho_{\rm grad}$ | $5.958\times10^{-11}$（占 $\rho$ 的 **2.72%**） |
| 反推 $\Lambda_{\rm UV}$ | $9.85\times10^{-3}\approx365\,H$ |
| $\langle\delta\phi^2\rangle$ | $2.46\times10^{-6}$，$\delta\phi_{\rm rms}/f=2.2\times10^{-4}$ |
| $m^2=V''=\Lambda^4/f^2$ | $3.64\times10^{-11}=0.050\,H^2$ |
| $\delta m^2=\frac12V''''\langle\delta\phi^2\rangle$ | $9.13\times10^{-19}$ |
| $\delta m^2/m^2$ | $\mathbf{2.5\times10^{-8}}$ |
| 全部 $V$ 参数的相对修正 | $\mathbf{-2.5\times10^{-8}}$ |

作为对比，真空能那一条是 $\Lambda_{\rm UV}^4/(16\pi^2\Lambda^4)\sim3\%$。**两者差 6 个
数量级**，原因就是 $\Lambda_{\rm UV}^4/\Lambda^4$ 与 $\Lambda_{\rm UV}^2/f^2$ 之比
$=(\Lambda_{\rm UV}/f)^2/\!\ldots\sim10^{-6}$。

### Gauss-Bonnet 耦合

$\xi(\phi)=\alpha\tanh[\xi_1(\phi-\phi_c)]$ 展开后，
$\delta\xi=\frac12\xi''\langle\delta\phi^2\rangle$，而 $\xi''/\xi'=-2\xi_1\tanh$，所以

$$\frac{\delta\xi}{\xi'}=-\xi_1\langle\delta\phi^2\rangle\simeq-7.4\times10^{-5}.$$

同样可忽略（但比对 $V$ 参数的修正大三个数量级，因为它正比于 $\xi_1$）。

### 什么时候才需要做

判据是 $\Lambda_{\rm UV}\ll2\pi f$。本文 $\Lambda_{\rm UV}/f\simeq1.4\times10^{-3}$，
要到 $\Lambda_{\rm UV}\sim2\pi f$ 需要 $N$ 再大 $\sim4000$ 倍（$N\sim10^6$）。
**任何可行的 $N$ 都不需要质量 counterterm。**

### 结论

1. **不需要**为 $V(\phi)$ 的参数实现 counterterm；$\delta m^2/m^2\sim2.5\times10^{-8}$。
2. 只需要减除 $\Lambda_{\rm UV}^4$ 那一项——它在 $V$ 里对应的是**加一个常数**
   $\delta V_0$（$\phi$ 无关），不是改 $\Lambda^4$。
3. 如果想"形式上完备"，可以两行代码实现
   $\delta V(\phi)=\frac{\Lambda_{\rm UV}^2}{8\pi^2}V''(\phi)$ + $\delta V_0$，
   然后在文章里说明第二项被数值验证为 $10^{-8}$ 量级、可以丢掉。这比只说
   "我们忽略了质量重整化"更有说服力。
4. 审稿人若问"为什么只减真空能不减质量"，就用上面这张表和判据
   $\Lambda_{\rm UV}\ll2\pi f$ 回答。

---

## 4. 对 §4.2.1 的建议

### 4.1 表述

不要写"主流方法都考虑了重整化"（事实相反）。建议的写法：

> 宇宙学格点程序通常把格点本身当作 UV 正规化，但不引入相应的 counterterm；
> 经典统计近似原则上不可重整 [Epelbaum–Gelis–Wu]，因此格点结果的 UV 依赖必须
> 通过显式的截断扫描来检验。在使用 TWA 从 Bunch-Davies 真空填充初态时，每一模的
> 零点能都进入初始能量密度，其紫外部分按 $\Lambda_{\rm UV}^4$ 增长；主流程序通过
> 只用齐次背景定义初始 Hubble 率 [CosmoLattice]、对初始谱加硬截断、或只把过视界模
> 计入能量密度 [PSpectCosmo] 来回避这一点，而没有系统地减除它。本文中……

### 4.2 可以实际做的四种方案

**A. 显式减除格点零点能（推荐）**

BD 真空的零点能是可以在代码里**精确求和**的，不需要用解析的 $\Lambda^4/16\pi^2$：

$$\rho_{\rm vac}^{\rm lat}=\frac{1}{2V}\sum_{\vec k}\omega_{\vec k},\qquad \omega_{\vec k}=k_{\rm eff}=\frac{2}{\Delta x}\sqrt{\sum_i\sin^2\frac{\pi n_i}{N}}$$

在 `solve_hubble` 之前把它从 $\langle(\nabla\phi)^2\rangle$、$\langle\dot\phi^2\rangle$、
$\langle V\rangle$ 里扣掉（等价于加一个 $\delta\Lambda(N)$ counterterm）。这样
$H_{\rm ini}(N)$ 就与 $N$ 无关，分辨率测试才干净。

> 注意：解析估计 $\rho_{\rm vac}=\Lambda_{\rm UV}^4/16\pi^2$ 只是量级估计，格点色散
> 加锐截断会带来 O(1) 的系数差异，应改用上式求和。
>
> 注意：`[noise] <L phi L phi> / a^2` 打印的是 `derivative_square`，**只是梯度项**。
> BD 真空里动能项与之同量级，所以总零点能约为该打印值的 2 倍，引用这个数做量级
> 估计时要乘 2。

**B. 只用齐次背景定义 $H_{\rm ini}$**（CosmoLattice 做法）
CosmoLattice 源码注释明确说初始 $H$ 由齐次初值给出；但涨落能是否在别处补进约束，
我只核对了注释、没有实际跑代码验证，引用前应自行确认。
代价：若确实只由齐次背景给出 $H$，则初始约束残差恰为涨落能，需要显式说明；而且这
只是把问题推迟到第一步之后。

**C. 固定物理 UV 截断**
用固定 $\Delta x$、改变 $L$ 的 box 测试（$N\propto 1/k_0$，即 `task.md` §6.1 第 2/4/5 项），
使 $\Lambda_{\rm UV}$ 不变，零点能不随 $N$ 变。这是**最无假设**的做法，且与 A 互为交叉
验证。现有 `box_n64_k0_6p25e-9`、`box_n128_k0_3p125e-9` 就是这一类，但方向是更小的盒子；
要做的是 **$N=256,320,400$ 配 $k_0=1.5625,1.25,1.0$ e-9**。

**D. PSpectCosmo 式**：只把过视界模计入能量密度。方案不自洽（时间依赖的减除），
不建议作为主方案，可作对照。

### 4.3 如何判定 §4.2.1 的机制是否成立

最直接的检验：**在 N=400 上加上 A 的减除，重跑**。

- 若 $N=400$ 的谱回到与 N=128/N=300 一致的 2–3%，机制确认，
  §4.2.1 从"候选机制"升级为"已定位的系统误差"，且给出了修正后的收敛性结论；
- 若没有变化，则 §4.2.1 应改写为"该渠道存在但不足以解释约 10%"，转而指向
  随机实现（补 `seed=11,12`）或其他系统误差。

### 4.4 关于现有的 `n128_h400` 对照（重要提醒）

`n128_h400` 把 N=128 的初始 $H$ 强行设成 N=400 的值，但**没有同时改变场的能量**，
因此它不是一个干净的"$H_{\rm ini}$ 通道"探针。日志里可以直接看出来：

| 算例 | step 0 的 $H$ | `hubble_constraint` | `metric_perts` |
|---|---:|---:|---|
| n128 | $2.70102\times10^{-5}$ | $2.3\times10^{-15}$（满足） | $(-5.72\times10^{-8},\,-5.77\times10^{-9})$ |
| n400 | $2.71915\times10^{-5}$ | $-1.5\times10^{-14}$（满足） | $(-5.61\times10^{-7},\,-5.48\times10^{-7})$ |
| n128_h400 | $2.71915\times10^{-5}$ | $-2.9\times10^{-11}$（**违反**） | $(-5.68\times10^{-8},\,-5.74\times10^{-9})$ |

三点：

1. `n128_h400` 的约束残差 $-2.9\times10^{-11}$ 约为 $\rho$ 的 1.3%，且在整个演化中
   一直存在（不收敛到零）；
2. 它的 `metric_perts` 仍是 N=128 的量级，而 N=400 的 `metric_perts` 大了约 10–100 倍
   （比值与 $\rho_{\rm vac}\propto N^4$ 一致，$95\approx(400/128)^4$）——说明在 N=400 里
   真空能**真实存在**于场中并进入度规扰动，而在 `n128_h400` 里只是把 $H$ 抬高；
3. 该算例的谱在窗口内**随 $k$ 单调上升**（$k=2.55\times10^{-9}$ 时 $9.6\times10^{-4}$，
   到 $k=1.4\times10^{-7}$ 升到 3.0），形状与 n128 / n400 完全不同，更像约束被违反后
   的 UV 增长，而不是 USR 放大的物理谱。

因此**不能**用 `n128_h400` 下"$H_{\rm ini}$ 偏移 0.68% 会让谱变 10 倍"的结论，
也不能用它来判断 §4.2.1 的机制对不对。要做干净的对照，应该：

- **方案 A**（减除零点能）在所有 $N$ 上重算，比较减除前后的 $N$ 依赖；或
- 在 N=128 上**自洽地**加上 N=400 那么多真空能（改 `populate_noise` 的归一化，
  而不是改 $H$），使 `hubble_constraint`、`metric_perts` 都与 N=400 一致，只有 $N$ 不同。

前者才是文章需要的那个数。

---

## 5. 引用清单

| 用途 | 文献 |
|---|---|
| 主程序不做重整化 | Figueroa et al., CosmoLattice, arXiv:2102.01031 |
| 综述亦不讨论 | Figueroa et al., The art of simulating the early Universe I/II, arXiv:2006.15122 / 2512.15627 |
| 经典程序同样不做 | Felder–Tkachev (LATTICEEASY) hep-ph/0011159；Easther–Giblin–Lim (PSpectRe) arXiv:1005.1921；Frolov (DEFROST) arXiv:0809.4904 |
| 本方法同源的 TWA 初态 | Caravano–Franciolini–Renaux-Petel, arXiv:2410.23942, 2506.11795 |
| 真空能发散被承认但只做回避 | Jiang, PSpectCosmo, arXiv:2411.17658 |
| 实时格点重整化模板 | Tranberg–Ungersbäck, arXiv:2312.08167, §3.2.2 |
| 经典统计近似不可重整 | Epelbaum–Gelis–Wu, arXiv:1402.0115 |
| 弯曲时空格点 Hadamard 重整化 | Lewis–Vidal, arXiv:1911.12978 |
| 经典格点 vs 2PI | Kainulainen–Nurmi–Väisänen, arXiv:2406.17468 |
| 同类格点系统误差（FLRW 近似） | Barker–Gladwyn–Zell, arXiv:2606.14853 |

---

## 6. 检索复现

```bash
# 关键词统计
curl -sL -o p.pdf https://arxiv.org/pdf/2102.01031
pdftotext -q p.pdf p.txt
grep -ic 'renormali' p.txt; grep -ic 'divergen' p.txt; grep -ic 'vacuum energy' p.txt

# arXiv 元数据检索
https://export.arxiv.org/api/query?search_query=ti:%22lattice%22+AND+ti:%22inflation%22&max_results=50

# 全文检索（INSPIRE，能命中正文）
https://inspirehep.net/api/literature?q=fulltext:%22vacuum%20energy%22%20and%20fulltext:%22truncated%20Wigner%22
```

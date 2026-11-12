# SGB 格点中「宇宙学常数重整化」方案的分析

日期：2026-11-08  
对象：`task.md` §4.2.1 提出的 UV 真空能问题，以及「加 $\int d^4x\sqrt{-g}\,\Lambda$、让 $\Lambda$ 抵消
$(\partial_i\phi)^2$、从而在 $H_{\rm ini}$ 和 $\ddot a$ 中忽略 $\phi$ 的空间分布」这一具体方案。

---

## 0. 结论摘要

**目标是对的，但「用常数 $\Lambda$ 抵消 $(\partial_i\phi)^2$」这个实现不成立。** 具体有四点：

1. **逐点不能抵消。** $\Lambda$ 是常数，$(\partial_i\phi)^2$ 是空间变化的。只有在
   *体积平均* 的 Hubble 约束里，$\Lambda$ 才能和 $\langle(\partial_i\phi)^2\rangle$ 写成同一项；
   逐点的约束（以及代码里逐点求值再平均的 $\ddot a$）没有这个抵消。

2. **只抵消梯度项不够。** BD 真空里动能与梯度能均分：
   $\rho_{\rm grad}=\rho_{\rm kin}=\rho_{\rm vac}/2$。约束里要减的是总真空能
   $\rho_{\rm vac}=\rho_{\rm grad}+\rho_{\rm kin}$，只减 $\rho_{\rm grad}$ 会剩下一半的 $H$ 偏移。

3. **$\ddot a$ 方程里纯梯度项本来就不出现。** 对 $\xi=0$，$\rho+3p=2\dot\phi^2-2V$，
   梯度项严格抵消（下面用代码公式数值验证到机器精度）。一个 $\Lambda$ 与一个 $w$ 组分在
   (约束, 加速) 两式中的系数只有在 $w=1/3$ 时才匹配；纯梯度 $w=-1/3$ 的加速系数为 0，
   所以同一个 $\Lambda$ 不可能同时抵消它。能同时被一个 $\Lambda$ 抵消的，只有
   **辐射型**（$p=\rho/3$）的整块能量。

4. **格点真空能是辐射，不是宇宙学常数。** 格点的 UV 截断是**共动**截断
   $k_{\max}=\sqrt3N/L$，模式频率 $\omega_k=k_{\rm eff}/a$，所以
   $\rho_{\rm vac}\propto a^{-4}$、$p_{\rm vac}=\rho_{\rm vac}/3$，会随膨胀红移掉。
   常数 $\Lambda$ 只能在一个时刻抵消它；之后真空能红移掉了，$\Lambda$ 还留着，
   残留一个常数 $-\rho_{\rm vac,0}$。$N=400$ 时这是 $\delta H/H\simeq-0.68\%$，
   与原效应（$+0.68\%$）同量级、符号相反。**常数 $\Lambda$ 并没有消除 $N$ 依赖，只是把它从
   「初始瞬变」换成了「持续偏移」。**

关于「这就是 CosmoLattice 的做法」：**只对了一半**。CosmoLattice 只用齐次背景定义
$H_{\rm ini}$（源码注释明确写了这一点），但**演化中仍然用全格点平均的梯度能/势能**驱动
$a''$。把 $\ddot a$ 也换成齐次背景是更强的近似（关掉 backreaction），不是 CosmoLattice 的做法。

**建议的最小方案**：只把 $H_{\rm ini}$ 用齐次背景算（等价于在 $t_0$ 做一次重整化条件），
其余不动。因为真空能约一个 e-fold 就红移掉，残留只是一个短瞬变。
若要「真正减除」，应减**瞬时**自由 BD 真空能动张量
$\rho_{\rm vac}(t)=\frac{1}{2V}\sum_k\omega_k(t)$、$p_{\rm vac}=\rho_{\rm vac}/3$，
而不是常数 $\Lambda$。

---

## 1. 作用量、符号与 $\Lambda$ 的符号

采用文章（`main-v2.tex`）的约定，$8\pi G=\kappa=1$，号差 $(-,+,+,+)$：

$$
S[\Lambda]=\int d^4x\sqrt{-g}\left[
\frac{R}{2}-\frac12(\partial\phi)^2-V(\phi)
-\frac{\xi(\phi)}{16}\mathcal G+\Lambda\right],
\qquad
\mathcal G=R^2-4R_{\mu\nu}R^{\mu\nu}+R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}.
$$

对常数 $c$ 的项 $\mathcal L\supset c$，有 $T_{\mu\nu}=c\,g_{\mu\nu}$，与真空能
$T_{\mu\nu}=-\rho_{\rm vac}g_{\mu\nu}$ 比较得

$$
\rho_\Lambda=-c,\qquad p_\Lambda=+c.
$$

所以 **$+\Lambda$ 在拉氏量里对应负的真空能**，确实能抵消正的真空能；但它的压强是
$p_\Lambda=+\Lambda$，即 $p_\Lambda=-\rho_\Lambda$，是宇宙学常数的状态方程。
这一点是后面所有问题的根源。

---

## 2. FLRW 约化：$\Lambda$ 到底进哪两个方程

### 2.1 Gauss-Bonnet 项在 FLRW 中的约化

FLRW 度规 $ds^2=-dt^2+a^2(t)d\vec x^2$ 下

$$
\mathcal G=24H^2(\dot H+H^2),\qquad
a^3\mathcal G=8\frac{d}{dt}(a^3H^3),
$$

其中 $H=\dot a/a$，$\dot H+H^2=\ddot a/a$。用 Peanotica 的 `ComputeCurvature` 独立核对了
$R=6(\dot H+2H^2)$ 以及 $R_{00}=-3\ddot a/a$、$R_{ij}=a^2(\ddot a/a+2H^2)\delta_{ij}$
（见 §6.3）。$\mathcal G$ 的表达式是标准结果，且上面的全导数恒等式可直接代入验证。

### 2.2 Hubble 约束

把 $\sqrt{-g}\mathcal G$ 用 lapse $N$（$dt=Nd\tau$，$H=\dot a/(Na)$）写出来：

$$
-\frac{1}{16}\int dt\,a^3\xi\mathcal G
=-\frac12\int d\tau\,\xi\frac{d}{d\tau}\!\left(\frac{\dot a^3}{N^3}\right).
$$

注意这里**不能**先分部积分再变分 $N$（边界项对 lapse 变分有贡献），这正是 $\frac32$ 因子的来源。
对 $N$ 变分并令 $N=1$：

$$
\frac{\delta S}{\delta N}\Big|_{N=1}
=3a^3H^2-a^3\rho_\phi+a^3\Lambda-\frac32a^3\dot\xi H^3=0,
$$

即

$$
\boxed{\;3H^2=\frac12\dot\phi^2+V(\phi)-\Lambda+\frac32H^3\dot\xi(\phi)\;}
\tag{2.1}
$$

齐次极限下 $\dot\xi=\xi'\dot\phi$，与文章的
$3H^2=\frac12\dot\phi^2+V+\frac32H^3\xi'\dot\phi$ 一致（$\Lambda=0$）。
$\Lambda$ 以 $-\Lambda$ 的形式进入能量密度，和 $\frac12\dot\phi^2$、$V$ 并列。

### 2.3 加速方程

$\Lambda$ 只通过 $T_{\mu\nu}\to T_{\mu\nu}+\Lambda g_{\mu\nu}$（等价于 $G_{\mu\nu}\to G_{\mu\nu}-\Lambda g_{\mu\nu}$）进入 Einstein 方程，
GB 部分不受影响。因此 $\rho_{\rm tot}=\rho_\phi-\Lambda$、$p_{\rm tot}=p_\phi+\Lambda$，

$$
\frac{\ddot a}{a}=-\frac16(\rho_{\rm tot}+3p_{\rm tot})
=-\frac16(\rho_\phi+3p_\phi)-\frac13\Lambda .
\tag{2.2}
$$

把 $\Lambda$ 看作 $\rho_\Lambda=-\Lambda$、$p_\Lambda=+\Lambda$ 的组分。它在两式中的贡献是

$$
\text{约束}: +\rho_\Lambda=-\Lambda,\qquad
\text{加速}: -\frac16(\rho_\Lambda+3p_\Lambda)=-\frac13\Lambda .
$$

对一般的 $p=w\rho$ 组分，贡献是 $\rho$（约束）和 $-\frac16(1+3w)\rho$（加速）。
要用一个 $\Lambda$ 同时抵消组分 $X$，需要

$$
-\Lambda=-\rho_X,\qquad
-\frac13\Lambda=-\frac16(1+3w_X)\rho_X,
$$

即 $\Lambda=\rho_X$ 且 $\Lambda=\frac12(1+3w_X)\rho_X$，相容**当且仅当 $w_X=1/3$**。

- 辐射（$w=1/3$）：可以；
- 纯梯度（$w=-1/3$，见 §4）：加速一侧要求 $\Lambda=0$，与约束一侧矛盾；
- 动能（$w=1$）：需要 $\Lambda=2\rho_{\rm kin}$，也与约束一侧矛盾。

**所以一个 $\Lambda$ 能同时抵消的，只有 $w=+1/3$ 的辐射型整块能量；梯度型（$w=-1/3$）
在加速方程里系数为 0，不可能被 $\Lambda$ 同时抵消。**

### 2.4 场方程不受 $\Lambda$ 影响

$\Lambda$ 是常数，不进入 $\phi$ 的变分方程；格点上就是

$$
\ddot\phi+3H\dot\phi+\partial_\phi V_{\rm eff}(\phi,H)-\frac{\nabla^2\phi}{a^2}=0
$$

（外加 SGB 的 $\Box\phi$ 项，见文章附录）。代码里的 `vv_phi()` 也确实不含 $\Lambda$。
所以「重整化」只改变背景 $a(t)$，再通过 $H(t)$ 间接影响 $\phi$ 的演化；
它不会直接修改涨落的传播方程。这一点对方案 B 很重要：减除真空能只改背景源，
不需要（也不应该）改 $\phi$ 的 EOM。

---

## 3. 格点上的具体形式（对照代码）

### 3.1 约束多项式

`inflat/src/gauss_bonnet.rs` 的 `hubble_constraint_coefs` 在 $d=3$、$\kappa=1$ 时给出

$$
c_0=\frac12\left(\dot\phi^2+\frac{(\nabla\phi)^2}{a^2}\right)+V,\quad
c_1=0,\quad
c_2=-\frac12\left[6+\frac{\nabla^2\phi\,\xi'}{a^2}
+\frac{(\nabla\phi)^2\xi''}{a^2}\right],\quad
c_3=\frac32\dot\phi\xi',\quad
c_4=0,
$$

约束为 $P(H)=c_0+c_2H^2+c_3H^3=0$，即

$$
3H^2=\frac12\dot\phi^2+\frac{(\nabla\phi)^2}{2a^2}+V
-\frac12H^2\left(\frac{\nabla^2\phi\,\xi'}{a^2}
+\frac{(\nabla\phi)^2\xi''}{a^2}\right)
+\frac32H^3\dot\phi\xi' .
\tag{3.1}
$$

初始化时 $a=1$，代码传的正是未除 $a^2$ 的 `derivative_square` 和 `laplacian`。
$\Lambda$ 只改 $c_0\to c_0-\Lambda$，也就是只和第一行的
$\frac12(\nabla\phi)^2/a^2$ 并列；$H^2$ 前的两个 GB 梯度项**不受 $\Lambda$ 影响**。
这是「$\Lambda$ 只抵消 $(\partial_i\phi)^2$」在 SGB 里已经不严格的第一层原因。

> 代码一致性备注：`hubble_constraint()`（诊断量）把 `derivative_square`/`laplacian`
> 直接当作物理量传入，没有除 $a^2$；而演化用的 `delta()` 里 `hubble2()` 是除 $a^2$ 的。
> 初始化 $a=1$ 时两者一致；若要在 $a\gg1$ 时用 `hubble_constraint` 做检验，需要先补上 $1/a^2$。

### 3.2 加速方程

代码的 `hubble2()` 是 $\ddot a/a$ 的闭式表达（由 $ij$ 方程消去 $\Box\phi$ 得到）。
在 $\xi=0$ 时它可以化简为

$$
\frac{\ddot a}{a}=-\frac16(\rho+3p),\qquad
\rho=\frac12\dot\phi^2+\frac{(\nabla\phi)^2}{2a^2}+V,\quad
p=\frac12\dot\phi^2-\frac{(\nabla\phi)^2}{6a^2}-V .
\tag{3.2}
$$

注意 $p$ 里梯度项的系数是 $-1/6$（各向同性平均后），所以

$$
\rho_g+3p_g=\frac12g^2-\frac12g^2=0,\qquad g^2\equiv\frac{(\nabla\phi)^2}{a^2}.
$$

**纯梯度对 $\ddot a/a$ 的贡献严格为零。** 我用代码的 `hubble2` 公式做了数值核对（§7.1）：
在约束满足时，随机取 6 组参数，代码值与 $-\frac16(\rho+3p)$ 的差为 $0$ 或 $\sim10^{-18}$；
固定 $V$、把 $g^2$ 从 $0$ 改到 $10^{-10}$，$\ddot a/a$ 恒为 $3.3333\times10^{-10}$。

$\xi\ne0$ 时，梯度通过 $\langle(\Box\phi)\xi'\rangle$、$\langle\dot\phi^2\xi''\rangle$
等项进入，但：这些项本身是 SGB 修正（本文参数下量级估计 $\sim10^{-4}$），
而且 $\Lambda$ 同样不能逐项抵消它们。

**结论：在 $\ddot a$ 方程里，$\Lambda$ 与 $(\partial_i\phi)^2$ 既不「同时出现」，
系数也不匹配。**

---

## 4. 格点 BD 真空是什么：辐射，不是宇宙学常数

`populate_noise` 逐模填充 BD 真空，$\omega_k=k_{\rm eff}/a$，$k_{\rm eff}$ 用格点色散关系。
对每个模，$\dot\phi_k^2=k_{\rm eff}^2\phi_k^2/a^2$，所以动能=梯度能（均分）。于是

$$
\rho_{\rm vac}=\frac{1}{2V}\sum_k\omega_k,\qquad
p_{\rm vac}=\frac{1}{3}\rho_{\rm vac},\qquad
\rho_{\rm vac}\propto a^{-4}.
\tag{4.1}
$$

$\rho_{\rm vac}\propto a^{-4}$ 的原因很直接：格点的 UV 截断是**共动**的
（$k_{\max}=\sqrt3N/L$ 固定），物理动量 $k/a$ 随膨胀红移，模式不被补充。
这与「物理截断下的 QFT 真空能是常数」不同；在格点上，它就是一个辐射型的初始能量。

用日志（`inflat/out/gauss_bonnet_convergence/*.run.log`）的实测值：

| $N$ | $\langle(\nabla\phi)^2\rangle/a^2$ | $\rho_g=\frac12\langle g^2\rangle$ | $\rho_{\rm vac}=2\rho_g$ | $\rho_{\rm vac}/\rho$ | 实测 $\delta H_{\rm ini}/H$ |
|---:|---:|---:|---:|---:|---:|
| 300 | $9.42853\times10^{-12}$ | $4.714\times10^{-12}$ | $9.429\times10^{-12}$ | $0.431\%$ | $0.215\%$ |
| 400 | $2.97906\times10^{-11}$ | $1.490\times10^{-11}$ | $2.979\times10^{-11}$ | $1.362\%$ | $0.678\%$ |

其中 $\rho=3H_{\rm bg}^2=2.1881\times10^{-9}$，$H_{\rm bg}=2.7006884\times10^{-5}$。
$\delta H/H\simeq\frac12\rho_{\rm vac}/\rho$，与表格最后两列一致，验证了
「总真空能 = 2×梯度能」的均分图像。
用连续估计核对量级：$\rho_{\rm vac}\simeq k_{\max}^4/(16\pi^2)$，
$k_{\max}=2\sqrt3/\Delta x$。$N=400$ 时 $\Delta x=447.4$、$k_{\max}=7.74\times10^{-3}$，
得 $2.3\times10^{-11}$，与实测 $2.98\times10^{-11}$ 差约 $25\%$
（来自格点色散与锐截断的 $O(1)$ 系数），量级正确。

**辐射型意味着它约一个 e-fold 就红移掉：**
$a^{-4}$ 在 $\Delta N=1$ 时为 $1.83\times10^{-2}$，在 $\Delta N=3$ 时为 $6.1\times10^{-6}$。
本文关心的模式在 $\ln a\simeq1.9$–$3.5$ 出视界，那时真空能已经可以忽略。

---

## 5. 逐条检验提议

### 5.1 「$\Lambda$ 抵消 $(\partial_i\phi)^2$」

在体积平均的约束 (3.1) 里，取 $\Lambda=\frac12\langle g^2\rangle$ 确实消掉梯度项。
但：

- 逐点不成立；
- 只消掉一半的真空能（$\rho_{\rm kin}=\rho_{\rm grad}$ 还在），$H$ 仍偏高
  $\frac12\rho_g/\rho\simeq0.34\%$（$N=400$）；
- 同一个 $\Lambda$ 在 $\ddot a$ 方程里产生 $-\frac13\Lambda$，而梯度项原本贡献 $0$，
  等于凭空引入一个 $-\rho_g/\rho\simeq-0.68\%$ 的 $\ddot a/a$ 偏差。

所以「只抵消梯度」是这几种做法里最不自洽的一个：约束里少减一半，加速里多减一份。

### 5.2 「$\ddot a$ 方程里 $\Lambda$ 也抵消 $(\partial_i\phi)^2$」

不成立，见 §2.3、§3.2：纯梯度在 $\ddot a/a$ 里系数为零；$\Lambda$ 的系数是 $-1/3$。
若想同时抵消，必须减的是**总**真空能（$w=+1/3$），此时
$\Lambda=\rho_{\rm vac}$ 在两式里给出的量值 $1$、$1/3$ 都与真空能相同，正好抵消。

### 5.3 「等价于忽略 $\phi$ 的空间分布」

不等价。

- 常数 $\Lambda$：在约束里只改 $c_0$，在 $\ddot a$ 方程里加一个常数源；两者在 $t_0$
  之后都还留着，残留常数 $-\rho_{\rm vac,0}$ 的能量。
- 忽略空间分布：把 (3.1)、(3.2) 里所有涨落项都换成齐次量，等价于
  $\rho\to\rho_{\rm bg}$、$p\to p_{\rm bg}$，是**关掉 backreaction**，不是减一个常数。

后者是一个自洽的「背景 + 试探场」近似：$a(t)$ 解齐次方程，$\phi$ 的涨落在该背景上演化，
$\delta N$ 本来就逐点用齐次背景方程算（`zeta_at_site`），所以内部是自洽的。
它确实完全去掉 $N$ 依赖。代价是同时丢掉了物理的 IR backreaction。
就本文而言，场涨落的能量是 $\sim H^4$ 量级（$\delta\phi\sim H/2\pi$，
而 $\zeta$ 的大值来自 $H/\dot\phi$ 而非 $\delta\phi$ 本身），相对背景
$\sim(H/M_p)^2\sim10^{-10}$，所以这个代价在数值上很小；
但「小」是物理结论，不是「$\Lambda$ 抵消」的代数结果。

### 5.4 「这就是 CosmoLattice 的做法」

`lit-survey-renormalization.md` §2(a)、§2bis 已经核对过源码：

- CosmoLattice：**只**用齐次势能定义 $H_{\rm ini}$；演化中 $a''$ 仍用全格点平均的
  `gradient_energy() + potential_energy()`（InflationEasy 的 `initializef`/`inflation.cpp` 同）。
- 本文代码：`populate_noise` → `solve_hubble` 把噪声的梯度、动能、势能一起代入约束，
  所以 $H_{\rm ini}$ 含真空能；演化 `delta` 里 `hubble2` 逐点求值再平均，也含涨落。

因此「$H_{\rm ini}$ 用齐次」= CosmoLattice；「$\ddot a$ 也用齐次」= 比 CosmoLattice 更强，
是本文可以选、但需要明确声明的近似。

### 5.5 常数 $\Lambda$ 的定量残留

若在 $t_0$ 近似取 $\Lambda=\rho_{\rm vac,0}$（忽略 GB 涨落修正，这是让 $H_{\rm ini}=H_{\rm bg}$ 的条件）：

$$
\Delta\rho(t)=\rho_{\rm vac,0}\left(a^{-4}-1\right)
\;\xrightarrow[a\gg1]{}\;-\rho_{\rm vac,0}.
$$

$N=400$ 时 $\rho_{\rm vac,0}/\rho=1.362\%$，所以晚期

$$
\frac{\delta H}{H}\simeq-\frac12\frac{\rho_{\rm vac,0}}{\rho}=-0.68\%,
\qquad
\frac{\delta(\ddot a/a)}{\ddot a/a}\simeq-\frac{\rho_{\rm vac,0}}{\rho}=-1.36\% .
$$

这与原效应（$+0.68\%$）同量级、符号相反，而且**残留本身仍按 $N^4$ 增长**，所以并没有消除 $N$ 依赖。
换句话说，常数 $\Lambda$ 把「初始时刻偏高 0.68%」换成了「整个演化偏低 0.68%」，
没有消除 $N$ 依赖。$N=300$ 对应 $\pm0.215\%$，结论相同。

---

## 6. 建议的实现与验证

### 6.1 方案 A（最小、推荐、= CosmoLattice 的 $H_{\rm ini}$ 部分）

在 `populate_noise` 里不要用含噪声的场重解 $H$，直接用背景态的 $H_{\rm bg}$：

```rust
// 现在:
self.v_a = self.a * solve_hubble(..., 含噪声的平均量 ..., hubble_initial_guess);
// 改为:
self.v_a = background_hubble * self.a;   // 齐次背景的 H
```

其余（演化、$\delta N$）不动。理由：真空能是辐射型、一个 e-fold 内红移掉，
它唯一的持久印记就是 $H_{\rm ini}$；把 $H_{\rm ini}$ 定准，剩下的是短瞬变。
这与 CosmoLattice 一致，且不需要引入任何 $\Lambda$。

### 6.2 方案 B（真正的「减除」，若审稿人要求）

在约束和 $\ddot a$ 方程中同时减掉**瞬时**自由 BD 真空能动张量：

$$
\rho_{\rm vac}(t)=\frac{1}{2V}\sum_{\vec k}\frac{k_{\rm eff}}{a(t)},
\qquad
p_{\rm vac}(t)=\frac{1}{3}\rho_{\rm vac}(t).
$$

实现上：

- `solve_hubble`：把 $\langle\dot\phi^2\rangle$ 和 $\langle g^2\rangle$ 各减去
  $\rho_{\rm vac}$（因为均分，各占 $\rho_{\rm vac}$；注意 $\frac12\langle\dot\phi^2\rangle
  =\rho_{\rm vac}/2$，故 $\langle\dot\phi^2\rangle=\rho_{\rm vac}$）。
- `hubble2`：在平均后的 $\ddot a/a$ 上加 $+\frac13\rho_{\rm vac}(t)$
  （抵消真空能原本贡献的 $-\frac13\rho_{\rm vac}$）。
- GB 项里的真空贡献是 $\mathcal O(10^{-4})$ 相对量，可一并忽略，或按同一 prescription 处理。

这个减除是自洽的：$\rho_{\rm vac}\propto a^{-4}$ 满足
$\dot\rho_{\rm vac}+3H(\rho_{\rm vac}+p_{\rm vac})=0$，不需要额外的补偿项。
它减掉的只是格点 UV 真空能这个 artifact，保留了物理的 IR backreaction。

### 6.3 方案 C（用户提议的「忽略空间分布」）

如果坚持这么做，应**同时**在 $H_{\rm ini}$ 和 $\ddot a$ 里做，并在文章中明确写成
「no-backreaction / 背景+试探场近似」，而不是「$\Lambda$ 重整化」。
数值上它与方案 A/B 的差别只有 IR backreaction，预计远小于当前的 $2$–$3\%$ 收敛水平，
但需要在文中说明，否则「$\Lambda$ 抵消」的推导会被审稿人抓住。

### 6.4 判定 §4.2.1 机制的实验

在 $N=400$ 上分别跑：

1. 方案 A（只改 $H_{\rm ini}$）；
2. 方案 B（全时减除）；
3. 方案 C（完全忽略空间分布）。

- 若 1/2 让 $N=400$ 回到 $N=128/300$ 的 $2$–$3\%$，机制确认；
- 若三者都仍有约 $10\%$ 偏差，则 §4.2.1 的候选机制不成立，应转向 seed / 盒子 / 其他系统误差。
- 预期 1 与 2 的差别只来自最初约一个 e-fold 的 transient：方案 A 保留真空能驱动
  $a''$，方案 B 把它减掉；两者对总 e-fold 数的差别约 $\Delta N\sim10^{-2}$。
  若这个差别仍足以解释 $10\%$，说明 USR 对初始 $H$ 极其敏感，这本身也是有用的信息。

---

## 7. 数值核对与复现

### 7.1 代码约束与 $\ddot a$ 的核对

`papers/gauss-bonnet/check_lattice_equations.py` 做两件事：

1. 用 `hubble_constraint_coefs` 的公式构造 $P(H)$，随机取点求正实根，检查
   $P(H)=0$；
2. 在同一个 on-shell 点上用 `hubble2` 公式算 $\ddot a/a$，与
   $-\frac16(\rho+3p)$ 比较。

结果（$\xi=0$，6 组随机参数）：差为 $0$ 或 $\le3.5\times10^{-18}$。
梯度敏感性：固定 $V$，$g^2=0,\,10^{-11},\,10^{-10}$ 时
$\ddot a/a=3.3333333333\times10^{-10}$ 不变。

### 7.2 数值表

见 §4、§5.5。$\rho=3H_{\rm bg}^2$，$H_{\rm bg}=2.7006884377829143\times10^{-5}$，
$H_{\rm ini}$ 取自 `n300.run.log`、`n400.run.log`。

### 7.3 Peanotica 核对与一个可疑点

用 `Peanotica` 的 `ComputeCurvature` 在 FLRW 度规上核对了
$R=6(\dot H+2H^2)$、$R_{00}=-3\ddot a/a$、$R_{ij}=a^2(\ddot a/a+2H^2)\delta_{ij}$，
均正确。

但 `ComputeCurvature["Riemann"]` 返回的分量不是全下指标，而是（至少对球面自洽地）
**最后一个指标抬升**的 $R_{\mu\nu\rho}{}^{\sigma}$。最小复现（单位二维球面，
全下指标应为 $R_{1212}=-R_{1221}=R_{2121}=\sin^2\theta$）：

```mathematica
R[1,2,1,2] = 1                 (* 期望 Sin[t]^2 *)
R[1,2,2,1] = -Sin[t]^2
R[2,1,2,1] =  Sin[t]^2
```

三个值与 $R_{121}{}^{2}=g^{22}R_{1212}=1$、$R_{122}{}^{1}=-\sin^2$、
$R_{212}{}^{1}=+\sin^2$ 一致。也就是说数组返回的是 $R_{\mu\nu\rho}{}^{\sigma}$，
而 `MakeCurvatureValueRules` 的抽象表达式 `Rsph[DI@a,DI@b,DI@c,DI@d]` 是全下指标；
两者直接相乘会得到量纲不齐的结果（FLRW 上出现 $a^4-1$ 因子）。
我因此没有用 `ComputeCurvature["Riemann"]` 验证 $\mathcal G$，
改用标准恒等式 $\mathcal G=24H^2(\dot H+H^2)$（$R$ 和 Ricci 已由 Peanotica 独立确认）。
这不是 $R$/$Ricci$ 的 bug，但 `Riemann` 的约定容易踩坑，使用前应先确认。

---

## 8. 一句话总结

$\Lambda$ 是一个**常数、$w=-1$** 的局部反项；格点 BD 真空能是**辐射型、$w=+1/3$、
$\rho\propto a^{-4}$** 的初始能量。两者只有在 $t_0$ 时刻、且减除的是**总**真空能时才重合。
「$\Lambda$ 抵消 $(\partial_i\phi)^2$」既漏掉了动能那一半，又在 $\ddot a$ 方程里用错了系数；
「忽略空间分布」则是另一件事（关掉 backreaction），不是重整化。
干净的做法是：$H_{\rm ini}$ 用齐次背景（方案 A），或全时减除瞬时 BD 真空能动张量（方案 B）。

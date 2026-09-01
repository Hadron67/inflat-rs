# 格点哈密顿量分解与 Yoshida 保辛演化格式

## 1. 模型与离散化设定

考虑单场暴胀模型，作用量为  
\[
S=\int d^{d+1}x\sqrt{-g}\left[\frac{1}{2\kappa}R-\Lambda-\frac12 g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi-V(\phi)\right]
\]
空间平坦 FLRW 度规，宇宙学时间 \(N=1\)：
\[
ds^2=-dt^2+a^2(t)\delta_{ij}dx^i dx^j .
\]
空间维数为 \(d\)，空间体积 \(V=L^d\)，格点数 \(M=N^d\)，格距 \(h=L/N\)，每个格点体积 \(h^d\)。

## 2. 辅助变量与离散哈密顿量

为消除尺度因子动能项中 \(a\) 与 \(\pi_a\) 的耦合，引入辅助变量  
\[
b=a^{d/2},\qquad \pi_b=\frac{2}{d}a^{1-d/2}\pi_a .
\]
场量 \(\phi_i\) 定义在格点 \(i\)，其共轭动量为  
\[
p_i=\pi_\phi(x_i)\,h^d .
\]

离散化后的总哈密顿量为  
\[
H=K_1+K_2+K_3,
\]
其中  
\[
\begin{aligned}
K_1&=-\frac{\kappa d}{8(d-1)V}\,\pi_b^2,\\[4pt]
K_2&=\frac{1}{2b^2 h^d}\sum_i p_i^2,\\[4pt]
K_3&=\frac12 b^{2-4/d}h^{d-2}\sum_{\langle ij\rangle}(\phi_i-\phi_j)^2
+b^2 h^d\sum_i\bigl[V(\phi_i)+\Lambda\bigr].
\end{aligned}
\]
这里 \(\langle ij\rangle\) 表示最近邻格点对，\(\pi_b\) 为全局共轭动量，与 \(b\) 共轭。

每一项都不含同一对共轭坐标：
- \(K_1\) 只含 \(\pi_b\);
- \(K_2\) 含 \(p_i\) 和 \(b\)，不含 \(\phi_i,\pi_b\);
- \(K_3\) 含 \(\phi_i\) 和 \(b\)，不含 \(p_i,\pi_b\)。

因此满足 Yoshida 保辛组合的要求。

## 3. 各 \(K_n\) 的精确更新格式

设时间步长为 \(\tau\)，以下所有更新均在旧值上进行。

### 3.1 \(K_1\) 演化

运动方程：
\[
\dot b=\frac{\partial K_1}{\partial \pi_b}=-\frac{\kappa d}{4(d-1)V}\pi_b,\qquad
\dot\pi_b=0 .
\]
更新为  
\[
b \leftarrow b-\frac{\kappa d}{4(d-1)V}\pi_b\,\tau
\]
其他变量不变。

### 3.2 \(K_2\) 演化

运动方程：
\[
\dot\phi_i=\frac{\partial K_2}{\partial p_i}=\frac{p_i}{b^2 h^d},\qquad
\dot p_i=0,\qquad
\dot b=0,\qquad
\dot\pi_b=-\frac{\partial K_2}{\partial b}
=\frac{\sum_i p_i^2}{b^3 h^d}.
\]
更新为  
\[
\phi_i \leftarrow \phi_i+\frac{p_i}{b^2 h^d}\,\tau,
\qquad
\pi_b \leftarrow \pi_b+\frac{\sum_i p_i^2}{b^3 h^d}\,\tau
\]
其他变量不变。

### 3.3 \(K_3\) 演化

运动方程中 \(\dot\phi_i=0,\ \dot b=0\)，只有动量改变：
\[
\dot p_i=-\frac{\partial K_3}{\partial \phi_i},\qquad
\dot\pi_b=-\frac{\partial K_3}{\partial b}.
\]
计算偏导数：
\[
\frac{\partial K_3}{\partial \phi_i}
=
b^{2-4/d}h^{d-2}\sum_{j\sim i}(\phi_i-\phi_j)
+b^2 h^d V'(\phi_i),
\]
\[
\frac{\partial K_3}{\partial b}
=
\left(1-\frac2d\right)b^{1-4/d}h^{d-2}
\sum_{\langle ij\rangle}(\phi_i-\phi_j)^2
+2b h^d\sum_i\bigl[V(\phi_i)+\Lambda\bigr].
\]
更新为  
\[
p_i \leftarrow p_i-\tau\left[
b^{2-4/d}h^{d-2}\sum_{j\sim i}(\phi_i-\phi_j)
+b^2 h^d V'(\phi_i)
\right]
\]
\[
\pi_b \leftarrow \pi_b-\tau\left[
\left(1-\frac2d\right)b^{1-4/d}h^{d-2}
\sum_{\langle ij\rangle}(\phi_i-\phi_j)^2
+2b h^d\sum_i\bigl(V(\phi_i)+\Lambda\bigr)
\right]
\]
其他变量不变。

## 4. Yoshida 组合

采用四阶 Yoshida 方法，系数为  
\[
w_1=w_4=\frac{1}{2-2^{1/3}},\qquad
w_2=w_3=-\frac{2^{1/3}}{2-2^{1/3}}.
\]
一个时间步的演化算子可近似为  
\[
e^{\tau H}\approx 
e^{w_1\tau K_1}
e^{w_2\tau K_2}
e^{w_3\tau K_3}
e^{w_4\tau K_2}
e^{w_3\tau K_3}
\]
或采用其他对称排列。实际计算中依次调用各 \(K_n\) 的更新子程序，步长乘以相应系数。

## 5. 备注

- 引力动能项符号为负，这是引力约束动力学的标准结果。
- 上述离散化使用前向差分（求和后等价于后向差分），变分后自然得到标准三点拉普拉斯，保持哈密顿结构，适合保辛演化。

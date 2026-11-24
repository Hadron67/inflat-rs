# `(\nabla\phi)^2` 离散化修正的验证脚本

对应任务记录 §4.2.10（`papers/gauss-bonnet/task.md`）。

## 文件

| 文件 | 作用 |
|---|---|
| `extract_expr.py` | 从 `inflat/src/gauss_bonnet.rs` 逐字抽取 `vv_phi` / `hubble2` / `hubble_constraint_coefs` 的返回表达式，生成 `gb_expr.py`（避免手工转录公式）。 |
| `gb_expr.py` | 生成物，不要手改；`d = 3` 是**空间**维数（`GaussBonnetBInput.dim`）。 |
| `init_h_check.py` | 复现 N=400 初始态（BD 谱 + 同一随机种子的统计），分别用四种组合解 Hamiltonian 约束，给出两处修正对 $H_{\rm ini}$ 的影响。 |

## 运行

```bash
cd papers/gauss-bonnet/grad-fix
python3 extract_expr.py      # 仅当 gauss_bonnet.rs 改了才需要
python3 init_h_check.py      # 约 1–3 分钟（N=400，64M 格点）
```

## 结论（详见 §4.2.10）

- 前向 → 对称梯度：$H_{\rm ini}$ 变化 $10^{-11}$ 量级，$\delta(a''/a)\sim8\times10^{-9}$。
  原因是差值是一个格点全导数，无权平均**精确为零**。
- `LatticeLaplacian::get` 修正（原来返回 `derivative_square_at`）：$H_{\rm ini}$ 变化
  $+4.2\times10^{-6}$（N=400），并恢复
  $\langle\Delta\phi\,\xi'\rangle+\langle(\nabla\phi)^2\xi''\rangle\approx0$。
- 两者都远小于 §4.2.1 的 N⁴ 真空项（N=400：$6.8\times10^{-3}$）和待解释的 9% 谱亏损，
  **不是 N=400 偏低的原因**。

脚本复现的初始态与真实日志一致：

| N | $\langle(\nabla\phi)^2\rangle$ | $H_{\rm ini}$ |
|---:|---|---|
| 400（脚本） | $2.97976\times10^{-11}$ | $2.71901488\times10^{-5}$ |
| 400（`n400.run.log`） | $2.97906\times10^{-11}$ | $2.71901044\times10^{-5}$ |

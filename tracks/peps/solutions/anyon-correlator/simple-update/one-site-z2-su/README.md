# 原生虚拟 Z₂ 的单组 V/P/λ 更新

本目录研究严格零场 $H=-\sum_s A_s-\sum_p B_p$，Pauli convention，$J_e=J_m=1$，无限方格，边上自旋，$d=D=2$。状态只保存一组 V、P、λ，分别采用 `N⊗E ← S⊗W`、`physical⊗head ← tail`、虚拟空间上的对角映射。每条虚拟腿的偶、奇 charge 各一维，两个物理态均取平凡表示。V 保持 90°循环旋转，P 保持端点交换，横竖边共用 P。

随机初态只在允许的 charge 块内取正态随机数，随后施加空间对称投影。解析 X-basis copy 张量仅供代数测试。导出数组通过 TensorKit 的公开转换取得，指标顺序为 `V[N,E,S,W]` 和 `P[physical,head,tail]`，不重排底层块存储。

Star 更新将四条 P 与中心 V 收缩，内部和外部连接各计入一次 λ，对四个物理自旋施加 $\exp(\tau A_s)$。随后对四个“物理腿、外侧虚拟腿”分组分别做 TensorKit 高阶 SVD，每组固定保留一偶、一奇。Plaquette 更新沿四个 V、四个 P 的闭环构造 $\exp(\tau B_p)$ 目标，在允许 charge 块内逐因子求线性最小二乘解；求逆使用分块支撑伪逆。环境仍是区域外独立的对角 λ。

闭环规范由左右 cavity 密度矩阵确定，它们是沿环传播后归一化的虚拟约化矩阵。各 charge 块的正平方根和 SVD 给出成对变基与新的连接权重。纯规范变化以加权区域张量的相对差检查；随后恢复共享参数和空间投影产生的变化单列报告。完整 sweep 在副本上执行，任一子更新失败都不提交。

## 执行与当前限制

从仓库根运行轻量检查：

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/one-site-z2-su/tests.jl
```

`--group=tensors`、`--group=updates`、`--group=runner` 分别运行表示、局部更新和保存/验收检查。正式优化入口为 `run.jl`；`--resume=<前段 final.json 或 latest.json>` 继续同一数值态，`--steps` 为本段新增轮数，每段写入全新目录并保存累计轮数和父状态哈希。

用户已授权分段增加累计预算。实际 seed=5、dt=0.05、ALS=12 的同一轨迹已按 300+300+100 轮完成累计 **700 轮**，结果保存在 `results/zero-z2-seed5-segment{1,2,3}/`。终态得到 $e\approx-0.5$、star 局部残差 $2.90\times10^{-15}$、plaquette 局部残差 $\sqrt2$；两种虚拟 charge 的物理 P 列均趋向 $|+\rangle$，物理验收失败。χ=8 的 CTMRG 环境残差约 $7.13\times10^{-11}$，χ=16 未收敛（0.04633）；没有按较低能量选择环境分支。状态保持 `accepted=false`，不能作为正式转移谱输入。

第 700 轮的独立零更新测量保存在 `results/zero-z2-seed5-final-audit/`；运行器在后续汇总时触及预算，其原始 budget 状态保留。`results/zero-z2-diagnostics-20261023/physics-audit-700.json` 是另行读取已保存测量并重算局部残差的完整只读审查，明确拒绝该状态。该目录也保存了逐轮诊断图、CSV、延续哈希核对和复现脚本。当前原生专项测试 90/90 通过；下一次随机初始化需要用户另行授权。

当四条物理边均处于 $|+\rangle$ 时，plaquette 门的目标包含 $\cosh\tau|++++\rangle+\sinh\tau|----\rangle$。固定其余三个物理因子、只改变一个因子的 ALS 方向，与 $|----\rangle$ 分量正交，因此这个秩亏物理映射可能成为逐张量 ALS 的驻点。此时小规范误差、稳定的权重或充分收敛的 star 均不能表示零场基态。执行进度、诊断与实际检查见 [进度记录](../../docs/2026-10-23-zero-field-z2-progress.md)。

复现第一段（使用全新目录，不覆盖现有记录）：

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/one-site-z2-su/run.jl --seed=5 --steps=300 --dt=0.05 --als-sweeps=12 --budget=600 --out=tracks/peps/solutions/anyon-correlator/simple-update/one-site-z2-su/results/reproduce-segment1
```

继续同一态时增加 `--resume=tracks/peps/solutions/anyon-correlator/simple-update/one-site-z2-su/results/reproduce-segment1/final.json`，其余参数不变，改用另一个全新 `--out`。验收以 `summary.json` 的 `passed` 及末态 metadata 中的 `accepted` 为准；进程正常结束本身不表示基态正确。

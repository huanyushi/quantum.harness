# 有限场 loop/cluster update benchmark

**结果：低场下有近似有效性，但尚未通过可控高精度和一般 D 的完整验证。** 三个磁场点在 D=2 从相同随机 tensors 开始；D=3 从对应 D=2 态扩大 bonds，D=4 在双分量点做有界延续。D=3 的主要磁化偏差约 0.77%；D=4 的短运行和 CTMRG 分支问题尚未解决。

完整结果见 **[REPORT.md](REPORT.md)**。逐点汇总见 [benchmark_summary.csv](benchmark_summary.csv)，完整 contraction 分支记录见 [contraction_audit.csv](contraction_audit.csv)。灰色/未接受数据不能当作已收敛结果。

**本地 VUMPS 后续已完成：** [VUMPS_REPORT.md](VUMPS_REPORT.md) 给出独立上下边界、gauge balancing、χ/初态核对。D=3 主分支获得约 10⁻¹¹ 量级的 CTMRG/VUMPS 一致性支持；D=4 仍未通过。`frozen_states/` 保存可移植 physical snapshots，无需重跑旧的时间中止轨迹。

直接重现 D=3 冻结态核对：

```bash
OPENBLAS_NUM_THREADS=1 julia -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/finite-field-loop-update/vumps_run.jl --state=d3-xz --chi=16 --seed=2 --tol=1e-9 --maxiter=100 --balanced=true --portable=true --budget=300
```

## 运行

从 repository root，使用现有 Julia environment：

```bash
# 第一阶段：三个 D=2 随机初始化场点，含 fresh χ=8/16 测量
OPENBLAS_NUM_THREADS=1 julia -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/finite-field-loop-update/reproduce.jl d2

# D=2 双分量点，等追加 imaginary time 的 dt/ALS 对照
OPENBLAS_NUM_THREADS=1 julia -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/finite-field-loop-update/reproduce.jl stability

# 更大 D 分点运行；每条都从对应 D=2 checkpoint 开始
OPENBLAS_NUM_THREADS=1 julia -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/finite-field-loop-update/reproduce.jl d3-z
OPENBLAS_NUM_THREADS=1 julia -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/finite-field-loop-update/reproduce.jl d3-x
OPENBLAS_NUM_THREADS=1 julia -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/finite-field-loop-update/reproduce.jl d3-xz

# 数学/实现检查；已完成记录的汇总与绘图
julia --project=julia-env tracks/peps/solutions/anyon-correlator/finite-field-loop-update/tests.jl
.venv/bin/python tracks/peps/solutions/anyon-correlator/finite-field-loop-update/analyze.py
```

绘图脚本汇总的是包括诊断分支在内的完整记录；从全新 checkout 复现诊断数据时，还需执行报告 §8 中的相应命令。`results/` 和 binary checkpoints 是本地生成文件，按仓库规则忽略；根目录 CSV 和 SVG 保存本次汇总证据。

## 实现

- `core.jl`：任意均匀 D 的 rank-4 V、rank-3 P、显式 λ；star HOSVD；精确 field gate。
- `loop.jl`：完整八张量 plaquette ALS、BP regauging、分离的拟合/重构诊断；QR 减小 least-squares 系统但不改变其数学目标。
- `reference.jl`：arXiv:1012.1740v2 十阶能量展开及解析磁化导数，J=1。
- `measure.jl`：CTMRG 测量所有 stabilizers 和 edge magnetizations；精确 blocking 只用于测量。
- `run.jl`：可调 D、field、dt、ALS、χ；CSV/JSON/checkpoint 增量输出。
- `cylinder_audit.jl`：冻结态的 circumference-4、无限长度圆柱 transfer-matrix 收缩；有限周长诊断，非无限平面基态解。
- `contraction_audit.jl`、`sequential_audit.jl`：实际 CTMRG 失败后触发的环境初始化/增长顺序诊断。

`--max-seconds` 是在更新/测量返回后检查的软时间预算；单次 CTMRG 和最终 audit 可能超出它。较长 D=4 任务需要重新估算资源。本次 active cluster 的 SSH 认证失败，未提交远程任务。

# 一般 D 与有限场 loop/cluster update

当前有限场主要实现：保留四臂 star HOSVD、八张量 plaquette ALS 和 diagonal 外环境 λ，加上精确 field gate。支持均匀连接维数 D；场项用半步夹住稳定子 sweep。

三个弱场点为 (0.1,0)、(0,0.1)、(0.1,0.1)。D=2 从随机态分别开始，D=3/4 从较小 D 结果扩大。D=2 X 单轴、D=3 Z 单轴及双分量达到能量约 10⁻⁵/spin、磁化 1% 的目标；D=2 Z 响应约偏差 1.16%，D=3 X 及 D=4 仍有未完成问题。详细数值见 [benchmark_summary.csv](benchmark_summary.csv) 与 [M3](../../M3_REPORT.md)。

```bash
# 从仓库根运行；选择新输出目录
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/loop/run.jl --D=2 --hx=0 --hz=0.1 --every=40 --out=tracks/peps/solutions/anyon-correlator/simple-update/loop/results/new-z
julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/loop/tests.jl
```

`core.jl`、`loop.jl` 进行物理更新，`measure.jl` 做 CTMRG 测量，参考公式读取 `../../model/reference.jl`。`vumps_core.jl` / `vumps_run.jl` 对保存的张量求上下边界，测量无限平面观测量；`cylinder_audit.jl` 另测量周长 W=4 的有限圆柱。

D=3 双分量的 CTMRG/VUMPS 主要观测量一致到约 10⁻¹¹，细节见 [VUMPS_REPORT.md](VUMPS_REPORT.md)。可从已保存的 JSON 直接测量，不必先重算物理态：

```bash
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/loop/vumps_run.jl --state=d3-xz --chi=16 --seed=2 --tol=1e-9 --maxiter=100 --balanced=true --portable=true --budget=300
```

`reproduce.jl` 重算固定批次并更新相应结果，新计算使用 `run.jl --out=<新目录>`。`results/` 保存本实验数据，`frozen_states/` 保存 D=2/3/4 的完整张量 JSON，`figures/` 保存本实验的图。

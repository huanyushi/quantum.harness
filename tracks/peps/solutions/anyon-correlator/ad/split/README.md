# Split V/P/λ 的弱场 AD

**当前结果：D=2、χ=8 优化/χ=16 检查，在 (hₓ,h_z)=(0.1,0)、(0,0.1)、(0.1,0.1) 达到能量 10⁻⁵/spin、非零磁化 1% 的目标。** 起点为解析零场或已有近基态，未显式施加虚拟 Z₂ 约束；随机寻优与非对称扰动后的恢复尚未通过。

优化变量是实数 V、P 和正 λ 的对数。D=2 时，四个位置独立有 160 个原始参数，平移共享有 40 个；当前成功计算采用后者。测量时将张量精确合并成 d=4 的 PEPS，求导仍对原来的 split 参数进行。

`network.jl` 做网络和独立算符；`objective.jl` 做 CTMRG、fixed-point/unrolled 求导及结果检查；`optimize.jl` 做 L-BFGS/Armijo；`run.jl` 提供显式 CLI。Fixed-point 伴随求解失败会拒绝梯度，成功弱场路线用 N=20 次 unrolled AD，末端比较 N=40；所选结果另有 N=40/80 和有限差分记录。

从解析零场开始的完整重算命令如下，从仓库根运行，使用新的输出目录：

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/ad/split/run.jl suite --out=tracks/peps/solutions/anyon-correlator/ad/split/results/new-unrolled --uniform=true --exact=true --gradient=unrolled --unroll=20 --ctm-tol=1e-11 --ctm-maxiter=1000 --chi=8 --chi-check=16 --seed=19 --maxsteps=400 --budget=3600
```

同类四点重算曾用时约 22 分 19 秒，峰值内存约 3.62 GiB。首次 Julia 编译占明显时间；`--budget` 限制优化阶段的时间，编译和后续检查另计。

`summary.csv` 汇总各次计算，`selected_fields.json` 指定三个场点采用的结果，`frozen_states/` 保存对应参数。`results/` 保存逐步数据、运行检查和程序副本，`figures/` 保存本实验的图。

`diagnose.jl` 检查导数，`repair.jl` 从近基态继续优化，`boundary_check.jl` 对同一张量作独立边界测量。边界测量使用 SU 目录的张量转换和收缩函数，物理态仍由 AD 优化。

软件检查：

```bash
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/ad/split/tests.jl
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/ad/split/gradient_tests.jl
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/ad/split/boundary_adapter_test.jl
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/ad/split/failure_control_test.jl
```

参考公式统一读取 `../../model/reference.jl`。详细数值和限制见 [REPORT.md](REPORT.md)，整体下一步见 [PLAN.md](../../PLAN.md)。

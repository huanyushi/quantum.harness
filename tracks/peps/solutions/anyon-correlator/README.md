# Toric Code 基态与任意子转移矩阵谱

本项目先用精确可解的 Toric Code 检查 PEPS 算法，再计算磁场中的基态，随后施加虚拟 Z₂ 对称约束，构造 transfer matrix，按对称性求解不同 sector 的 spectrum，最终研究任意子关联函数与关联长度。

**当前进展：零场 loop/cluster update 已从随机态成功；三个弱场点已有 loop 和 AD 结果。显式虚拟 Z₂ 约束后的完整验证、跨相变路径和任意子转移矩阵不同 sector 的谱仍未完成。** 后续求基态以 loop/cluster update 为主，精度不足或无法收敛时再使用 AD。

## 模型与主要结果

```text
H = −Σₛ Aₛ − Σₚ Bₚ − hₓΣᵢXᵢ − h_zΣᵢZᵢ
Aₛ = ∏star X，Bₚ = ∏plaquette Z，Jₑ = Jₘ = 1
```

X、Z 为 Pauli matrices。物理自旋在无限平方晶格的边上，d=2；数值采用 2×2 顶点重复单元，包含 8 个边自旋。所有能量按边自旋归一化。零场严格值 E/N=−1，所有稳定子 ⟨Aₛ⟩=⟨Bₚ⟩=1。8-spin 周期小系统的 ED 仅用于模型检查，不代替无限平面计算。

| 计算 | 已得到的结果 | 适用范围 |
|---|---|---|
| 零场 loop/cluster update，D=2 | E/N≈−0.999999999913；最大稳定子误差约 4.42×10⁻¹⁰ | 随机 seed=1，χ=8/16 一致 |
| 有限场 loop，(0.1,0)，D=2 | 能量误差约 1.76×10⁻⁶/spin，磁化误差约 0.143% | 达到既定弱场精度 |
| 有限场 loop，(0,0.1) 和 (0.1,0.1)，D=3 | 能量误差约 4.2–5.9×10⁻⁶/spin，Z 磁化误差约 0.77% | 来自 D=2 的增大 D 延续；双分量态另有 VUMPS 测量支持 |
| 有限场 split AD，三个弱场点，D=2 | 能量误差≤10⁻⁵/spin，非零磁化误差≤1% | 解析零场或近基态起点，采用 unrolled AD；随机全局寻优尚未通过 |

有限场 loop 的 D=2 Z 磁化误差约 1.16%，未达到 1% 标准；D=3 X 单轴尚有局部拟合和较大 χ 的问题；D=4 双分量计算未通过。详见 [零场总结](M2_REPORT.md)、[有限场总结](M3_REPORT.md) 和 [主要结果表](RESULTS.csv)。

## SU 与 AD 分别放在哪里

两种算法用文件夹区分，里面也记录了不同实现方式：

| 目录 | 内容 |
|---|---|
| [model/](model/README.md) | Hamiltonian、解析零场张量、ED、低场级数参考 |
| [simple-update/](simple-update/README.md) | SU 类方法总说明：旧 merged L 形 SU、split path/loop 对照、一般 D 的 loop 方法 |
| [ad/](ad/README.md) | AD 总说明：旧 merged tied-tensor 零场优化，以及新 split V/P/λ 的弱场优化 |
| `tests/` | 保留算法的测试、历史张量读取检查及结果汇总脚本测试 |
| [docs/REORGANIZATION.md](docs/REORGANIZATION.md) | 本次移动、删除、备份和检查的说明 |

每套实现的 `results/` 保存本地计算数据，`figures/` 保存该实验的图，`frozen_states/` 保存可随代码携带的最终张量 JSON。项目根目录的 `RESULTS.csv` 只汇总主要结果，注明算法、初态、D/χ、状态和详细数据位置，不用它替代原始计算数据。

## 如何运行

命令均从仓库根目录运行，使用已安装的 `julia-env`，不需要更新软件包。以下新计算请选择未使用的输出目录，避免覆盖已有结果。

```bash
# 零场：loop/cluster update，另带解析零场对照
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/split/run.jl --init=both --plaquette=loop --seed=1 --schedule=0.05:200,0.01:200,0.002:200 --measure-every=40 --out=tracks/peps/solutions/anyon-correlator/simple-update/split/results/new-zero-loop

# 有限场：从随机态求 (hₓ,h_z)=(0.1,0)，D=2
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/loop/run.jl --D=2 --hx=0.1 --hz=0 --every=40 --out=tracks/peps/solutions/anyon-correlator/simple-update/loop/results/new-x

# 全部保留的 Julia 实现测试；不做新的优化或磁场扫描
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/tests/runtests.jl

# 根据已有表格重新生成主要结果表，不执行物理计算
python3 tracks/peps/solutions/anyon-correlator/scripts/summarize_results.py
```

AD 的弱场重算命令见 [ad/split/README.md](ad/split/README.md)。首次 Julia 编译显著慢于后续调用；重算前仍需估计墙时和内存。

整体后续安排见 [PLAN.md](PLAN.md)。本项目源自 challenge #50，但当前计划不再受当时的限时提交流程约束。

# 模型、解析态与参考值

`tc_peps.jl` 保留最初实现的 Hamiltonian、解析零场张量、Pauli 算符及旧 SU circuit，供模型检查、merged AD 和旧 SU 重放使用。新 split SU/AD 的独立算符检查没有改成调用同一构造器，以免实际计算和测试参考产生相同错误。

`reference.jl` 为 SU 与 AD 共用的低场十阶能量及解析磁化导数，来自 arXiv:1012.1740v2 补充材料，已换算到 J=1。它只用于当前弱场比较，不是任意磁场的精确解。

`ed_checks.jl` 用 2×2 torus、8 个物理自旋作有限系统检查。`ed_2x2.csv` 是原计算的完整 12 点表；零场 E₀=−8、简并度 4、gap=4。新生成的 ED 结果保存在本目录 `results/<time>-ed-checks/`，原表不覆盖。

从仓库根运行：

```bash
julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/model/ed_checks.jl
julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/model/tests.jl
```

完整模型约定见 [整体计划](../PLAN.md)。

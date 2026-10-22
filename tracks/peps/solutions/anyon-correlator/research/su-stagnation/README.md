# 旧 SU 停滞原因研究

这项研究分析旧 merged L 形 SU 为什么停在非基态，分为逐门检查和闭式动力学两部分。

- **[single-step/REPORT.md](single-step/REPORT.md)**：保留旧停滞态的张量和连接权重，逐门比较精确物理作用、库的截断及无限平面观测。定位 star 开放 L 形区域的截断损伤；后续 plaquette 改善被抵消。
- **[dynamics/REPORT.md](dynamics/REPORT.md)**：对明确的 D=2 filter 态族推导闭式 SU 更新，预测 dt=0.05 的 e≈−0.753303645771；在限定表示、权重、门序与截断分支内，dt→0 仍趋于 e=−3/4。它没有证明所有 SU 或所有初态都如此。

每部分的 `results/` 保存原始数据和张量，表格与图保存在对应目录。小 torus 波函数用于精确诊断，无限平面值另用 CTMRG 测量。

## 基础测试

从仓库根运行，单独检查这项研究的 38 项代数和更新关系：

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/research/su-stagnation/runtests.jl
```

## 完整重算

以下命令重新运行闭式动力学研究，并更新本目录的计算结果和图：

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/research/su-stagnation/dynamics/reproduce.jl
```

文件移动和备份说明见 [REORGANIZATION.md](../../docs/REORGANIZATION.md)。

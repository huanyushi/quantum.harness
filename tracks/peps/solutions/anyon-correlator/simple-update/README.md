# Simple-update 类算法

七月和十月的 SU 实现统一放在这里，按张量表示和更新规则区分。当前求基态主要采用 **loop/cluster update**：使用 diagonal λ 近似外环境，保留完整 plaquette 环进行局部拟合。它与开放路径截断的 SU 不同，也不是采用完整二维环境的 full update。

| 目录 | 实现 | 结果与用途 |
|---|---|---|
| [one-site-su/](one-site-su/README.md) | 共享一组 V/P/λ，保持1-site平移不变性、V的90°旋转和P的端点交换关系 | 当前单组参数零场结果：D=2、随机seed=5，700轮后e=−1；多起点收敛尚待检查 |
| [merged/](merged/README.md) | 七月一般 merged PEPS，L 形三站点 SU | 随机 seed1 停在 e≈−0.7533；用于重放旧停滞与门检查 |
| [split/](split/README.md) | 十月 D=2 V/P/λ；同一个 star 更新下比较 path-SU 和 loop/cluster | path 在所测轨迹未成功；loop 从同一随机态成功得到零场基态 |
| [loop/](loop/README.md) | 一般 D 的 loop 实现，含磁场门 | 三个弱场点、D=2/3/4 的结果和限制；当前有限场主要程序 |

`split/core.jl` 中是共同网络、star 和 path 更新，`split/loop.jl` 中是闭环 ALS。两种方法共享 state、star 和测量，通过 `split/run.jl --plaquette=path|loop` 选择。有限场一般 D 版位于 `loop/`。

所有计算仍采用 H=−ΣA−ΣB−hₓΣX−h_zΣZ，J=1、Pauli convention、无限方格边自旋。零场结果见 [M2](../M2_REPORT.md)，有限场结果见 [M3](../M3_REPORT.md)。同一 D 下，merged 和 split 允许的张量不同，结果差异不能仅归因于求解器名称。

旧 SU 停滞原因的专题研究单独放在 [research/su-stagnation/](../research/su-stagnation/README.md)。

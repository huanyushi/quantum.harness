# Automatic differentiation（AD）算法

AD 通过能量对张量参数的导数优化基态，与施加虚时门并截断的 simple update 分开。七月和十月的实现都归在本目录，按张量表示区分，不再以月份建立平行项目。

| 目录 | 参数化与方法 | 完成情况 |
|---|---|---|
| [merged/](merged/README.md) | 一个 dense complex D=2 merged 张量共享在四个位置，固定点求导、投影梯度、Armijo descent | 七月零场成功；作为已完成对照 |
| [split/](split/README.md) | 实数 V/P/λ，位置独立或平移共享；fixed-point / unrolled AD | 十月三个弱场点从解析零场或近基态起点成功；随机全局寻优未通过 |

两者的张量表示不同：一般 merged 张量允许的自由度比固定内部 D 的 split 网络更宽，相同 D 下的可变参数也不同。当前求基态仍以 loop/cluster update 为主，AD 作备用和比较。

有限场的具体结果见 [M3](../M3_REPORT.md)。

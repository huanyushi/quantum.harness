# 原 merged L 形 SU

保留七月使用的算法，方便与十月 path/loop 比较及研究停滞机制。它将 V 和 outgoing P 合并为一般 PEPS 张量，star/plaquette 都在三个 merged 张量的 L 形区域上由 PEPSKit 截断。

`run.jl` 重放原随机初始化。Seed1、dt=0.05 的轨迹在约 175 轮后权重稳定，能量为 e≈−0.753303645769，未达到基态目标，程序以非零退出码结束。逐轮结果输出到终端。

```bash
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/merged/run.jl 1 600
```

`gate_isolation.jl`、`plaq_continue.jl`、`seed_probe.jl` 分别用于门隔离、仅 plaquette 延续和早期 graded 初态检查。停滞原因见 [专题研究](../../research/su-stagnation/README.md)，零场方法比较见 [M2](../../M2_REPORT.md)。

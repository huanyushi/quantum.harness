# 共享 merged 张量的零场 AD

七月已完成的 AD 路线：一个 dense complex D=2 张量复制到四个位置，四个位置的梯度取平均，再作归一化 Armijo descent。χ=4 处理主要下降，χ=8 用于接近零场基态时。

随机轨迹的 step86 给出 E_cell=−7.999999995071725、E/N≈−0.999999999384，最大 star/plaquette 误差约 5.95×10⁻¹⁰/6.37×10⁻¹⁰。原记录见 [final_measurements.csv](final_measurements.csv)，完整保留轨迹在 `results/`，收敛图在 `figures/`。

优化在线搜索停止时结束，原日志记为 FAILED；所得态已达到零场能量和稳定子标准，保留作零场方法对照。

```bash
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/ad/merged/tests.jl
```

`ad_tied_core.jl` 处理共享张量和梯度，`ad_tied_gd.jl` 为原优化器，`diagnostics/` 保存导数对比与 retract-path 检查。有限场计算使用 [split AD](../split/README.md)。

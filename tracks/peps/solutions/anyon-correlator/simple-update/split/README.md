# D=2 零场 path-SU 与 loop/cluster update

**结果：两种方法从相同的随机 V/P/λ 出发，path-SU 在 600 轮内未得到基态，loop/cluster update 在 120 轮成功。** Loop 的 χ=16 能量为 −0.999999999913340，最大稳定子误差约 4.42×10⁻¹⁰，χ=8/16 一致。

网络始终保留 rank-4 V、rank-3 P（physical d=2）及每条 V–P 连接的 λ。共同 star 更新使用四臂 HOSVD。Path 沿七张量开放路径做 TT-SVD；loop 保留八张量闭环做 ALS，并用局部 cavity messages 选择 gauge。CTMRG 只测量已有张量，不参与物理更新。

| 文件 | 内容 |
|---|---|
| `core.jl` | state、随机/解析初态、star HOSVD、path 更新、精确 blocking |
| `loop.jl` | 闭环拟合及 λ/gauge 重建 |
| `measure.jl` | 独立算符与 CTMRG 测量 |
| `run.jl` | `--plaquette=path` 或 `--plaquette=loop`；逐步 CSV 与张量保存 |
| `final_audit.jl` | 最终 χ 检查、解析控制、旧 merged SU 重放 |
| `final_measurements.toml` | 原完整终态测量表的原样副本，可随代码携带 |
| `results/`、`figures/` | 本实验的本地逐步数据/二进制张量与图 |

推荐只重算成功方法并选择新输出目录：

```bash
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/split/run.jl --init=both --plaquette=loop --seed=1 --schedule=0.05:200,0.01:200,0.002:200 --measure-every=40 --out=tracks/peps/solutions/anyon-correlator/simple-update/split/results/new-loop
julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/split/tests.jl
```

`reproduce.jl` 是当时完整 path/loop/旧方法对照的重算脚本，会重写命名固定的生成结果，只用于有备份的完整重算。平时请用 `run.jl --out=<新目录>`。`plot.py` 仅根据已有数据画图。

完整零场算法推导见 [REPORT.md](REPORT.md)，有限场与一般 D 结果见 [../loop/](../loop/README.md)。旧 SU 停滞原因另见 [专题研究](../../research/su-stagnation/README.md)。

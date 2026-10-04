# 有限场 loop/cluster update 验证计划

用户已批准：独立目录，按三个低场点推进，允许 D>2。H = −Σ A −Σ B −hₓΣX −h_zΣZ，Jₑ=Jₘ=1，Pauli operators，无限方格、2×2 vertex cell、edge physical dimension d=2。主指标 mₓ、m_z，辅以全部 stabilizers、对偶关系及每 spin 能量。

## 实施顺序

1. `core.jl`、`loop.jl`：将零场实现独立移植为任意均匀 D；保留四臂 HOSVD star、八张量 ALS plaquette、diagonal 外部 λ。添加精确 2×2 field gate，采用 field 半步 → 完整 stabilizer sweep → field 半步。记录 ALS 前后残差、轮数与停止标志，BP 真实固定点 residual，regauging 重构误差。
2. `reference.jl`：arXiv:1012.1740v2 补充 Eq.(1) 的 h_y=0 十阶展开，J=1；逐项解析求导给 magnetization；八阶与十阶差只作阶数敏感性。
3. `tests.jl`：field gate 与独立矩阵 exponential 对比；D=2/3 带非均匀 λ 的 ring gauge 重构；ALS 目标下降；级数原始 convention 与变换后数值、导数交叉核对。
4. `measure.jl`、`run.jl`：精确 blocking 只用于 fresh CTMRG，记录四 stars/four plaquettes、各 edge X/Z。三个点 `(0,0.1),(0.1,0),(0.1,0.1)` 从同一个 seed 的随机态开始。先 D=2，再按误差提升 D；D-continuation 与直接随机初始化分别标注。
5. 对双分量点单独检查 dt、ALS 12→24 轮和冻结态 χ=8→16（更高 D 时相应增大 χ）。出现不一致才追加定位实验。
6. `REPORT.md`、数据与图：按测量误差、update 误差、ansatz 偏差区分结论；附完整复现命令。

## 资源与精度

Ring 目标含 16D⁸ 个实数，D=2/3/4 分别为 4096/104976/1048576 元素；ALS 临时 design matrix 最大约 8D¹⁰ 个实数，D=4 约 67 MB。先实测 1–10 sweeps，估计全任务。预计 D=2（含首次 Julia 编译）数分钟；任何预计超过 10 min/16 GB 的后续计算走 active Slurm profile，避免本地长跑。

第一轮目标：磁化相对参考误差约 1% 内，能量误差约 1e-5/spin 内；这是实用 benchmark 目标，不是精确解保证。守恒 sector 偏差目标 1e-6。最终判定同时报告 dt/ALS/χ 敏感性和级数阶数差，不将阶数差冒称严格误差条，也不将 D=2 失败直接解释为代码 bug。

## 执行记录

- 完成三个 D=2 随机点，以及等追加 τ 的 dt/ALS 对照；完成三个 D=3 bond-growth 点。D=3 双分量另做较小 dt，X 单轴另比较 ALS 12/24。
- 为高 D 实现 exact QR-compressed least squares 和 balanced ring contraction。34/34 代数检查通过；优化核心从头复现三个 D=2 随机轨迹（`qr-recheck`）。
- Active SCNet SSH 认证失败，未提交远程任务。优化后有界 D=3 任务在本地数分钟完成；D=4 pilot 触及时间预算，final audit 未完成，保留 checkpoint 与失败记录。
- 实际 contraction 不一致触发 frozen-state sequential CTMRG、identity-boundary 和 circumference-4 cylinder 检查。后者是有限周长诊断，不是无限平面基态 oracle。
- 结论见 `REPORT.md`：低场近似有效性有证据，D=2/3 方向偏差仍在，D=3 X 方向与 D=4 没有完整通过；未执行 same-ansatz variational polish。

## 用户批准的后续：仅本地 frozen-state boundary VUMPS

- 保持 physical checkpoints 不变，先校准 exact state，再核对 D=3/D=4 双分量态；不使用超算。
- 使用 MPSKit multiline VUMPS 分别求上下边界；对两行 strip 求水平环境，并插入完整四体 stabilizers。
- D=3 原 gauge 失败后，添加精确的 measurement-only λ balancing；实际 checkpoint 的 periodic wavefunction 恒等式与 hash 检查通过。
- 完成 D=3 χ=8/16、seeds1/2/17；完成 D=4 χ=8/16/24 的有界检查。主分支及失败结果全部保留于 `results/vumps/`。
- 结果与原报告的关系见 `VUMPS_REPORT.md`；D=3 测量证据增强，D=4 仍未通过 χ 收敛和 benchmark。

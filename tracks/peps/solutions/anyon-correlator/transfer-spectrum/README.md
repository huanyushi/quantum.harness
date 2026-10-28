# 零场普通行转移算符

局部网络、解析边界、动量谱接口与输出检查已实现；当前仍缺少验收通过的数值基态，尚无生产谱。物理设置与实施要求见 [六项计划](../docs/2026-10-23-zero-field-z2-transfer-spectrum-plan.md)。

把每个顶点北、东的 P 合入 V，定义

$$
A^{p_Ep_N}_{nesw}=\sum_{a,b}V_{basw}P^{p_E}_{ae}P^{p_N}_{bn}\lambda_a\lambda_e\lambda_b\lambda_n.
$$

物理融合为 $p=p_E+2p_N$。每条原始 V–P 连接恰好计入一次 λ。沿物理指标收缩 A 与其共轭得到双层张量 Q，每个双层腿的原始次序是 ket 快、bra 慢。通过保留的 TensorKit 融合映射 $F:\mathcal V\otimes\bar{\mathcal V}\to W$ 转到偶、奇各二维的空间。独立群作用是 $F(Z\otimes I)F^\dagger$ 和 $F(I\otimes\bar Z)F^\dagger$，并未将两种作用合为一个标签。

Q 每次传播一个顶点晶格间距。普通无限 MPO 的局部腿为 `left,down ← up,right`，周期为一位点，局部张量除以其 Frobenius 范数，除去的尺度另存。没有插入 seam 或 twist，因此它是普通 transfer matrix，而非 twisted transfer matrix。边界激发的 charge sector 与畴壁背景选择属于后续分析，不能由普通 TM 的本征值直接命名为 e/m 关联函数。

解析 copy 张量在此 blocking 下，宽度 2 的极小闭合行经主本征值归一化后满足 $T^2=T$，两个本征值为 1，十四个为零（实际测试容差 $10^{-12}$）。这是固定点维数和局部指标的代数检查，不作为无限平面生产谱。生产谱必须读取新优化且通过物理验收的数值态。

本次实际非零值约为 0.25；归一化零谱的最大数值残差为 $1.21\times10^{-16}$，投影关系残差为 $9.43\times10^{-16}$。新的 seed=5 数值态已完成累计 700 轮，但停在 X 极化乘积态附近，未通过基态验收；详见 [进度与资源记录](../docs/2026-10-23-zero-field-z2-progress.md)。

### 解析参考的任意长度代数关系

在未融合的 ket/bra 指标中，解析态的物理求和要求 $n=\bar n$、$e=\bar e$，并满足 $n\oplus e\oplus s\oplus w=0$ 与 $n\oplus e\oplus\bar s\oplus\bar w=0$。按本目录归一化约定，每个非零 Q 分量为 $1/512$，$\|Q\|_F=1/128$。由于每个站点向右输出的双层指标均为对角态，闭合行的全部水平指标必须落在 $|00\rangle,|11\rangle$ 空间，向下输出也随之落在同一空间；向上的非对角输入直接被消去。

在这个二维局部空间中，以 n、s 表示上下的二元指标，以 w、e 表示左右指标，除以 $\|Q\|_F$ 后的 MPO 分量为 $\delta_{n\oplus e\oplus s\oplus w,0}/4$。对任意行长 L，水平求和给出

$$
T_L=2^{-L}\left(|u_+\rangle^{\otimes L}\langle u_+|^{\otimes L}
                      +|u_-\rangle^{\otimes L}\langle u_-|^{\otimes L}\right),
\qquad |u_\pm\rangle=(|00\rangle\pm|11\rangle)/\sqrt2.
$$

这解释了两个固定点及主导本征值密度 $1/2$；L=2 的检查对应两个本征值 $1/4$。两种向量均保持对角 Z₂，单层 Z 将二者交换。对同一背景的正交局部扰动，或者左右分别取两个不同背景的畴壁扰动，上式投影在无限边界上的作用均为零；含局部奇 charge 的输入也被消去。因此解析参考在本计划的单准粒子变分空间中预期为零转移谱，与动量无关。该推导只用于校准接口和零模式判据，不能替代从新数值基态求解的生产谱。

## 边界和类别

边界 MPS 的物理空间是双层腿 W，只约束对角作用 $C=G_kG_b$。求解从最小 χ=1 开始，以非厄米 VUMPS 求一个代表 $r_0$；若残差或代表不同性检查失败，才尝试 χ=2、4。由 $r_1=G_kr_0$ 构造另一代表，重新建立各自环境。用混合 MPS 转移矩阵的主本征值模（每格重叠）比较两个无限态；单纯的张量数组差或整体相位不作为不同性证据。零 Schmidt 支撑只在 MPS 键上截除，局部 W 的偶、奇空间均保留。当前解析边界测试 20/20 通过，两个代表 χ=1，Galerkin 残差约 $1.57\times10^{-16}$。

普通 TM 一般不能仅由实数张量认定为厄米；程序另对物理输入输出交换后的伴随 MPO 求边界，并比较主导本征值密度。伴随边界是原本征问题的左态检查，和畴壁的两个渐近背景是不同对象。Galerkin 残差与每格固定点保真度分别记录，都不直接构成完整无限态误差的严格界。

激发用左正交规范下的局部扰动 B 构造：B 左侧固定为 $r_0$，右侧依 g 取 $r_0$ 或 $r_1$，各位置叠加相位 $e^{ikn}$；q 是未破缺对角 Z₂ 的 charge。因此 g 标记畴壁背景、q 标记对称性 sector，**二者均没有向普通 MPO 插入 twist**。按解析 P 的物理 X/虚拟 Z 恒等式以及开放 Z/X 字符串与稳定子的端点对易关系，类别 $(g,q)=(0,0),(0,1),(1,0),(1,1)$ 对应 $1,e,m,\epsilon$。这里只校准标签，并未计算这些任意子的物理关联函数。

谱接口针对 MPSKit 0.13.12，采用显式单行 Multiline 的 statmech 最大模入口，QP 对象事先携带指定的 sector、实数动量和左右渐近背景。普通 MPO 实数动量包装入口及显式单行入口曾在本机 Julia 1.12 的常规首次类型推断中超过 600 秒，故将兼容调用集中在 `spectrum.jl`，没有升级依赖或更改数值容差。数值校准使用额外的 `--compile=min` 减少首次编译；同一套全目录测试 **97/97 通过**。这一结果不等于常规编译方式的耗时问题已经解决。

残差用同一归一化有效算符重新作用于单位范数 B，计算绝对量 $\|T_{\rm eff}B-rB\|$，不除以趋零的 r。允许参数维数很小时，可在完整的 B 参数基上确认有效作用为零；这个有限矩阵属于无限 MPS 准粒子变分空间，不是圆柱转移矩阵。非零控制例返回 0.6、0.4、0.2，整体 MPO 乘 7 后不变，最大比值误差 $1.66\times10^{-16}$、最大有效残差 $1.77\times10^{-16}$。解析参考四类、两个动量共 8 点均为 zero_mode，最大数值模 $9.26\times10^{-17}$。完整参数、数据和复现脚本见 `results/interface-calibration-20261023/`。它们仅是接口及解析参考检查，没有替代生产数值态。MPSKit 的畴壁相位规范警告保留；这些零参考模式的相位未定义，未据此报告非零畴壁模式的相位物理结论。

输出区分 `zero_mode`、`below_resolution`、`empty_sector`、`not_converged` 和 `budget`。精确参考的零模式对应无穷转移谱量；生产态的小值只能作为低于分辨率记录。JSON 对未定义相位、无穷谱量使用 null。`plot.py` 只读取 CSV 生成 `rho.svg` 与 `epsilon.svg`，不会启动 Julia 或重新优化张量。

从仓库根运行：

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/transfer-spectrum/tests.jl --group=network
```

运行已通过的整套数值接口检查：

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --compile=min --project=julia-env tracks/peps/solutions/anyon-correlator/transfer-spectrum/tests.jl
```

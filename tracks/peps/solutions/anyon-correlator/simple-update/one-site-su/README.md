# 单组 V/P/λ 的 Toric Code 零场计算

对于零场 Toric Code，我们为其施加了 1-site 平移不变性和旋转对称性，全程演化始终只有一组 `V/P/lambda`。

**当前完整算例采用 $D=2$、随机 seed=5，700轮后测得每边自旋能量 $e=-1$，$\chi=8,16$ 的最大稳定子偏差均约为 $4.44\times10^{-16}$。** 本文说明实际采用的局部更新和共享参数恢复方法；数值结果与收敛范围见 [REPORT.md](REPORT.md)。

## 模型与共享张量

自旋 $1/2$ 位于无限平方晶格的边上。采用 Pauli convention，$J_e=J_m=1$，Hamiltonian 为

$$
H=-\sum_s A_s-\sum_p B_p,\qquad
A_s=\prod_{i\in\mathrm{star}(s)}X_i,\qquad
B_p=\prod_{i\in\partial p}Z_i.
$$

$A_s$ 作用于顶点 $s$ 相接的四条边，$B_p$ 作用于方格 $p$ 边界上的四条边。零场基态同时满足所有稳定子的本征值为 $+1$；能量按边自旋归一化后为 $e=-1$。

网络由顶点张量 $V_{nesw}$ 和边张量 $P^p_{ab}$ 组成。$V$ 的四条虚拟腿依次朝北、东、南、西；$P$ 的物理指标 $p$ 标记边自旋，两个虚拟指标 $a,b$ 分别连向这条边两端的顶点张量。每条 $V$—$P$ 连接插入对角矩阵 $\Lambda=\operatorname{diag}(\lambda)$。固定所有物理指标后，对全部虚拟指标求和，便得到对应的波函数振幅。

所有顶点共用一个 $V$，所有横边和竖边共用一个 $P$，所有 $V$—$P$ 连接共用一个 $\lambda$。当前采用实数张量，物理维数 $d=2$、虚拟维数 $D=2$，因此 $V$ 有16个数组分量、$P$ 有8个数组分量、$\lambda$ 有2个权重；对称约束和归一化进一步限制这些分量。以一个顶点为平移重复单元时，可把它北侧和东侧的两条边分配给该单元，因而每个单元含两个物理自旋，两条边使用同一组 $P$ 参数。

记 $R$ 为 $V$ 的90°循环旋转，$S$ 为 $P$ 的端点交换，其作用定义为

$$
(RV)_{nesw}=V_{eswn},\qquad (SP)^p_{ab}=P^p_{ba}.
$$

共享参数始终满足 $RV=V$、$SP=P$。由 $R$ 生成的 $C_4$ 群作用还包含180°和270°旋转；单独交换两条方向腿通常不属于这一循环置换。共享 $V,P,\lambda$ 保证顶点平移不变性，上述指标关系使该表示与晶格90°旋转相容。

初始 $V,P$ 由正态随机数生成，施加这两种张量对称性后归一化，初始 $\lambda=(1/\sqrt2,1/\sqrt2)$。本次参数化未额外施加虚拟 $\mathbb Z_2$ 对称约束。

## 局部虚时更新

一轮依次建立一个代表性 star 的局部更新问题和一个代表性 plaquette 的局部更新问题。每次局部压缩完成后，先恢复为一组满足对称关系的 $V/P/\lambda$，再用这组参数进行下一次更新。整个晶格的变化由这一共享参数映射定义。

star 区域包含中心的一个 $V$ 和周围四个 $P$；plaquette 区域包含四个角上的 $V$ 和四条边上的 $P$，构成 $V,P$ 交替排列的八张量环。区域内部的每条连接计入一次 $\lambda$，外接虚拟腿乘上各自的 $\lambda$，以独立的对角权重近似区域外的环境。局部压缩使用这一加权张量的 Frobenius 范数 $\|\cdot\|_F$。这一环境近似忽略不同外接腿之间的关联；CTMRG 用于后续无限晶格期望值的测量。

因为 Hamiltonian 中的局部项分别为 $-A_s,-B_p$，且 $A_s^2=B_p^2=I$，门可以直接写成

$$
\begin{aligned}
U_s(\tau)&=\exp(+\tau A_s)=\cosh(\tau)I+\sinh(\tau)A_s,\\
U_p(\tau)&=\exp(+\tau B_p)=\cosh(\tau)I+\sinh(\tau)B_p.
\end{aligned}
$$

单个门对局部张量的作用按此式计算，固定 $D$ 的近似发生在随后的压缩和共享参数恢复中。

### star：高阶 SVD 截断

对第 $i$ 条边，记 $p_i$ 为物理指标、$a_i$ 为连向中心 $V$ 的内部虚拟指标、$b_i$ 为外接虚拟指标，并组合 $q_i=(p_i,b_i)$。更新前的加权局部张量为

$$
\Theta_s(q_1,\ldots,q_4)
=\sum_{a_1,\ldots,a_4}V_{a_1a_2a_3a_4}
\prod_{i=1}^4\bigl(\lambda_{a_i}P^{p_i}_{a_ib_i}\lambda_{b_i}\bigr).
$$

对其四个物理指标作用 $U_s$，得到 $Y_s=U_s\Theta_s$。每个 $q_i$ 的维数为 $dD=4$。依次将一个 $q_i$ 与其余三个指标分开，把 $Y_s$ 展开为 $4\times64$ 矩阵 $M_i$，并计算 $M_i=\widetilde U_i\Sigma_iW_i^{\mathsf T}$。每次保留最大的 $D=2$ 个奇异值及对应的两列左奇异向量，记保留后的 $4\times2$ 矩阵为 $U_i$。

这四个 $U_i$ 为四条边分别选择一个二维保留子空间。把 $Y_s$ 投影到这些子空间，得到中心核张量

$$
C=Y_s\times_1U_1^{\mathsf T}\times_2U_2^{\mathsf T}
\times_3U_3^{\mathsf T}\times_4U_4^{\mathsf T},
$$

其中 $\times_i$ 表示沿第 $i$ 个指标与矩阵收缩。将每个 $U_i$ 的复合行指标拆回物理指标和外接虚拟指标，再除去构造 $\Theta_s$ 时乘入的外接 $\lambda$，按该边的指标顺序还原为新 $P_i$。前两个奇异值归一化后给出该内部连接的 $\lambda_i$；从 $C$ 的四条腿分别移出这些 $\lambda_i$，得到中心的新 $V$。逆权重采用截断伪逆，在数值零方向取0。各张量最后归一化，整体尺度相应消去。

这一过程是高阶 SVD 截断：它在四个矩阵展开中分别保留主要子空间。记第 $i$ 次 SVD 的奇异值为 $\sigma_{ik}$，程序记录

$$
\delta_{\mathrm{star}}
=\max_i\frac{\sum_{k>D}\sigma_{ik}^2}{\sum_k\sigma_{ik}^2},
$$

用以衡量单方向截断舍弃的权重。它描述这一局部 SVD 的截断强度，后续参数平均的影响另行讨论。

### plaquette：闭环交替最小二乘压缩

沿方格依次编号八个张量，奇数位置为 $V$，偶数位置为 $P$。把每个张量的两条环内连接作为左右指标，其余指标组合为 $q_i$，得到三指标数组 $g_i[\alpha_{i-1},q_i,\alpha_i]$。这里内部连接权重每条吸收一次，外接腿权重吸收到对应 $V$ 中；$V$ 位置的 $q_i$ 包含两条外接虚拟腿，$P$ 位置的 $q_i$ 就是物理指标。对固定 $q_i$，$g_i(q_i)$ 是 $D\times D$ 矩阵，完整闭环收缩为

$$
\Theta_p(q_1,\ldots,q_8)
=\operatorname{Tr}\bigl[g_1(q_1)g_2(q_2)\cdots g_8(q_8)\bigr].
$$

在四个 $P$ 的物理指标上作用 $U_p$，得到目标 $Y_p$。压缩保持所有环内连接的维数为 $D=2$，使用交替最小二乘（ALS）拟合归一化目标 $\widehat Y_p=Y_p/\|Y_p\|_F$：

$$
\eta_{\mathrm{plaq}}
=\bigl\|\Theta_p(\{g_i\})-\widehat Y_p\bigr\|_F^2.
$$

固定另外七个 $g$，只对一个 $g$ 求解时，闭环收缩对待求张量是线性的，因此可以用 SVD 伪逆求解线性最小二乘问题。依次更新八个位置构成一轮 ALS；本次每次门作用最多迭代12轮，满足程序的误差停止条件时提前结束。$\eta_{\mathrm{plaq}}$ 在拟合结束、恢复共享参数之前记录。保持闭环进行拟合，使压缩目标保留了方格内的环状连接。全部八个张量的联合拟合仍是非线性问题，ALS的最终结果可能依赖初始张量。

拟合后需要从 $g_i$ 中重新分离连接权重。程序沿周期链迭代左、右转移映射，求每条连接上的 $D\times D$ 正半定矩阵 $L_i,R_i$：

$$
\begin{aligned}
L_i&\propto\sum_q g_i(q)^{\mathsf T}L_{i-1}g_i(q),\\
R_i&\propto\sum_q g_{i+1}(q)R_{i+1}g_{i+1}(q)^{\mathsf T}.
\end{aligned}
$$

指标按八个位置周期延拓，每次迭代把矩阵归一化到迹为1。这些矩阵用于选择局部环上的连接表示。对每条连接计算 $\sqrt L\sqrt R=U\Sigma W^{\mathsf T}$，以 $\Sigma$ 的奇异值作为显式权重，并在连接左、右张量的对应腿分别作用 $H_{\mathrm L}=U^{\mathsf T}(\sqrt L)^{-1}$、$H_{\mathrm R}=W^{\mathsf T}(\sqrt R)^{-1}$。在 $L,R$ 满秩时，有

$$
H_{\mathrm L}^{\mathsf T}\Sigma H_{\mathrm R}=I,
$$

因此插入这些未归一化权重及两侧变换后，原来的连接收缩保持不变。小特征值方向使用截断伪逆。随后拆开 $q_i$、移出外接腿权重，并将各张量及各组奇异值分别归一化，得到四个 $V_i$、四个 $P_i$ 及八条内部连接各自的 $\lambda_i$；分别归一化只改变局部张量的整体尺度。已有闭环还原测试通过比较还原前后归一化张量的重叠来检查这一步。

## 从局部结果恢复共享参数

上述分解分别返回区域内各个位置的张量。要用它们定义全晶格共用的 $V/P/\lambda$，需要先选定一致的虚拟指标表示，再对对应数组取平均。这样可以避免把虚拟基底不同造成的分量差异直接混入平均。

首先对本次区域内部连接的 $\lambda_i$ 取平均并归一化，得到共用权重 $\lambda_{\mathrm{new}}$。程序给其分量设置 $10^{-12}$ 的正下限后再次归一化，以便稳定地求逆。$\lambda$ 的改变与虚拟变基一起写入连接两侧的张量。固定其他指标，将同一条 $V$—$P$ 连接两侧的分量记作列向量 $v,w$，则原收缩为 $v^{\mathsf T}\Lambda_{\mathrm{old}}w$。对这条虚拟腿采用实正交矩阵 $Q$，使用

$$
v'=Qv,\qquad
w'=\Lambda_{\mathrm{new}}^{-1}Q\Lambda_{\mathrm{old}}w,\qquad
Q^{\mathsf T}Q=I,
$$

便有

$$
v'^{\mathsf T}\Lambda_{\mathrm{new}}w'
=v^{\mathsf T}Q^{\mathsf T}Q\Lambda_{\mathrm{old}}w
=v^{\mathsf T}\Lambda_{\mathrm{old}}w.
$$

此式同时解释了两侧的变换以及新旧权重的比值。对区域边界的外接腿，保持其基底不变，只在区域内那个张量上逐分量乘入 $\lambda_{\mathrm{old}}/\lambda_{\mathrm{new}}$，使乘上外接权重后的数组不变。

$Q$ 的选择由更新前后的 $P$ 决定。记 $P_p$ 为固定物理指标 $p$ 后的 $D\times D$ 矩阵，先把两侧权重吸收为 $F_p=\Lambda_{\mathrm{left}}P_p\Lambda_{\mathrm{right}}$。正交变换作用为 $F'_p=Q_{\mathrm{left}}F_pQ_{\mathrm{right}}^{\mathsf T}$，选择矩阵时，使包含两个物理切片的完整 $F$ 数组归一化后更接近更新前的对应数组。只有属于区域内部连接的腿参与变基，外接腿对应的 $Q$ 取单位矩阵。两条腿均需调整时，交替求解左右两个正交拟合问题，并比较多个符号、置换初始矩阵的结果。

某些 $P$ 对两个虚拟腿同时反射保持不变。例如两个物理切片均为对角矩阵时，$J=\operatorname{diag}(1,-1)$ 可满足 $JF_pJ=F_p$；因此仅比较 $P$ 仍可能保留两种不同表示，而相邻 $V$ 的数值通常能够区分它们。程序额外检查保持 $P$ 不变的反射候选，在保留区域外接腿权重的 $V$ 上比较归一化张量与更新前对应张量的距离，并允许整体正负号差异。相邻两个 $V$ 的距离平方和较小的表示被保留，变换仍在对应连接的 $V,P$ 两侧共同实施。

完成这些变换后，对本次区域中的 $V_i,P_i$ 分别归一化，并按与更新前对应张量内积的正负选定整体符号。记参与本次更新的顶点数为 $N_V$，则先形成

$$
\overline V=\frac1{N_V}\sum_i s_i\frac{V_i}{\|V_i\|_F},\qquad
\overline P=\frac14\sum_j t_j\frac{P_j}{\|P_j\|_F},
$$

其中 $s_i,t_j=\pm1$，star 的 $N_V=1$，plaquette 的 $N_V=4$。随后在张量分量空间施加群平均投影：

$$
V_{\mathrm{new}}\propto\frac14\bigl(\overline V+R\overline V+R^2\overline V+R^3\overline V\bigr),\qquad
P_{\mathrm{new}}\propto\frac12\bigl(\overline P+S\overline P\bigr).
$$

归一化后的 $V_{\mathrm{new}},P_{\mathrm{new}}$ 和共用 $\lambda_{\mathrm{new}}$ 构成下一组参数。程序将每组由循环旋转联系的 $V$ 分量写入同一个浮点数，以保持数值上的精确相等。

这里包含两类不同操作。成对变基和连接权重重写保持加权局部收缩不变；张量取平均及群平均投影则在共享、对称的参数空间中重新选择张量，会改变所得 PEPS 近似。PEPS 波函数对共享张量是非线性的，因此应结合完整物理量检查这种参数恢复的效果。

## 误差检查与结果解释

局部更新记录 star 的 $\delta_{\mathrm{star}}$、plaquette 的 $\eta_{\mathrm{plaq}}$，以及共享参数恢复中成对变换前后的归一化局部张量差。记 $\widehat\Theta=\Theta/\|\Theta\|_F$，后者定义为

$$
\epsilon_{\mathrm{gauge}}
=\min\left\{
\bigl\|\widehat\Theta_{\mathrm{before}}-\widehat\Theta_{\mathrm{after}}\bigr\|_F,
\bigl\|\widehat\Theta_{\mathrm{before}}+\widehat\Theta_{\mathrm{after}}\bigr\|_F
\right\}.
$$

程序在张量取平均之前检查此量小于 $10^{-8}$，并在每次更新后检查 $\|V-RV\|_F$、$\|P-SP\|_F$。$\delta_{\mathrm{star}}$ 和 $\eta_{\mathrm{plaq}}$ 对应各自的局部压缩问题；$\epsilon_{\mathrm{gauge}}$ 检查随后的连接表示变换。取平均造成的额外变化目前尚未单独量化。

物理结果通过末态的能量和全部稳定子期望值判断。同一末态分别用 $\chi=8,16$ 的 CTMRG 环境测量，比较两个环境下的能量、稳定子偏差及环境收敛误差；另外保留外接虚拟指标，检查 $A_s\Theta_s^{(0)}=\Theta_s^{(0)}$、$B_p\Theta_p^{(0)}=\Theta_p^{(0)}$ 的局部张量关系。这里上标 $(0)$ 表示外接腿不乘环境权重，内部连接权重仍保留；其残差定义和数值见 [REPORT.md](REPORT.md#能量稳定子与局部张量检查)。

本方法的局部环境近似、固定 $D$ 压缩和共享参数恢复共同定义了实际迭代。其收敛性需要由数值结果判断；当前完整基态检查来自 seed=5，多起点同预算比较尚未完成。

## 复现与程序对应

从仓库根目录运行，选择尚不存在的输出目录：

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/one-site-su/strict_run.jl --seed=5 --steps=700 --dt=0.05 --budget=480 --out=tracks/peps/solutions/anyon-correlator/simple-update/one-site-su/results/new-zero-one-site
```

当前设置为本地单线程，完整算例含首次编译和测量约需3分钟。每轮保存一组参数，每100轮测量一次 $\chi=8$ 的期望值，最终分别测量 $\chi=8,16$。`initial.json`、`latest.json`、`final.json` 保存初态、最近状态和末态；末态同时保存为 Julia 二进制 `.jls`。逐轮诊断在 `trajectory.jsonl`，最终数值在 `summary.json`，参数和当次源程序的校验值在 `settings.json`。

算法与代码的对应关系为：star 高阶 SVD 位于 [`../split/core.jl`](../split/core.jl) 的 `star_update!`；闭环 ALS 和连接权重还原位于 [`../split/loop.jl`](../split/loop.jl) 的 `fit_ring!`、`ring_gauge`；共享参数恢复位于 [`strict.jl`](strict.jl) 的 `restore_shared!`、`commit!`。全局状态由 `State(V, P, lambda)` 保存。

局部函数用带方向和行列编号的临时数组记录各条连接。代码中的 `L=2` 指这组位置编号的周期大小；`L=1` 会把相邻位置映射为相同编号，使环上的不同连接无法据此区分。star 的中心与四条边、plaquette 的四个角与四条边都从当前共享参数复制到这些位置；本次结果恢复为一组参数后，临时数组丢弃。测量时也复制到 $2\times2$ 重复单元，通过 CTMRG 收缩无限晶格环境。

专项测试命令如下：

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/one-site-su/strict_tests.jl
```

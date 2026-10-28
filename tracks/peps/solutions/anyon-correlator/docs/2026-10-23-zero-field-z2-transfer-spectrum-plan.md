# 严格零场 Toric Code：虚拟 Z₂ 对称基态与无限平面转移谱实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` or `subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 执行方式由执行 session 与用户确定；这份文档本身不授权启动计算、升级环境或提交 Git 改动。

**Goal:** 在保持单组 V/P/λ、1-site 平移和 C₄ 旋转约束的 loop/cluster simple update 中引入 TensorKit 原生虚拟 Z₂ 分块，重新求数值零场基态，再构造无限行 double-layer 转移算符，得到按动量和四类拓扑 sector 分类的转移谱。

**Architecture:** 基态更新与转移谱分析分为两个有明确输入输出的模块；前者输出可移植的数值 V/P/λ，后者只读取张量，不继续优化物理态。使用 TensorKit 表达对称张量，使用 MPSKit 的 `leading_boundary` 和 `excitations` 求边界与边界激发；优先采用显式 double-layer TensorMap 组成的普通无限 MPO。

**Tech Stack:** 仓库 `julia-env/` 中的 TensorKit、PEPSKit、MPSKit、KrylovKit、JSON；绘图优先生成无需新依赖的 SVG。编写本文时本地版本为 TensorKit 0.16.5、PEPSKit 0.8.0、MPSKit 0.13.12。

**Spec:** 本文[科学设置与已确认设计](#scientific-design)是随计划交付的完整设计依据，来自用户已确认的讨论；不要求新 session 读取旧聊天。原有[整体研究计划](../PLAN.md)提供项目背景，其有限场扫描与其他后续工作不属于本轮范围。

日期：2026-10-23。本文是待执行计划，所有任务尚未执行。用户特别确认：原先独立的“定义对称张量”任务已合并入任务 1，不再单列状态管理或通用基础框架任务。

## Global Constraints

- 先读仓库根 `AGENTS.md` 和子项目 `tracks/peps/solutions/anyon-correlator/AGENTS.md`；其中的资源、历史数据和科研写作要求继续适用。
- 本轮严格取 $h_x=h_z=0$、$J_e=J_m=1$，采用 Pauli convention；无限方晶格，物理自旋位于边上，$d=2$、$D=2$，能量按边自旋报告。
- 始终只有一组共享的 V/P/λ；保留 V 的 90°循环旋转、P 的端点交换以及横竖边共用 P 的约束。
- 使用原生 TensorKit 对称空间完成更新中的关键收缩、分解和规范变换；不采用“完整 dense 更新后再投影末态”的替代路线。
- “dense”仅描述完整数组存储，不表示没有平移或旋转对称性；不得以此为由移除已有空间约束。
- 更新算法继续采用 star 高阶 SVD 和 plaquette 闭环 ALS，区域外环境由 diagonal λ 近似；不自动替换成 AD、path-SU 或 full update。
- 解析态仅用于代数与接口检查；正式谱的输入必须是新计算并通过验收的数值基态。
- 不新增有限场、滤波变形、多起点统计、较大 D 扫描、任意子物理能量或关联函数计算；不以有限圆柱谱替代无限平面结果。
- 保留现有 `one-site-su/`、历史结果、可移植末态和环境；不恢复已删除的七月驱动，不启动 SU 停滞专题。
- 不执行包升级或 `Pkg.resolve`，不改写 Julia 缓存源码，不自动连接、修复或提交旧集群；资源不足时停止并向用户说明。
- 软件测试与正式物理计算分开。任何正式计算前，重新说明设置、时间和内存预算并取得确认；本地单次长任务上限不超过 10 分钟，预计常驻内存小于 16 GB，持续输出并保存进度。
- 本轮写计划时不写数值实现、不运行 Julia 计算、不提交或推送 Git 改动。未来执行中只有实际通过的步骤才能勾选。

## Review Focus

1. **charge 与指标顺序：** TensorKit 重排、对偶和融合后，虚拟群作用仍对应正确的 ket/bra 与几何方向；由任务 1、3 的独立数组对照检查覆盖。
2. **简并下的规范变换：** SVD 简并、符号和排列不能造成偶奇空间混合；由任务 1 的分块规范与共享恢复检查覆盖。
3. **秩亏与零支撑：** λ 或 Schmidt 分量为零时使用支撑空间上的伪逆，不产生 NaN，也不删除激发所需的局部 charge 空间；由任务 1、4 覆盖。
4. **固定点代表是否真正不同：** 两个 MPS 的数组不同不等于物理边界不同，不能把同一态的不同规范当作畴壁背景；由任务 4 的规范不变重叠检查覆盖。
5. **零谱与失败状态：** 零本征值、低于分辨率、空变分空间和未收敛分别记录，不能都变成一条有限谱线；由任务 5、6 覆盖。

---

<a id="scientific-design"></a>
## 1. 科学设置与已确认设计

### 1.1 Hamiltonian、几何与已有基础

研究 Hamiltonian 为

$$
H=-\sum_s A_s-\sum_p B_p,\qquad
A_s=\prod_{i\in\operatorname{star}(s)}X_i,\qquad
B_p=\prod_{i\in\partial p}Z_i.
$$

$X,Z$ 是 Pauli matrices，没有额外的 $1/2$。零场基态的所有稳定子为 $+1$；每个顶点重复单元分配北、东两个边自旋，因此每边自旋能量为 $e_0=-1$，每 composite site 为 $-2$。

现有 `simple-update/one-site-su/` 已采用一组实数 V/P/λ，并保持空间约束。`REPORT.md` 记录了 seed=5 的一个成功零场算例，完整进程约 173 秒、峰值内存约 2.19 GiB。这些是旧实现的历史测量，不是新 TensorKit 实现的成本或随机收敛保证。读取该报告以了解起点，不重复开展历史研究。

V 的指标顺序为 $V_{nesw}$；P 的数组约定为 $P^p_{ab}$，$a,b$ 对应边两端的虚拟指标。空间约束是

$$
V_{nesw}=V_{eswn},\qquad P^p_{ab}=P^p_{ba}.
$$

第一个关系是循环旋转，不可替换成任意指标交换。局部门可以使用带位置编号的临时区域张量；每次子更新提交后，全局状态必须仍只有一组 V/P/λ。测量用的 2×2 排布只是同一组张量的复制。

### 1.2 虚拟 Z₂ 的具体含义

每条虚拟腿取 $\mathcal V=\mathcal V_0\oplus\mathcal V_1$，两块各一维，群生成元为 $Z=\operatorname{diag}(1,-1)$。物理维数为 2 的空间整体取平凡表示；这里的虚拟 charge 不等于物理自旋的 $Z$ 本征值。

允许的张量满足

$$
V_{nesw}=0\quad\text{if }n\oplus e\oplus s\oplus w=1,\qquad
P^p_{ab}=0\quad\text{if }a\oplus b=1,\qquad [\Lambda,Z]=0.
$$

P 的允许块自由优化，不固定为解析 copy tensor。初态从允许块中随机生成，再施加已有空间投影和归一化。已有无显式约束的数值态不直接投影到固定对角 Z 的表示；除非先找到其虚拟表示和相应规范，否则这种投影会改变物理态。

TensorKit 的张量分区、对偶和几何指标必须有唯一约定。实现时可将 V 表示为 `N⊗E ← S⊗W`，P 表示为 `physical⊗head ← tail`，λ 表示为虚拟空间上的对角映射；连接和旋转采用相应对偶/重分区操作。对照数组仍统一写成 V[n,e,s,w]、P[p,a,b]，不能直接把 TensorMap 的底层块内存 reshape 成该顺序。

### 1.3 更新算法与近似

保留现有两类虚时门 $U_s=\cosh(\tau)I+\sinh(\tau)A_s$、$U_p=\cosh(\tau)I+\sinh(\tau)B_p$。Star 更新对四个物理指标及相应外接虚拟指标组成的四臂张量做高阶 SVD；每个保留空间固定为一偶、一奇。Plaquette 更新拟合由四个 V、四个 P 组成的闭环，逐张量求解允许块中的线性最小二乘问题。

区域外由独立 diagonal λ 近似。规范对齐、环规范化和连接权重重写均需在 charge 块内完成。固定 Z 后，允许的虚拟变基与 Z 对易。成对变基保持加权局部收缩不变；随后的位置平均和空间投影会改变近似态，应分别记录其影响。

### 1.4 Double layer 与无限行算符

为与现有 `split/core.jl::blocked_tensor` 一致，先采用把每个 V 的北、东两个 P 合入同一 composite tensor 的约定：

$$
A^{p_Ep_N}_{nesw}=\sum_{a,b}V_{basw}P^{p_E}_{ae}P^{p_N}_{bn}
\lambda_a\lambda_e\lambda_b\lambda_n.
$$

这里同一条物理边连接两端 V 时有两条 V–P 连接，两个 λ 均需计入。所有边按其西端或南端顶点分配到北、东 P，因此整个网络每条 V–P 连接恰好计入一次。采用其他边界规范必须显式给出两侧补偿及收缩不变性检查，不静默改变约定。

物理融合顺序是 $p=p_E+2p_N$，$p_E$ 为快指标。按物理腿收缩

$$
Q_{n\bar n,e\bar e,s\bar s,w\bar w}
=\sum_{p_E,p_N} A^{p_Ep_N}_{nesw}\overline{A^{p_Ep_N}_{\bar n\bar e\bar s\bar w}}.
$$

每对虚拟腿融合为维数 4 的空间 $W=\mathcal V\otimes\overline{\mathcal V}$；保存融合映射，以便把 $Z\otimes I$、$I\otimes\bar Z$ 正确搬到融合空间。普通 MPO 的腿顺序为 `[left,down;up,right]`。Q 沿水平方向无限重复，向下一行传播；每次转移跨越一个顶点晶格间距，保留一位点周期。Composite tensor 的自旋分配有方向性，C₄ 仍在原始 V/P 网络上定义。

### 1.5 边界固定点与四种激发类别

转移算符保持 $G_k=\bigotimes(Z\otimes I)$ 和 $G_b=\bigotimes(I\otimes\bar Z)$。Toric-code 固定点的两个代表 $r_0,r_1=G_kr_0$ 保持对角子群 $C=G_kG_b$，单层作用则将不同代表联系起来。边界 MPS 只强制保持对角 Z₂，以允许这种固定点结构出现。$W$ 的对角 charge 为 ket、bra charge 之和模 2，偶奇块维数各为 2。

用边界张量 $a_0,a_1$ 构造动量叠加的局部扰动：激发张量 B 左侧用 $a_0$，右侧按 $g=0,1$ 选择 $a_0$ 或 $a_1$，并对插入位置 n 乘 $\exp(ikn)$ 后求和。g 表示无畴壁/有畴壁；q 表示未破缺对角 Z₂ 的 charge，作为 `Z2Irrep(q)` 传给 `sector`。

| g | q | 背景 | 物理标签（经字符串约定校准） |
|---|---|---|---|
| 0 | 0 | 相同固定点 | 1 |
| 0 | 1 | 相同固定点 | e |
| 1 | 0 | 不同固定点 | m |
| 1 | 1 | 不同固定点 | ε |

开放物理 Z 字符串沿原晶格路径在端点产生 star 违例，标记 e；开放物理 X 字符串沿对偶路径产生 plaquette 违例，标记 m。当前解析 X-basis copy 张量满足物理 X 等价于单条虚拟腿上的 Z 作用，可用于固定 flux 标签。只做端点与张量代数检查，不额外求物理激发态。

`leading_boundary` 求一个边界代表，不自动枚举整个简并空间。畴壁左右的两个渐近背景也不同于转移算符本身的左、右本征向量。固定点空间预期为二维；四种任意子类别不意味着四个这样的固定点。

### 1.6 零场谱的解释

统一用 r 表示相对主导转移本征值的归一化本征值，记录 $|r|$、其有定义时的相位，以及 $\epsilon=-\log|r|$。k 是沿无限边界链的动量，不能从 r 的相位代替得到。$\epsilon$ 是转移谱量，不是原 Hamiltonian 的任意子能量。

先在同一 blocking 和规范下推导解析参考的固定点与零谱结构。精确 toric-code 固定点的非主导谱退化或为零，零本征值对应 $\epsilon=\infty$。数值张量仍有有限误差，小而非零的 r 不能直接解释成零场物理色散。报告只针对所选准粒子变分空间中的若干主导本征值，不宣称完整谱或多粒子连续谱。

## 2. 源码阅读地图与包接口限制

以下路径相对仓库根；先阅读与当前任务有关的部分，不重跑历史程序：

- `tracks/peps/solutions/anyon-correlator/simple-update/one-site-su/{README.md,REPORT.md,strict.jl,strict_run.jl,strict_tests.jl}`：现有共享参数与空间对称流程，尤其检查 `restore_shared!` 和完整 sweep 的失败回滚。
- `tracks/peps/solutions/anyon-correlator/simple-update/split/{core.jl,loop.jl,measure.jl}`：star、闭环拟合、λ 计数、开放外腿稳定子残差和独立物理算符；dense 内核可作小型测试对照，不能直接充当新方案的无约束更新器。
- `tracks/peps/solutions/anyon-correlator/model/tc_peps.jl`：解析张量、物理基底与 Pauli 约定。
- `tracks/peps/solutions/anyon-correlator/simple-update/loop/{vumps_core.jl,vumps_tests.jl,VUMPS_REPORT.md}`：显式 double layer、普通 row MPO、正反向边界与伴随检查；其两行排布不照搬为本轮的物理周期。
- `tracks/peps/solutions/anyon-correlator/tests/runtests.jl`：通过独立模块隔离测试命名空间。
- `skills/using-pepskit/references/pepskit-api.md`、`skills/using-mpskit/references/mpskit-api.md`：包接口地图；精确调用以当前安装版本源码为准。

本计划编写时静态核对到以下事实，但尚未做本项目端到端调用验证：

1. PEPSKit 的 `InfiniteTransferPEPS` 局部对象保留 ket/bra tuple，已有 boundary-MPS 收缩支持；未发现完整的对应准粒子收缩适配。因此本轮使用显式融合后的普通 TensorMap MPO。
2. MPSKit 0.13.12 的 `src/algorithms/excitation/quasiparticleexcitation.jl` 包含 `MultilineMPO` 的 statmech 路径，选择最大模 `:LM`；支持实数动量、`sector` 和不同的渐近背景。
3. 同文件的带实数动量普通 MPO 入口会包装成 Multiline；直接接收 `InfiniteMPO` 与 `InfiniteQP` 的底层重载在构造有效算符时存在可疑自引用。优先走公开的实数动量入口或显式单行 Multiline，并用非零谱对照测试确认；不要为绕过它升级依赖或修改包缓存。
4. Statmech 有效作用包含相对于背景的归一化；返回变量即使名为 energies 也不能按 Hamiltonian gap 解读。必须用已知本征值比值校准。
5. 默认特征求解器可能采用厄米设置；边界和激发调用显式选择适当的非厄米算法，不根据实数张量就假定转移算符厄米。
6. `excitations` 的此版本接口返回本征值和状态，不能假定它另外返回完整收敛记录。自己通过同一有效作用重新计算投影残差；兼容性调用集中在 `spectrum.jl`，附上版本及来源说明。

参考：[Shadows of Anyons](https://arxiv.org/html/1410.5443v2)，主要为 §II.3、§II.5 和 Fig. 7；[PEPSKit boundary-MPS 示例](https://quantumkithub.github.io/PEPSKit.jl/stable/examples/boundary_mps/)；[MPSKit 0.13.12 激发源码](https://github.com/QuantumKitHub/MPSKit.jl/blob/v0.13.12/src/algorithms/excitation/quasiparticleexcitation.jl)。Fig. 7 的非零滤波变形不属于本轮。

## 3. 文件职责与模块接口

下文的短路径均相对 `tracks/peps/solutions/anyon-correlator/`。这些是未来新增路径，当前只创建本计划文件。

- `simple-update/one-site-z2-su/core.jl`：模块 `OneSiteZ2SU` 的入口，组织下列更新文件。
- `simple-update/one-site-z2-su/tensors.jl`：仅一组 V/P/λ 的 `Z2State`、空间、初始化、空间投影、诊断与有约定的数组对照。
- `simple-update/one-site-z2-su/{star.jl,plaquette.jl,shared.jl}`：分别负责 star 分解、闭环拟合/环规范化、共享恢复与原子 sweep。
- `simple-update/one-site-z2-su/{state_io.jl,measure.jl,run.jl}`：可移植状态读写、接入现有物理测量、带预算计算入口；不另建通用状态管理框架。
- `simple-update/one-site-z2-su/{tests.jl,README.md,.gitignore}`：测试入口、算法说明、本地结果忽略规则；`results/` 存原始数据，验收后 `frozen_states/` 存可移植末态。
- `transfer-spectrum/core.jl`：模块 `ToricTransferSpectrum`，显式引用前一模块的状态接口。
- `transfer-spectrum/{network.jl,boundary.jl,sectors.jl,spectrum.jl}`：合并与 double layer、边界固定点、sector 与背景、准粒子求解和残差。
- `transfer-spectrum/{run.jl,plot.py,tests.jl,README.md,.gitignore}`：数据生成、只读数据绘图、测试与说明；作图绝不触发优化或谱重算。
- `tests/runtests.jl`：最后接入新增轻量测试。既有源文件与历史状态不覆盖。

接口约定如下；内部具体 TensorMap 参数类型由实现者根据安装版本选用，但这里的函数名、字段含义和几何约定保持一致：

| 接口 | 消费与产出 |
|---|---|
| `random_state(seed::Int)::Z2State`、`exact_reference_state()::Z2State` | 前者是生产随机初始化，后者仅用于检查。状态的持久物理参数字段仅为 `V,P,lambda`，lambda 是原生对角映射。 |
| `symmetry_residuals(st)` | 返回 `virtual_V,virtual_P,rotation_V,exchange_P,weight_commutator` 五个归一化残差。 |
| `dense_arrays(st)` | 返回遵守本文指标顺序的 V/P/λ 数组，仅用于导出、测量适配及小型测试对照。 |
| `star_step!(st,dt)`、`plaquette_step!(st,dt;als_sweeps)`、`sweep!(st,dt;als_sweeps)` | 原地更新；返回 `fit_residual,discarded_weight,gauge_residual,sharing_change,symmetry`，不适用的诊断填 `nothing`。完整 sweep 失败时不改变传入状态。 |
| `local_residuals(st)` | 返回未乘外接环境权重、保留内部 λ 的开放局部稳定子残差 `star,plaquette`。 |
| `save_state(path,st;metadata)`、`load_state(path)` | JSON 保存允许块/charge、几何腿顺序、对偶与归一化约定；读取返回 `(state,metadata)`，不隐式修复状态。 |
| `measure_state(st;chi,tol,maxiter,envseed)` | 返回每边自旋能量、各稳定子、CTMRG 是否收敛及实际可获得的残差；未提供的迭代次数记 `nothing`。 |
| `build_transfer(st;direction=:down)::TransferNetwork` | 保存 composite A、Q、普通无限 MPO、融合映射、融合后的单点 ket/bra 作用、步长 1、归一化因子与输入状态哈希。 |
| `solve_boundaries(net;options)::BoundaryPair` | 保存 `r0,r1,env0,env1`、必要的伴随边界和归一化、残差及重叠诊断；边界名称不复用为物理任意子标签。 |
| `SectorLabel(g::Int,q::Int)`、`sector_backgrounds(pair,label)` | g、q 均只接受 0/1；q=0 对应 +，q=1 对应 −。无畴壁返回同一背景对象，有畴壁返回真正不同的代表及其环境。 |
| `solve_sector(net,pair,label;momenta,num,options)` | 返回按动量、分支组织的 `SectorPoint`，保留归一化比值 r、有效残差、变分维数和状态；不写作物理激发能量。 |

`SectorPoint` 的对外数据至少包括 `g,q,k,branch,ratio_real,ratio_imag,rho,phase,quasi_energy,projected_residual,variational_dim,status`。状态集合固定为 `converged,zero_mode,below_resolution,empty_sector,not_converged,budget`。`zero_mode` 用于有解析或有效算符零作用依据的模式；数值生产态中的不可分辨小值记 `below_resolution`，不据此证明物理本征值精确为零。

## 4. 实施与检查的共同约定

本计划给出测试名、断言和命令，不预先写出数值算法函数体。每项按“添加测试 → 确认预期失败 → 最小实现 → 通过测试 → 记录实际结果”推进。失败应来自待实现的功能，而非依赖缺失或任意异常。每项形成可独立审查的改动；只有用户授权提交时才创建相应的本地 commit，不自动 push。

所有命令从仓库根执行，Julia 使用 `--startup-file=no -O1 --project=julia-env`，BLAS 和 Julia 均单线程。若在新 worktree 执行，先确认本地被忽略的 Manifest、已有环境和参考数据的可用性，不通过自动重解析环境来填补缺失。首次导入/编译时间与实际迭代时间分别记录。

测试入口接受 `--group=<name>`，省略时运行该目录全部轻量检查；未知 group 报错。新增代码之后才运行下面的未来命令，不在当前写计划 session 中执行它们。

数值初始设置属于实施建议，不是已取得结果：空间与虚拟相对残差阈值 $10^{-12}$；纯规范变换的加权张量相对残差阈值 $10^{-8}$。基态物理验收沿用 $|e+1|\le10^{-6}$、所有稳定子偏差不超过 $10^{-6}$，并要求 `local_residuals` 的两项不超过 $10^{-6}$。CTMRG 沿用 χ=8/16、目标 tol=$10^{-10}$、最大 400 次及最终残差不超过 $10^{-9}$；两个 χ 的能量差不超过 $10^{-6}$。这是继承既有验收，不额外发起收敛扫描。

边界、有效谱的初始求解 tol 取 $10^{-10}$。解析代数测试通常使用 $10^{-12}$ 的相对/绝对容差。零值判别和绘图分辨率要同时保存为参数，不能仅凭一个统一常数给出谱的严格误差界。

## 5. 六项实施任务

### 任务 1：实现保持已有空间约束的原生 Z₂ loop/cluster simple update

**Files:** 新建 `simple-update/one-site-z2-su/{core.jl,tensors.jl,star.jl,plaquette.jl,shared.jl,tests.jl,README.md}`。参考原 `one-site-su/strict.jl`、`split/core.jl` 和 `split/loop.jl`，不改写旧实现。

**Interfaces:** 产出第 3 节的 `Z2State`、初始化、诊断、数组对照、三个 step 接口及 `local_residuals`；后续任务直接使用这些名称。

- [ ] **Step 1 — 添加表示与约束测试。** `test_native_spaces_and_initialization` 检查虚拟 0/1 各一维、物理两个状态均为 charge 0、状态只持有一组 V/P/λ、初态不是解析 copy、五项对称残差不超过 $10^{-12}$。`test_array_conventions` 用逐分量独立收缩核对几何腿和物理基底，误差不超过 $10^{-12}$。
- [ ] **Step 2 — 确认上述测试因新模块/接口尚未实现而失败。** 只运行 `--group=tensors`，保存实际失败原因。
- [ ] **Step 3 — 实现 `tensors.jl`。** 构造原生 graded 空间、允许块随机初始化、空间对称投影及对照转换；归一化仅去掉整体尺度。定义 TensorMap 的腿分区，不创建额外的通用状态框架。
- [ ] **Step 4 — 添加更新测试并确认失败。** `test_charge_preserving_star`、`test_constrained_ring_fit` 检查固定一偶一奇、闭环目标与独立收缩一致、每个子更新后空间/虚拟约束成立。`test_degenerate_gauge_alignment` 在分块内施加符号规范后要求加权局部张量不变；混合偶奇的变换应被拒绝。`test_zero_support_pseudoinverse` 要求秩亏输入不产生 NaN。`test_atomic_sweep` 要求 plaquette 失败后状态逐块与更新前一致。
- [ ] **Step 5 — 实现 star、plaquette 与 shared 文件。** 在 TensorKit 分块中完成 HOSVD、闭环 ALS、环规范化和共享恢复；保留区域内部 λ、外接 λ 的原有计数。分别记录规范变换残差与共享/空间投影造成的张量变化，不把两者合成一个“规范误差”。
- [ ] **Step 6 — 运行两组测试。** 下列命令均须退出 0，所有目标测试通过；一次完整轻量 sweep 可以从随机对称态开始执行。完成后报告任务 1 的检查结果，默认先停在这里，正式长计算留到任务 2 确认设置后。

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/one-site-z2-su/tests.jl --group=tensors
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/one-site-z2-su/tests.jl --group=updates
```

**交付：**真正以原生对称张量运行的局部更新，不是只完成数据结构定义，也不要求本任务跑完完整基态优化。

### 任务 2：求解并验收受约束的数值零场基态

**Files:** 新建 `simple-update/one-site-z2-su/{state_io.jl,measure.jl,run.jl,.gitignore}`，扩充同目录 `tests.jl` 和 `README.md`；运行后新增该目录自己的 `results/<new-run>/` 和验收后的 `frozen_states/`。

**Interfaces:** 消费任务 1 的状态、step 和残差接口；产出状态读写、测量接口，以及 `final.json`、`summary.json` 和明确的成功/失败标记。

- [ ] **Step 1 — 添加运行器检查并确认失败。** `test_portable_state_roundtrip` 检查保存、读取后允许块及复合收缩一致；`test_invalid_state_rejected` 检查 charge/维数/格式错误不被自动修复。`test_output_not_overwritten` 检查已有路径不覆盖。`test_failed_audit_stays_failed` 覆盖超时、环境未收敛、稳定子失败和非有限值，即使能量接近 −1 也必须失败。
- [ ] **Step 2 — 实现状态与测量适配。** 复用现有物理算符和测量收缩；可通过约定明确的数组视图只读地接入 `split/measure.jl`，不得在该视图中执行生产更新。JSON 保存 charge、腿顺序、初态来源、seed、步数、容差、约束、源文件和输入哈希；二进制只能作为附加缓存。
- [ ] **Step 3 — 实现有预算的 runner。** 接受 `--seed`、`--steps`、`--dt`、`--als-sweeps`、`--budget`、`--out`，默认建议 seed=5、700 轮、dt=0.05、ALS 最多 12 轮，但不能把旧 seed 的成功视为新算法保证。每轮保存最新状态；约每 40 轮输出局部残差、拟合和对称性，并 flush；沿用每 100 轮测量和最终 χ=8/16 验收。总预算不超过 600 秒，首次编译单列。
- [ ] **Step 4 — 先通过轻量 runner 测试。** 运行下方 `--group=runner`；该测试不得启动 700 轮优化。
- [ ] **Step 5 — 提交正式运行前的设置摘要。** 明确 H、无限平面、D=2、三项空间关系、虚拟 Z₂、随机对称初态、每边自旋能量和预算；取得用户确认。先用很短的更新计时估计新实现成本，再决定是否能在本地预算内完成。超预算时停下讨论资源，不自动使用旧集群。
- [ ] **Step 6 — 在全新输出目录执行并验收。** 保存 `initial.json`、`latest.json`、`final.json`、`trajectory.jsonl`、`settings.json` 和 `summary.json`。成功须满足第 4 节物理与环境标准，记录实际局部残差。没有通过时保留失败原因和张量，不自动替换成解析初态或扩大成多 seed 扫描。

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/one-site-z2-su/tests.jl --group=runner
```

未来 runner 的示例调用（新目录、正式运行前仍需确认预算）：

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/one-site-z2-su/run.jl --seed=5 --steps=700 --dt=0.05 --als-sweeps=12 --budget=600 --out=tracks/peps/solutions/anyon-correlator/simple-update/one-site-z2-su/results/zero-z2-seed5
```

**交付：**通过验收的数值末态及证据。任务 3 以后正式物理结果必须读取该末态；解析测试可以独立使用解析参考。

### 任务 3：构造并检查无限行 double-layer 转移算符

**Files:** 新建 `transfer-spectrum/{core.jl,network.jl,tests.jl,README.md}`。

**Interfaces:** 消费 `Z2State` 及读写约定；提供 `TransferNetwork` 和 `build_transfer(st;direction=:down)`，字段与含义按第 3 节。

- [ ] **Step 1 — 添加网络测试并确认失败。** `test_composite_lambda_counting` 使用同一对称 V/P 和正的非均匀权重 (0.8,0.6)，要求本文 composite 公式与未合并小块收缩一致。`test_double_layer_direct_sum` 独立对物理指标求和，逐分量核对 Q。`test_fused_symmetry_actions` 检查融合后的 ket/bra 作用、相互对易、平方为 I，以及两种独立单层对称的局部 pulling-through 恒等式。
- [ ] **Step 2 — 实现 A、Q 和普通无限 MPO。** 用同一个状态构造一位点周期，不沿用旧两行物理步长。记录整体归一化；不得把两个原本不同的 ket/bra 指标错误地合成一个失去群作用信息的数组。为微型闭合行测试显式构造全局 Gₖ、Gᵦ，检查它们与转移矩阵对易。
- [ ] **Step 3 — 添加并实现伴随和解析参考检查。** `test_transfer_adjoint` 要求小型收缩满足内积伴随恒等式。`test_exact_reference_fixedpoint_structure` 在相同 blocking 下推导解析参考；可用宽度 2 或 3 的极小闭合行检验其归一化非零谱与零谱、固定点维数。该有限收缩只是代数单元测试，不进入生产谱或替代无限平面。
- [ ] **Step 4 — 运行网络测试。** 所有差值符合 $10^{-12}$ 量级检查，步长等于 1，局部 double-layer 空间维数为 4。将实际推导与约定写入新目录 README，不只记录函数名。

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/transfer-spectrum/tests.jl --group=network
```

**交付：**从数值态构造的无限 MPO，及不会混淆 λ 计数、指标顺序和步长的局部定义。

### 任务 4：确定边界固定点及其对称结构

**Files:** 新建 `transfer-spectrum/boundary.jl`，扩充测试与 README。

**Interfaces:** 消费 `TransferNetwork`；提供 `BoundaryPair`、`solve_boundaries(net;options)`。

- [ ] **Step 1 — 添加固定点测试并确认失败。** `test_exact_boundary_representatives` 检查解析参考的 r₀、r₁ 具有一致主导本征值密度，分别保持 C，单层作用联系两者。`test_gauge_equivalent_boundary_is_not_distinct` 对一个代表做可逆 MPS 规范变换，要求判定为同一态；真正的两个代表应有不同的规范不变重叠。`test_boundary_zero_schmidt_support` 检查移除冗余虚拟支撑后固定点不变、局部 W 的偶奇空间仍完整。
- [ ] **Step 2 — 接入 `leading_boundary` 与 VUMPS。** 从只保持对角 Z₂ 的边界 MPS 开始；解析检查先使用最小充分 χ，数值态也从低秩表示开始。χ=1、2、4 是按失败需要扩张的上限序列，不是默认全套收敛扫描；达到目标即停止。达到 χ=4 仍不能获得可靠背景时，报告并讨论，不自动扩算。
- [ ] **Step 3 — 构造第二个代表及必要的伴随边界。** 对 r₀ 施加 Gₖ，规范化为 r₁，独立检查其固定点关系和不同性；边界环境从各自背景建立。若当前规范非厄米，另求伴随边界并统一重叠归一化；不把它们误作 g=1 的两个渐近背景。相等的主导本征值密度与单层关系必须在残差预算内成立。
- [ ] **Step 4 — 运行边界测试，并在正式数值态上求边界。** 保存背景张量、实际 χ、Galerkin 残差、归一化和每格重叠。记录 Galerkin 残差是投影诊断，不将其单独当作完整本征方程误差的严格界。若边界是不可用于畴壁构造的对称叠加或冗余非 injective 表示，先修正表示再继续。

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/transfer-spectrum/tests.jl --group=boundary
```

**交付：**两个可用于边界激发的背景、它们的关系及所需环境，不以“调用返回了 MPS”作为完成条件。

### 任务 5：求解四类 sector 的动量分辨转移谱

**Files:** 新建 `transfer-spectrum/{sectors.jl,spectrum.jl}`，扩充测试与 README。

**Interfaces:** 消费 `TransferNetwork`、`BoundaryPair`；提供 `SectorLabel`、`sector_backgrounds`、`solve_sector` 和第 3 节的 `SectorPoint` 数据约定。

- [ ] **Step 1 — 添加非零谱接口对照并确认失败。** `test_statmech_largest_modulus_and_normalization` 使用局部虚拟 MPO bond 为 1 的乘积算符：物理 W 的偶、奇块各二维，偶块对角值 (5,3)，奇块对角值 (2,1)，背景为值 5 对应的乘积态。要求 q=0 的非平凡比值为 3/5，q=1 的两个比值为 2/5、1/5；只取一个分支时应得到最大模 2/5。将整个局部算符乘 7 后比值不变。这是 API 控制例，不是另一种物理模型的研究任务。
- [ ] **Step 2 — 实现受版本约束的激发适配。** 用普通 MPO 的实数动量入口或显式单行 Multiline 调用 `excitations` 与 `QuasiparticleAnsatz`；确认进入 statmech 最大模路径和正确归一化。对照测试至少含 k=0.0、π/2。底层有效作用的残差重算如需使用非导出接口，集中到本文件并记录包版本，不传播到物理网络模块。
- [ ] **Step 3 — 添加四 sector 与物理标签检查。** `test_sector_background_selection` 要求 g=0 使用同一背景对象，g=1 使用真正不同的代表，q 正确传入。`test_e_m_string_dictionary` 通过解析 P 的虚拟 Z/物理 X 关系，以及开放物理字符串在端点与 A/B 的对易或反对易关系，核对表中名称；无需额外优化激发态。
- [ ] **Step 4 — 实现逐动量求解和残差。** 默认先求每 sector 一个最大模分支；只有需要展示更多谱时才增加，始终不超过实际允许变分维数。对单位范数准粒子参数，用 $\|T_{\rm eff}B-rB\|$ 作为归一化有效算符的绝对残差，不能除以趋零的 r。记录投影残差与背景残差，完整空间的误差没有据此自动得到严格界。
- [ ] **Step 5 — 添加并实现零谱和失败状态检查。** `test_zero_empty_and_unconverged_are_distinct` 覆盖第 3 节六种状态。精确零有效作用可使 Krylov 提前终止；在解析低维检查中可用允许准粒子参数基上的有效作用确认其为零，避免反复迭代零算符。必要的小型投影矩阵检查只在该变分空间中进行，不变成有限圆柱替代。空融合通道不能伪造一个零本征值，未收敛返回值也不能标为通过。
- [ ] **Step 6 — 运行谱接口测试后，再生成正式数据。** 先取 k=−π、−π/2、0、π/2、π 的诊断点；接口无误后可取 [−π,π) 上 32 个均匀点作为零场图的初始网格。这不要求出现非平凡谱支。任何算例没有通过残差检查时，先保存失败点而非自动增加网格或改换物理态。

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/transfer-spectrum/tests.jl --group=spectrum
```

**交付：**按 (g,q,k) 分类、归一化清楚且带有效残差与状态的谱；固定点本征值在单独记录中展示，不强行塞入与背景正交的准粒子空间。

### 任务 6：正式运行、谱图、报告与集成检查

**Files:** 新建 `transfer-spectrum/{run.jl,plot.py,.gitignore}`，扩充 `tests.jl`、`README.md`；修改 `tests/runtests.jl` 接入两套轻量检查。正式运行后新增独立结果目录与报告；只有实际新结果取得后才更新项目的结果说明。

**Interfaces:** 输入任务 2 的已验收 `final.json`；输出固定点、逐动量谱、状态、图及报告。Plot 只消费保存的数据，不调用 Julia 或重新求谱。

- [ ] **Step 1 — 添加输出与绘图测试并确认失败。** `test_input_state_provenance` 要求输入记录为数值优化态且验收通过；解析态只能通过显式 reference 测试路径读取，不能成为正式结果。`test_incremental_output_and_no_overwrite` 检查已有目录拒绝覆盖、每个 sector/动量结束即写入、超时记录 budget。`test_plot_zero_and_invalid_modes` 用合成数据验证颜色/符号、空值与未收敛样本不会被画成有效有限谱支。
- [ ] **Step 2 — 实现独立 runner 和只读作图。** Runner 保存输入哈希、包版本、方向、步长、χ、背景、动量网格、分支数、容差、停止原因及源文件哈希。输出 `settings.json`、`boundaries.*`、`spectrum.jsonl`、`spectrum.csv`、`summary.json`。Plot 优先用 Python 标准库输出 SVG，避免为了作图安装或升级环境；若复用已有绘图库，先确认可用并保持无新依赖。
- [ ] **Step 3 — 明确零值表示。** 同时画 $|r(k)|$ 与 $-\log|r(k)|$；蓝/红分别表示 g=0/1，加号/圆圈分别表示 q=0/1。精确参考零值在能谱图标为 ∞；低于数值分辨率的生产值另作标记。相位未定义、无模式或能谱无穷时，JSON 使用 null 加状态字段，不能写非标准 NaN/Infinity。不得用 10⁻¹⁶ 替换零并宣称有限能量约 36。
- [ ] **Step 4 — 通过输出组测试并估算正式任务预算。** 输出组只使用合成数据和极小对象，不重算基态；正式转移谱的本地预算另设为最多 600 秒，与任务 2 分开。每个动量结束输出一行、flush、增量写入。开跑前重新确认本次使用的数值态和计算设置。
- [ ] **Step 5 — 生成正式图和短报告。** 在全新结果目录运行，从数值末态构造网络、求边界、求四 sector；用保存的 CSV/JSONL 独立作图。报告说明零场退化、零谱与分辨率、实际拟合和边界精度，以及 e/m 标签的校准依据。不把单一起点推广为普遍随机收敛，不把谱量解释成物理能量。
- [ ] **Step 6 — 接入并运行相关回归。** 修改总测试入口时继续隔离模块命名空间；默认只跑轻量软件检查，不运行历史优化、场扫描或新生产任务。运行下面专项与总回归；检查新增说明和链接。用户允许时才将独立任务改动提交到本地仓库，不自动推送。

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/transfer-spectrum/tests.jl --group=output
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/one-site-z2-su/tests.jl
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/transfer-spectrum/tests.jl
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/tests/runtests.jl
```

正式 runner 接受 `--state=<已验收数值final.json>`、`--num=1`、`--nk=32`、`--budget=600`、`--out=<全新目录>`；图命令接受 `--input=<spectrum.csv>`、`--out=<图目录>`。不要在计划中预先伪造一个已存在的新末态路径。实现后在 README 与最终交付消息中给出实际可运行的一行命令。

**交付：**可重复运行的严格零场计算、四类谱数据及图、物理和数值解释。即使零场没有 Fig. 7 那样的有限色散支，只要零谱、固定点和 sector 结构按定义正确记录，仍可完成本轮目标。

## 6. 执行边界、完成状态与新 session 交接

六项顺序为 **1 → 2 → 3 → 4 → 5 → 6**。任务 1 内含张量定义与初始化，不再拆成独立工程任务。首次执行回合建议只完成任务 1 的实现和轻量检查，然后报告；任务 2 的正式计算须按项目要求再次确认设置与预算。若用户在新 session 明确授权更大范围，也仍需遵守正式计算前的确认规则。

新 session 可直接使用以下指令：

```text
请阅读 AGENTS.md、tracks/peps/solutions/anyon-correlator/AGENTS.md，以及
tracks/peps/solutions/anyon-correlator/docs/2026-10-23-zero-field-z2-transfer-spectrum-plan.md。
按已确认的六项方案执行，不重新选择研究路线。先完成任务1：用 TensorKit 原生 Z2
对称张量实现保持单组 V/P/lambda、1-site 平移和 C4 旋转约束的 loop/cluster
simple update，并运行该任务的轻量检查。暂不运行完整基态或转移谱生产计算。
保留旧程序与数据，不升级 Julia 环境。完成任务1后报告改动、测试和剩余风险，
正式计算前再确认设置及预算。当前方案仅针对严格零场，不启动旧 PLAN.md 中的有限场工作。
```

本文撰写后的自查范围：六项任务覆盖科学设计；接口名与状态名在任务之间一致；五项 Review Focus 均有对应检查；测试步骤不包含数值算法函数体；执行命令只指向计划新增文件并明确其尚未创建。后续实际结果和复现命令写入相应实验 README/报告，不回填成本文撰写时已经完成的事实。

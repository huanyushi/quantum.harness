# Independent V–λ–P automatic-differentiation experiment

Approved 2026-10-05. Execution agent: OpenCode / openai/gpt-6.1-sol.
This is a new numerical experiment, not a continuation of the old dense tied-PEPS optimizer.

## Physics and scope

H = −Σₛ Aₛ −Σₚ Bₚ −hₓΣᵢXᵢ −h_zΣᵢZᵢ; Jₑ=Jₘ=1, Pauli operators.
Aₛ=∏X, Bₚ=∏Z. Infinite square lattice, physical spins on edges.
The repeating 2×2 vertex cell has four rank-4 V tensors, eight rank-3 P tensors
(physical d=2), and sixteen diagonal nonnegative λ connections. Different cell
positions are independent variables. Start at D=2; D=3 is an accuracy fallback.
No virtual-Z₂ constraint is imposed in this ground-state-only experiment.

Primary field points: (0,0.1), (0.1,0), (0.1,0.1). The h=0 exact state is a
calibration, not an undisclosed initial condition for a random-start trajectory.
Main initialization: random real V/P tensors and positive λ. Existing loop-update
states are benchmarks and separately labelled optimization controls, not main starts.

## Ansatz and differentiation

Train real V, P_E, P_N, and logarithmic λ parameters. Remove trivial tensor scales
by smooth Frobenius normalization; center each λ logarithm to fix its scale.
Keep the split-network bond-rank restriction throughout optimization. Blocking is
an exact, differentiable contraction for measurement, not an optimization of a
general unconstrained merged tensor. Distribute each λ by square roots at its two
endpoints for a balanced measurement gauge; do not differentiate through ALS/SU.

Use the installed TensorKit/PEPSKit/Zygote stack. Independently construct the
Hamiltonian and all local observables, and differentiate the CTMRG energy using
PEPSKit's fixed-point pullback. Verify the complete parameter-to-energy derivative
with multiple centered finite differences before accepting optimization evidence.
Full-iteration AD is a diagnostic fallback if fixed-point differentiation fails.

## Optimization and environment reliability

Use a safeguarded descent optimizer with a backtracking line search. The baseline
energy, differentiated environment and trial objective must refer to compatible
contractions. Fresh deterministic CTMRG trials prevent blindly transporting a warm
environment into a pathological basin. Independently recontract accepted/final
states with random initial environments. Failed convergence, substantial imaginary
parts, operator-bound violations or branch disagreement are failures, not successes.

Start contraction probes at χ=8 and check χ=16. Increase χ or use the already
available boundary-VUMPS measurements only when actual disagreement warrants it.
Record operator values at every position; a small CTMRG residual alone is insufficient.
Optimizer stationarity and contraction convergence are separate acceptance conditions.

## Benchmarks and acceptance

References: exact h=0 energy e=−1 and all stabilizers +1; the tenth-order low-field
energy and its analytic magnetization derivatives from arXiv:1012.1740v2,
e_ours(hₓ,h_z)=2e_paper(hₓ/2,h_z/2); existing loop-update measurements.
Series-order differences are sensitivity estimates, not certified error bars.

First-stage accuracy targets: |e_AD−e_series|≤1e-5 per edge spin and relative
error≤1% for nonzero reference magnetizations. Require |local Pauli expectation|≤
1+1e-6; single-axis conserved-stabilizer error≤1e-6; fresh-initialization energy
agreement≤1e-6/spin, observable agreement≤1e-5; the same tolerances for χ checks.
Zero reference magnetizations use an absolute tolerance of 1e-6. Report actual
gradient norms and optimization termination; iteration-budget exhaustion or a failed
line search is not automatically stationarity. The projected, scale-normalized
gradient-norm target is 1e-5 for the per-spin objective. Independently evaluate rejected
states and retain their failure evidence. These are finite-D approximation targets,
not a proof of the exact infinite-lattice ground state.

Measure improvement relative to loop-update in the same V–λ–P ansatz separately
from main random-start success. Do not change the thresholds after seeing results.

## Files and reproducibility

All new experiment code, design/plan, tests, report and figures live here.
Generated trajectories, settings, logs and checkpoints live in a local `results/`
subdirectory. Flush progress and write incremental output after accepted steps and
field points. Record source revision, package versions, seeds, D, χ, tolerances,
initialization, optimization status, wall time and benchmark errors. Retain a
single root-relative run command and explicit artifact paths in the final report.
No existing scientific code or results are overwritten; no commits/pushes without
an explicit user request.

## Compute feasibility

Before physical runs, benchmark a bounded D=2 objective/gradient probe, separating
compilation from repeat-call runtime. Forecast optimization cost from measured
objective/gradient/line-search costs and the step cap. Local work must fit below
10 minutes and 16 GB per declared bounded task under the initial policy. D=3 is
forecast separately.

### User-directed local diagnostic amendment

After finite-field AD cold-start timeouts, the user explicitly excluded SCNet: its
profile belongs to the harness holder and is not the group's compute resource.
No further use or repair of that connection. The user approved trying local runs,
including zero field, smaller χ and boundary VUMPS if needed. The next diagnostic
uses random D=2 parameters at hₓ=h_z=0, χ=4 with χ=8 auditing. A single process is
externally capped at 25 minutes (cold-start allowance plus a 300-second optimizer
loop), with memory monitoring; this is an explicit local diagnostic exception, not
an unattended production sweep. Its result cannot substitute for finite-field
accuracy or a larger-χ validation. The two-stage AD chain computes the packaged
PEPS-energy gradient first and then differentiates blocking to V/P/λ; this is the
same derivative, split to isolate constructor/compilation issues.

### Approved translation-uniform random experiment

The independent-tensor seed-19 zero-field trajectory reached e≈−0.75 rather than
the ground state despite a passing six-direction/step-size derivative probe. The
user then approved a translation-uniform random experiment: one V, one P_E, one
P_N and four directional λ vectors, repeated exactly across the stored 2×2 cell.
At D=2 this has 40 raw parameters rather than 160. Differentiation is with respect
to the shared parameters (positional contributions summed by AD, not averaged by
an undocumented factor). The V–λ–P network and its bond-rank restriction remain
unchanged. The independent trajectory is retained as rejected evidence. Each field
in the new suite starts directly from the same recorded uniform random parameters,
not from the exact state or a previous optimized field. One bounded local suite
reuses compilation across the zero and three finite-field points; optimizer budgets
are 120 seconds per point and an external wall cap is 25 minutes. These small
budgets may reject a point; no threshold is relaxed to turn that into success.

### Approved analytic-anchor optimization control

After the bounded uniform random suite failed at all four points, the user approved
a separately labelled analytic-anchor AD control: initialize the same uniform split
parameters with the exact h=0 V/P/λ, and optimize at the three h=0.1 weak-field
points (optionally h=0.05 if the larger points fail). This is a test of finite-field
AD response and near-ground-state optimization, not evidence for random global
search. Each field starts from the zero-field analytic parameters independently;
the exact zero-field state is not mislabelled a finite-field ground state. The
existing independent and uniform random failures remain in the report. Main source
and reference snapshots are saved per new run; the same first-stage acceptance
criteria apply to controls.

### 当前推进阶段（用户批准）

当前目标是验证不显式施加虚拟 Z₂ 约束的有限场 AD；sector-resolved transfer
spectrum 和对称张量约束留到后续阶段。首先在冻结 Z 场失败检查点核对中心有限
差分与环境分支，再实现有限次数 CTMRG 展开的 AD 参照（unrolled AD）。伴随
求解未收敛必须明确拒绝，不再将包返回的有限向量自动视为有效梯度。展开迭代
次数和 χ 的稳定性需实测；可靠导数建立后再继续小场优化及通用扰动起点。

用户明确允许数小时本地运行，取代此前 10–25 分钟的硬限制。每次长任务仍有
明确步数、时间和内存预算，逐步 flush、保存检查点、监控过程。若实测需求超过
本机资源，先向用户申请课题组资源；仓库内全部集群信息已过时，不再探测、修复
或提交其中任何连接。REPORT.md 用中文详细撰写，保留专业英文术语。

### 本阶段交付的精确定义

本阶段接受的是 uniform D=2、解析锚点或近基态起点的三个小场近似；随机
Gaussian 寻优与一般非对称恢复分别记录失败，不改称成功。接受路径为 N=20
unrolled AD，最终 N=40、逐 seed χ=8 与 χ=16 核对；选定 continuation cohort
另有最终 N=40 同图 FD、N=40/80 与 physical fresh FD。梯度范数是 normalized
tensor map 对 raw coordinates 的 chain-rule gradient，未额外实施完整 internal
gauge quotient projection。原精度门槛保持不变。

主 AD 路径只复用 sibling 的 `reference.jl`；独立 boundary 诊断复用 `core.jl`
和 `vumps_core.jl` 的 state adapter/边界测量，不调用其 physical SU/ALS 更新。
`dominant_contract` 按最大 positive norm cluster 选择，不按最低测量能量。无 warm
环境时至少两个 fresh 支持；带 warm 时至少两个支持且至少一个 fresh 匹配；
最终 dominant audit 每个 χ 至少三个 fresh 支持。这是诊断性主导分支证据，
不是全谱认证，也不改变失败恢复试验的拒绝结论。

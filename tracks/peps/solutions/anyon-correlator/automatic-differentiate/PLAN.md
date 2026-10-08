# Split-network AD Implementation Plan

> **For agentic workers:** Use `executing-plans` for inline task-by-task execution.
> Delegation requires explicit user selection. Do not commit or push without a request.

**Original goal:** Independently optimize the infinite toric-code V–λ–P network from
random parameters at three weak-field points and verify its accuracy against series data.
**Current approved milestone:** Validate unconstrained-virtual-Z₂, uniform D=2 AD
from analytic-anchor/near-ground-state starts at those points, while retaining random
and generic nonsymmetric-start failures as separately labelled evidence.

**Architecture:** Pure parameter-to-network construction followed by a packaged
CTMRG fixed-point or explicitly unrolled pullback; a safeguarded vector optimizer
operates only on V/P/λ. The fixed-point route rejects logged adjoint nonconvergence;
the accepted small-field route uses N=20, checked against N=40.
Separate implementation tests, derivative probes, production trajectories and
frozen-state audits; checkpoint every accepted step.

**Tech Stack:** Julia 1.12.6; installed TensorKit 0.16.5, PEPSKit 0.8.0,
Zygote 0.7.12, JLD2, JSON; Python/matplotlib for collected-data plots.

## Global Constraints

- Follow `DESIGN.md`; all new files live in `automatic-differentiate/`.
- H=−ΣA−ΣB−hₓΣX−h_zΣZ; J=1, Pauli convention; e is per edge spin.
- Infinite 2×2 repeat: four V, eight P, sixteen λ; physical d=2, initially D=2.
- Independent-cell or approved translation-uniform sharing, real parameters,
  logarithmic positive λ; no SU gates or old optimizer in the AD trajectory.
- Energy accuracy 1e-5/spin; nonzero magnetization relative accuracy 1%; zero
  magnetization absolute accuracy 1e-6; gradient-norm target 1e-5/spin objective.
- Physical excess≤1e-6; conserved-sector error≤1e-6; initialization/χ energy
  agreement≤1e-6/spin and observable agreement≤1e-5.
- Residuals, termination, rejected branches and compute limits stay explicit.
- No physics run before setup ratification (already obtained). The latest user
  amendment permits multi-hour monitored local tasks and overrides the original
  10-minute limit. All repository cluster configurations are obsolete and excluded;
  request group resources from the user if measured demand exceeds this machine.

## Files

- `network.jl`: flat parameter layout, random/exact/snapshot input, differentiable
  balanced blocking, independent Hamiltonian and site observables.
- `objective.jl`: CTMRG contraction, fixed-point AD, directional finite differences,
  physical gates and frozen-state multi-initialization/χ audit.
- `optimize.jl`: safeguarded L-BFGS with descent-direction reset and Armijo search;
  incremental trace and checkpoints, budget and stationarity handling.
- `run.jl`: explicit CLI modes `probe`, `run`, `audit`, settings/provenance.
- `tests.jl`: independent dense reconstruction, operator, chain-rule and acceptance
  regression tests; no long physics trajectories.
- `analyze.py`: read saved artifacts and produce numerical summary/plots.
- `README.md`, `REPORT.md`: entry command, results, evidence and limitations.

## Original task breakdown

The checklists below preserve the original instructions, including conditional work.
The execution record at the end tracks the approved milestone and distinguishes
completed experiments from a successful physical result.

## Task 1 — Differentiable split-network representation and operators

**Interfaces:** `initial_parameters(D, seed; exact=false)::Vector{Float64}`;
`split_parameters(x,D)` yields V/P/λ arrays; `blocked_arrays(x,D)` yields a 2×2
matrix of arrays with shape (4,D,D,D,D); `make_peps(x,D)::InfinitePEPS`;
`operators(hx,hz)` returns Hamiltonian plus named individual LocalOperators.

- [ ] Add an independent scalar-loop blocking oracle in `tests.jl` and compare it
  with `blocked_arrays` for nonuniform λ. For internal contractions use
  `T[p,q,n,e,s,w]=Σa,b V[b,a,s,w]P_E[p,a,e]P_N[q,b,n]` and include each λ once.
- [ ] Run the test before implementation: expect an undefined-function failure.
- [ ] Implement `network.jl` with immutable slicing/map contractions. Normalize
  each V/P by Frobenius norm; set `λ=exp.(u .- sum(u)/length(u))`; balance by
  multiplying √λ into both incident tensors. Preserve Julia first-index-fastest.
- [ ] Verify the complete balanced wavefunction on a 2×2 torus against unbalanced
  reconstruction, and verify blocked-vs-split derivative agreement for an arbitrary
  dense linear functional with centered differences.
- [ ] Construct physical operators independently using index-level Pauli action:
  star edges E(r,c−1),E(r,c),N(r,c),N(r+1,c); plaquette edges
  E(r−1,c),E(r,c),N(r,c),N(r,c+1). Compare dense operator matrix elements.

Root-relative command for short tests:
`OPENBLAS_NUM_THREADS=1 julia -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/tests.jl`

## Task 2 — Objective, physical audit and true AD-chain derivative

**Interfaces:** `contract(x,D,H; chi,tol,maxiter,seed)` returns energy/psi/env/info;
`energy_gradient(x,D,H,contraction; tol)` returns per-spin energy and real gradient;
`measure(psi,env,ops,hx,hz)` returns all local values and mean quantities;
`audit(x,D,hx,hz; chi,chi_check)` returns labelled rows and a verdict.

- [ ] Write regression checks that operator-bound violations, insufficient seeds,
  incompatible branches, nonstationary optimizer and χ instability fail acceptance.
- [ ] Use `CTMRGEnv(randn,ComplexF64,psi,ComplexSpace(chi))` with a recorded seed;
  converge `SimultaneousCTMRG(tol=1e-9,maxiter=300,verbosity=0)` and require
  `info.converged`. Actual iteration telemetry unavailable: store `missing`, never
  infer it from `length(info.contraction_metrics)`.
- [ ] Differentiate the complete map using
  `withgradient(x) do y; psi=make_peps(y,D); env_ad,_=PEPSKit.hook_pullback(leading_boundary,env,psi,boundary_alg;alg_rrule=gradient_alg); real(PEPSKit.cost_function(psi,env_ad,H))/8; end`.
- [ ] Run exact zero-field calibration and a random-state probe. Compare two random
  directional derivatives and the negative-gradient derivative at at least two
  step sizes against centered finite differences of fresh contractions; record
  disagreement rather than proceeding with an unverified derivative.
- [ ] Time repeat objective/gradient calls, record package versions, and forecast
  the full optimization before scheduling it.

## Task 3 — Safeguarded optimizer and reproducible runner

**Interfaces:** `optimize(x,D,hx,hz; config,outdir)` returns checkpoint, terminal
gradient and explicit status; `run.jl` accepts field, seed, χ, step cap, time budget,
initialization and output path. Output must use a new or explicit resume directory.

- [ ] Test L-BFGS two-loop direction against a positive-definite quadratic;
  reject non-descent directions and reset history. Test budget exhaustion is not
  convergence, and branch/physical failures reject trials.
- [ ] Store at most 8 secant pairs `(s,y)` only when `dot(s,y)>1e-10*norm(s)*norm(y)`.
  If the proposed direction has `dot(g,d)>=0`, reset and use `d=-g`.
- [ ] Armijo acceptance requires compatible fresh energy decrease
  `e_trial<=e+1e-4*alpha*dot(g,d)` and convergence/physical checks. Cap the parameter
  step norm, halve α at failed trials, and veto inconsistent independent seeds.
- [ ] Record step, energy, gradient norm, α, χ, residual, mₓ/m_z, stabilizers,
  rejection reason and elapsed seconds; flush each progress line and checkpoint
  every accepted step. Recompute the gradient on the final tensor.
- [ ] Implement CLI `probe`, `run`, `audit` and explicit snapshot-control input.
  Save settings and initial/final parameters so trials can be repeated exactly.

## Task 4 — Bounded random-start experiments and independent controls

- [ ] After successful probes, run (0,0.1), (0.1,0), (0.1,0.1) from recorded random
  V/P/λ values; no hidden exact-state restart or SU trajectory.
- [ ] At final stationarity audit frozen parameters from deterministic and ≥2 random
  environments at χ=8 and fresh χ=16. If needed, diagnose rather than average
  discrepant branches; use existing boundary-VUMPS tooling only as a labelled
  independent contraction after an actual failure.
- [ ] Compare e/m/stabilizers with tenth-order reference data and loop-update values.
  For a same-ansatz control load the existing D=2 portable snapshot, and independently
  optimize its V/P/λ. Label it `loop-control`, never `random`.
- [ ] If D=2 is stationary and contractions reliable but accuracy insufficient,
  report the D=2 result, forecast D=3 separately and execute only within the declared
  compute policy. Do not silently relax acceptance thresholds or claim guaranteed
  arbitrary-D accuracy.

## Task 5 — Evidence review and handoff

- [x] Review derivative records, final gradient, every audit verdict, physical bounds,
  reference errors, input provenance and wall-time budget against `DESIGN.md`.
- [x] Read artifacts with `analyze.py`; save a convergence plot and a benchmark-error
  plot. No new physics computation is launched by analysis.
- [x] Write `REPORT.md` with accepted/rejected results, random/control distinction,
  exact-state and AD checks, contraction uncertainty, and next steps.
- [x] Write `README.md` with reproducible root-relative commands and artifact paths.
- [x] Run targeted tests and `git diff --check`; inspect changed files and do an inline
  code review. No commit or push has been requested.

## 当前执行补充：无显式 Z₂ 约束的导数诊断与修复

用户已批准继续这一阶段，并允许数小时本地运行；旧集群信息全部禁止使用。
只修改本实验目录，中文详细报告与 simple-update/REPORT.md 的叙述方式一致。

- [x] 为非收敛伴随解增加真实日志捕获回归测试：失败求解即中止梯度使用，保存
  失败原因，不依靠返回向量是否 finite 来判断成功。
- [x] 新建 `diagnose.jl`，加载 Z 场原失败 checkpoint，在原参数化、χ=8 上执行
  N=10/20/40 的固定次数环境迭代展开 AD。比较同一冻结初始环境下的计算图有限
  差分，以及 fresh CTMRG 的中心有限差分和分支稳定性；每段即时写 JSON。
- [x] 用 `leading_boundary` 的 `miniter=maxiter=N` 与 `alg_rrule=nothing` 定义
  明确的展开计算图，复用 PEPSKit 而非自行重写 CTMRG。起始环境是冻结参考，不
  对它伪造导数。有限展开导数与无限固定点导数要明确区分。
- [x] 选择通过有限差分和迭代次数稳定性检查的梯度方法，并在优化中显式记录
  方法、N、梯度检查、最后张量上的导数与物理量。必要的内存优化须保持同一图。
- [x] 先完成 Z 单轴和双分量场，再复核 X 单轴。采用可信起点；随后以小幅通用
  随机扰动检查未施加虚拟对称性时的恢复能力。随机 Gaussian 全局寻优单独报告。
- [x] 每个最终态执行已声明的 multi-seed/χ audit；输出中文完整 REPORT.md、
  复现命令、源代码快照、检查点、逐步数据和结果图。

## 当前执行记录

- [x] 独立位置与 uniform V–λ–P 构造、算符、parameter→TensorMap chain rule 和
  safeguarded optimizer 已实现，并有对应测试和实际方向有限差分。
- [x] Independent/random-uniform 实验已执行并保留；随机寻优未通过，不以
  analytic-anchor 成功替代。原计划 same-ansatz loop-control 优化及 D=3 fallback
  未执行，未纳入本阶段成功结论。
- [x] `diagnostic-z-unrolled02` 完成 frozen Z 的 N=10/20/40、同图/physical FD；
  generic fresh 分支歧义保留。
- [x] 选定 X/Z/双分量结果分别来自 `unrolled-repair01/x`、`unrolled-repair01/z`、
  `unrolled-repair02/xz`，通过最终导数、N 和逐 seed/χ 验收。
- [x] `fresh-unrolled-suite01` 从解析零场独立启动四点并全部通过，无历史 finite-field
  checkpoint 依赖；初始同图 FD 与最终 N=20/40 证据单独保存。
- [x] 通用非对称扰动及独立 boundary/norm 检查完成；较低能量不用于选择同一
  frozen tensor 的环境。`noisy-recovery03` 更新后伴随/最终 audit 失败，恢复未通过。
- [x] Accepted portable states 已导出，summary/checkpoint hashes 及源码快照已核对。
- [x] 最终图表、中文文档、软件测试和全部新文件审查完成后交付。

交付前复核：三组主结果及 fresh 四点的原始 observables 重新计算全部验收条件；
源码快照与 portable-state hashes 相符。四组软件测试共 91 项再次通过，记录于
`results/final-verification.log`。15 个 Julia 文件和 Python source 语法检查通过；
28 个源码/文档/数据文件无 whitespace errors，3 个 SVG 的 XML 与3 个 PNG 校验
通过。Matplotlib 原生 SVG 的 path 属性保留其生成格式。四个 CSV 统一 LF 换行，
再生成前后全部解析数据逐项相同，选定主结果来源不变。

未解决的研究目标：Gaussian 全局随机寻优、一般非对称起点的稳定恢复、一般
D/χ 收敛、精确 duality closure 以及 transfer/anyon sector。它们不因当前软件
验收或三个小场点通过而自动完成。

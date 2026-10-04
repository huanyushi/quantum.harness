# Toric code in a parallel magnetic field — 项目审计摘要

- 审计日期：2026-10-02。
- 项目定位：源自 Quantum.Harness challenge #50，现转为长期科研项目，计划迁移到独立私有仓库。
- 审计对象：`tracks/peps/solutions/anyon-correlator/`、本地 `tracks/peps/results/`、Julia 环境与相关 package 源码。
- 审计方式：只读检查 Git、文档、26 个保留的 Julia scripts/tests、结果目录、关键 CSV/日志及软件 API 实现；未运行 tests、科研计算、package update 或 instantiate。
- 本文件在审计结束后，经用户另行授权创建；没有修改科研源码或数据，没有 commit 或 push。

## 1. 总体结论与证据等级

**项目已有可靠的零场能量与局域稳定子校准结果，但尚未得到被接受的有限场基态，也没有完成 anyon correlator 或 sector-resolved transfer spectrum。适合迁移的是科研代码与诊断证据，而不是已经验证的有限场生产流程。**

本文件使用以下区分：

| 等级 | 含义 |
|---|---|
| 代码存在 | 源码可读，可确认实现意图；不代表运行成功 |
| 运行有据 | 对应 CSV、日志或 checkpoint 存在；不代表物理结果可信 |
| 历史测试声明 | 文档或 commit 声称测试通过，本轮没有重新执行 |
| 物理结果可信 | 已有数据支持限定范围内的物理结论；不是严格认证 |
| UNVERIFIED | 缺少直接证据，或本轮未做必要验证 |

路径缩写：

- `P/` = `tracks/peps/solutions/anyon-correlator/`
- `R/` = `tracks/peps/results/`

## 2. Git 基线

| 项目 | 审计时状态 |
|---|---|
| Repository path | `/home/huanyushi/code/quantum.harness` |
| Branch | `challenge/peps-anyon-correlator` |
| HEAD | `7cf2a066625d6e514265e03562956c5f3f392d2c` |
| Commit 标题 | `[peps] anyon-correlator: uncompleted challenge report (M3/6, time limit)` |
| Commit 时间 | 2026-07-30 20:05:28 +0800 |
| Git status | 审计开始及结束时干净；创建本文件前再次确认干净 |
| Tracking branch | `origin/challenge/peps-anyon-correlator` |
| 与本地 tracking ref 的距离 | ahead 0 / behind 0 |

Remotes，fetch/push 相同：

- `origin`: `https://github.com/huanyushi/quantum.harness.git`
- `upstream`: `https://github.com/QuantumBFS/quantum.harness.git`

没有 fetch，以上同步状态不能证明远端在审计当日没有新增 commit。

`P/` 源码、文档、两张 SVG 和 `julia-env/Project.toml` 已跟踪。`R/` 被根目录 `.gitignore:22` 忽略，`julia-env/Manifest.toml` 被 `.gitignore:9` 忽略。干净工作树不意味着数据和环境锁定文件已纳入 Git。

## 3. 科研设置与当前进度

源码约定：

> H = −Σₛ Aₛ − Σₚ Bₚ − hₓΣᵢXᵢ − h_zΣᵢZᵢ，Jₑ = Jₘ = 1。

- Aₛ 是四边 X-type star，Bₚ 是四边 Z-type plaquette。
- 无限平方晶格 iPEPS；每个 composite site 含 east/north 两个物理 qubit，physical dimension 4。
- `(2,2)` supercell 有 8 个 edge spins。已成功 M2 与尝试过的 M3 使用一个 dense complex D=2 tensor，绑定到四个位置。
- 当前有限场路径为 hₓ=0 的 `(0,h_z)` 轴，不是一般二维 `(hₓ,h_z)` 扫描。
- 依据：`P/scripts/tc_peps.jl::toric_code_hamiltonian`、`ad_tied_core.jl::tied_peps`、`m3_series_validation.jl::run_series_point`。

| 阶段 | 审计判断 |
|---|---|
| M0 | Julia 与主要 package 版本有本地证据；当前 package-load smoke 未重跑 |
| M1 | Hamiltonian/ED 源码和 12 点 CSV 存在，零场谱符合预期 |
| M2 | 零场能量与局域稳定子达标，有直接数据支持 |
| M3 | 未完成；已有失败数据，A2 实现尚无运行证据 |
| M4 | 没有可接受的相变位置估计 |
| M5 | 没有完整 anyon correlator / sector spectrum 工作流或相应结果 |
| M6 | 有 challenge 历史总结，不是科研目标的完整交付 |

已阅读：`CHALLENGE_SUMMARY.md`、`PROGRESS.md`、`PLAN.md`、`M2_REPORT.md`、`M3_REPORT.md`，并与源码及本地数据交叉核对。项目 `README.md` 仍写“paused before M3”，已经过时。

## 4. 可信结果及限定范围

### 4.1 零场 2×2 torus ED

`R/20260728-114418-ed-checks/ed_2x2.csv` 直接记录 E₀=−8、ground-state degeneracy=4、ground space 以上的 gap=4。

对应：`P/scripts/ed_checks.jl::hamiltonian`、`spectrum0`、`acceptance_gates`。这是有限系统零场谱校准；“7/7 gates 全通过”为历史测试声明，本轮未重跑。

### 4.2 M2 random-start step086

| 量 | 记录值 |
|---|---:|
| E_cell | −7.999999995071725 |
| E/N | −0.999999999383966 |
| 最大 star error | 约 5.95×10⁻¹⁰ |
| 最大 plaquette error | 约 6.37×10⁻¹⁰ |
| CTMRG residual | 约 2.82×10⁻⁹ |

直接依据：`R/20260730-m2-chi8-warm-continue-77-to100/random-continue_energy.csv` 与 `random-continue_stabilizer_trace.csv` 的 step86。

同一 tensor 的 fresh contraction 给出 E_cell≈−8.000000001560，能量差约 6.49×10⁻⁹；后续 series anchor audit 的 χ8 多初始化与 χ16 检查也支持近零场一致性。

**可信范围：零场能量与局域稳定子在声明的 10⁻⁶ 容差内达标。** 这不等于 virtual symmetry、topological sector、anyon form factor 或 correlator 已验证；有限 χ 收缩也不是严格变分认证。

### 4.3 Exact tensor benchmark

`R/20260729-m2-single-process-chi4-20step/exact-smoke_energy.csv` 与 `exact-smoke_stabilizers.csv` 支持 E_cell≈−8、最大稳定子误差≈1.665×10⁻¹⁵。

文档的 projected gradient norm=4.526×10⁻¹⁶ 未找到独立原始输出，标记 **UNVERIFIED**。初始化 CSV 的 `gradnorm=0` 是占位值，不能作为零梯度证据。

## 5. 失败但有价值的诊断与记录偏差

| 诊断 | 直接证据及意义 |
|---|---|
| Graded SU 停在非基态 | `R/20260728-170126-groundstate-h0/energy_convergence.csv`：SU/AD 后 E/N≈−0.5，χ40 无改善；小梯度和 χ稳定不保证基态 |
| 小 χ 非物理收缩 | `R/20260729-m2-chi4-continue-20-to100/chi_diagnostic_steps67_68.csv`：部分 χ4/χ6 能量低于 −8、稳定子超过 1，χ8 给出不同结果 |
| M3 首轮曲线无效 | `R/20260730-m3-hz-coarse-chi6-fresh-armijo/m3_hz_points.csv`：11 点存在，五个正场点没有接受更新，稳定子超界 |
| χ6 repair point 失败 | `R/20260730-m3-repair-point-chi6/hz_0p100_audit.csv`：两个收敛 seed 均给出 max⟨Bₚ⟩≈1.0001717978；另一个 seed 失败 |
| Series pilot 分支分裂 | `R/20260730-m3-series-pilot-chi8/hz_0p050_audit.csv`：warm E/N≈−1.05000077、fresh≈−0.95581966；warm 稳定子明显超界 |
| A1 小步长仍失败 | `R/20260730-m3-series-pilot-chi8-a005.log`：步长持续缩小，6 步后 Armijo 失败；随机 seed 1 最终不收敛 |

重要偏差：

1. **M2 原始 verdict 不是 PASSED。** `…-warm-continue-77-to100/run.log:21–23` 写 Armijo failed、`VERDICT: random continuation FAILED`。后来 `M2_REPORT.md` 按能量/稳定子达标接受 step86，有数据支持，但不是当时程序自动判定成功。当前 `ad_tied_gd.jl::run_random_continuation` 计算 fresh repeat，却只用 primary 的 `continuation_passed` 决定 `passed`，未把 fresh agreement 加入最终布尔判定，与 `PLAN.md` 验收文字有差异。
2. **A1 不是实际预算耗尽。** `…-chi8-a005/series_points.csv`：accepted_steps=6、attempts=7、status=`armijo_failed`、audit_converged=3、consistency=`insufficient_inits`、point_accepted=false。文档的“budget-exhausted”不能替代实际终止状态。
3. **分支分裂起点的叙述过于简化。** `…-chi8/branch_divergence_0p05.csv`：steps1–2 的 fresh-random 差异已约 5.49×10⁻⁶ / 2.11×10⁻⁵ per spin，超过后来 10⁻⁶ 门槛；step3 fresh-det 失败，step4 两个 fresh 失败，step6 fresh-random 仍匹配 warm；step9 后两个 fresh 的一致恶化才清楚。不能把所有 fresh estimate 无条件称为“真实能量”。
4. **CTMRG iteration telemetry 有 API 误读。** `ad_tied_gd.jl::ctm_iterations` 返回 `length(info.contraction_metrics)`。PEPSKit 0.8.0 的 `~/.julia/packages/PEPSKit/rBPLa/src/algorithms/ctmrg/ctmrg.jl:100–143` 与 `simultaneous.jl` 显示该字段是最后一轮 metrics NamedTuple，不是 iteration history。大量 `ctm_iters=1` 是字段计数，实际迭代数 **UNVERIFIED**；残差和 `info.converged` 不因此自动失效。

## 6. 三个重点技术核查

### 6.1 Boundary MPS spectrum 与 PEPS row-to-row spectrum

**旧实现存在对象混淆。** `P/scripts/groundstate_h0.jl::main` stage8 求 `MultilineTransferPEPS` 的 boundary state 后，调用 `MPSKit.transfer_spectrum(mps; num_vals=8)`。

MPSKit 0.13.12 的 `~/.julia/packages/MPSKit/7FTCR/src/algorithms/toolbox.jl::transfer_spectrum` 求的是 `TransferMatrix(above.AL, below.AL)`，即 boundary MPS 自身的 overlap transfer matrix；调用没有传入 PEPS operator `T`，不是 `T` 的完整 row-to-row spectrum。

stage7 的 `correlation_length(ψ,env)` 使用 PEPSKit CTMRG edge-transfer approximation，依据 `~/.julia/packages/PEPSKit/rBPLa/src/algorithms/toolbox.jl::_correlation_length`。它也不等于已完成 flux-twisted、parity-resolved、form-factor-selected M5 spectrum。

未找到 `spectrum_h0.csv` 或 anyon spectrum 数据；optional code 是否实际执行成功 **UNVERIFIED**。

### 6.2 Exact tensor pulling-through relation

**tensor 公式一致，但文档单腿 Pauli 对应错误。** `tc_peps.jl::exact_tensor_dense` / `exact_tensor_dense_VP` 实现：

```text
T[pE,pN,n,e,s,w] = (−1)^(pE·e+pN·n) δₙ⊕ₑ⊕ₛ⊕w,₀ / 2
```

由源码公式直接解析推导：

- ZₙZₑZₛZ_w T=T：正确；
- Zₑ T=X_pE T；
- Zₙ T=X_pN T。

`P/PLAN.md:123–126` 却写 Zₑ T=Z_pE T、Zₙ T=Z_pN T，与当前 X-basis copy tensor 不一致。

`tests/runtests.jl` 没有显式测试这些 identities；既有测试声明不能为该段文档背书。seam、e/m assignment 与完整 sector machinery 仍 **UNVERIFIED**。

### 6.3 Series-validation 最终物理界 acceptance

**没有显式检查 ⟨Aₛ⟩、⟨Bₚ⟩、m_z 的绝对物理界。** `m3_series_validation.jl`：

- `evaluate_series_observables`：有限值、能量分解；
- `assess_series_consistency`：初始化之间的差；
- `assess_series_chi`：χ之间的差；
- `series_point_accepted`：optimizer status、consistency、χ stability。

位置：`116–128,268–308`。未要求每个 |⟨Aₛ⟩|、|⟨Bₚ⟩|、|m_z|≤1+tol。Armijo 的能量窗口不能替代这些界；多个初始化/χ一致地超界，静态逻辑仍可能接受。

旧 `m3_hz_continuation.jl::assess_m3_audit` 有稳定子界，`evaluate_m3_observables` 有 m_z 界，但 series driver 没有继承。

此外，`delta_e_per_spin` 只是输出，不是 acceptance gate；h_z=0 以 `:m2_anchor` 特例接受，不要求 measured gradnorm≤grad_tol。已有 anchor gradnorm≈1.145×10⁻⁴，高于配置 10⁻⁶。

## 7. 代码地图

### 7.1 正式入口与共享核心

| 文件（相对 P/） | 角色/关键函数 |
|---|---|
| `scripts/m3_series_validation.jl` | 当前科研继续入口；`series_main`、`run_series_pilot`、`run_series_point`；当前 grid=[0,0.05]，含未执行验证的 A2 |
| `scripts/ad_tied_gd.jl` | M2 driver 与 M3 helper；`tied_ad_main`、`optimize_tied`、`converge_tied_environment` |
| `scripts/ed_checks.jl` | 独立 ED benchmark；`main`、`acceptance_gates` |
| `scripts/tc_peps.jl` | spaces、Pauli、exact tensor、Hamiltonian、SU circuit |
| `scripts/ad_tied_core.jl` | tying、gradient projection、normalized descent direction |

```text
m3_series_validation.jl
└─ ad_tied_gd.jl
   ├─ tc_peps.jl
   └─ ad_tied_core.jl
```

没有已验证的 ground state → anyon correlator 统一生产入口。

### 7.2 Tests

- `tests/runtests.jl`：M1 ED、M2 tensor/Hamiltonian/SU/normalization；最后 include tied suite。
- `tests/tied_ad_core_tests.jl`：tying、projection、配置、M2 acceptance helper、mock Armijo。
- `tests/m3_hz_tests.jl`：旧 continuation、physical bounds、staging、CSV/checkpoint schema。
- `tests/m3_series_tests.jl`：series、配置、mock objective/veto、branch/χ acceptance、输出与 loader。

文档声明 94/94、38/38、132/132、69/69；部分文字仍写 65/65。本轮未执行，当前通过状态 **UNVERIFIED**；94/94 已包含 tied suite，不能与 38/38 作为独立总数相加。部分 tests 会创建临时文件。

### 7.3 已 supersede 的旧实现

- `scripts/groundstate_h0.jl`：graded SU → AD polish → optional spectrum，已标 legacy。
- `scripts/ad_ungraded_m2.jl`：四独立 tensor 早期 fixedpoint 优化；注释写 L-BFGS，源码实际选 GradientDescent。
- `scripts/ad_staged_gd.jl`：四独立 tensor staged Armijo GD。
- `scripts/ad_staged_gd_continue.jl`：stageC_step20 continuation 与 slow plaquette trace。
- `scripts/ad_symmetrized_gd.jl`：RotateReflect 投影实验。
- `scripts/m3_hz_continuation.jl`：首轮 M3 与 χ6 repair，已被 series 主协议替代。

### 7.4 Inspection/debug scripts

| 文件（相对 P/inspection/） | 用途 |
|---|---|
| `norm_probe.jl` | 归一化 |
| `gate_isolation.jl` | star-only / plaquette-only gates |
| `plaq_continue.jl` | stalled SU 后仅加 plaquette gates |
| `su_ungraded_test.jl` | dense V/P 随机 SU |
| `d_sweep.jl` | D=3,4,6 SU 对照 |
| `seed_probe.jl` | graded SU 六 seed 对照 |
| `ad_gradient_diagnosis.jl` | gradient solve / finite difference |
| `ad_full_vs_fixedpoint.jl` | full AD / fixed-point AD |
| `retract_path_probe.jl` | retract-path profile |
| `slow_plaquette_diagnosis.jl` | defect plaquette 方向导数 |
| `series_branch_divergence.jl` | pilot checkpoint 逐步 fresh re-contraction |

保留 Julia 文件的意图均可确认。不能确认的主要是部分产物生成 provenance 和历史执行输出：combined `figures/m2_energy_convergence.svg` / 本地 PNG 的完整生成过程未确定；stdout-only probes 缺少原始输出。`inspection/README.md` 未收录后来新增的 series divergence probe。

## 8. 本地数据盘点

- 28 个 run directories，另有 2 个顶层 series logs；5 个目录为空。
- `du` 占用：5,939,200 bytes，约 5.66 MiB。
- `du --apparent-size`：5,173,918 bytes，约 4.93 MiB。

### 8.1 全部 run directories（相对 R/）

| 类别 | Run names |
|---|---|
| ED | `20260728-114418-ed-checks` |
| 早期 SU | `20260728-152722-groundstate-h0`、`20260728-153221-groundstate-h0`、`20260728-154002-groundstate-h0`、`20260728-162725-groundstate-h0`、`20260728-163828-groundstate-h0`、`20260728-164534-groundstate-h0`、`20260728-170126-groundstate-h0` |
| 独立/对称化 GD | `20260729-090528-ad-staged-gd`、`20260729-150201-ad-symm-gd` |
| M2 smoke/probe | `20260729-m2-local-chi4-smoke`、`20260729-m2-serial-gradient-probe`、`20260729-m2-single-process-chi4-smoke`、`20260729-m2-single-process-chi4-20step` |
| M2 continuation | `20260729-m2-chi4-continue-20-to100`、`20260730-m2-chi8-continue-68-to100`、`20260730-m2-chi8-continue-69-to100`、`20260730-m2-chi8-warm-continue-77-to100` |
| M3 smoke | `20260730-m3-hz-smoke`、`20260730-m3-hz-smoke-chi6`、`20260730-m3-hz-smoke-chi6-guarded`、`20260730-m3-hz-smoke-chi6-fresh-armijo` |
| M3 coarse | `20260730-m3-hz-coarse`、`20260730-m3-hz-coarse-chi6-guarded`、`20260730-m3-hz-coarse-chi6-fresh-armijo` |
| M3 repair | `20260730-m3-repair-point-chi6` |
| Series pilot / A1 | `20260730-m3-series-pilot-chi8`、`20260730-m3-series-pilot-chi8-a005` |

早期 SU 的前五个目录为空；`164534` 只有 `su_log.csv`；`170126` 有四个失败结果文件。

### 8.2 M2 checkpoint

`R/20260730-m2-chi8-warm-continue-77-to100/random-continue_step086.jld2` 确实存在，14,723 bytes。SHA-256：

```text
f6a96e737075737bc7f808d9414cf0da3d9dda826c97f5eed0d5befbac8895b6
```

主要随机轨迹 checkpoints 仍在：random-smoke steps001–020、χ4 steps021–068、χ8 step069、χ8 steps070–077、warm χ8 steps078–086。

本轮未反序列化 JLD2；tensor 内容、schema 和当前版本 loader 兼容性 **UNVERIFIED**。文件存在与 CSV/日志一致，不代替 loader 验证。

### 8.3 M3 产物完整程度

| 批次 | 实际产物 |
|---|---|
| 原始 series pilot | 三点 summary/final/audit；0.05 steps01–23，0.10 steps01–10；SVG；23-step divergence CSV；顶层 log |
| A1 `…-chi8-a005` | 0 与 0.05 两点 summary/final/audit；0.05 steps01–06；SVG；顶层 log |
| A2 | 只有当前源码/tests 与准备记录，未找到运行结果 |
| χ6 repair | 两点 summary/final/audit，0.10 steps01–04；无 `STAGE_PASSED` |
| χ6 fresh-armijo coarse | 11 个 final checkpoints、14 个 step checkpoints、summary CSV、SVG；物理结果已否决 |

在审计的项目/results 路径中未找到：原始 test suite 输出、M0 smoke log、FD/full-AD/retract-path/slow-plaquette cosine 的持久化输出、SU isolation/recovery/完整 D-seed sweep 的独立结果表、exact gradient 原始输出、A2 数据、`spectrum_h0.csv`、sector spectra、form factors、correlators、ξ(h)、`run.json`。

旧 GD 有 checkpoints，SU 有部分数据；缺失的是上述具体诊断结论的直接输出，不能笼统说全部旧工作仅有 Markdown。

## 9. 环境与 harness 依赖

### 9.1 环境版本

Julia executable：`/home/huanyushi/.juliaup/bin/julia`；`julia --version` 实测 1.12.6，与 Manifest 一致。

| Package | Manifest version |
|---|---:|
| TensorKit | 0.16.5 |
| PEPSKit | 0.8.0 |
| MPSKit | 0.13.12 |
| MPSKitModels | 0.4.7 |
| Zygote | 0.7.12 |
| JLD2 | 0.6.5 |
| KrylovKit | 0.10.4 |
| QuadGK | 2.11.3 |
| OptimKit（transitive） | 0.4.3 |
| TensorOperations（transitive） | 5.6.2 |

共享 Project 另外五个 direct deps：ITensors 0.9.30、ITensorMPS 0.4.1、JSON 1.6.1、Plots 1.41.6、XDiag 0.5.0。`Project.toml` 只有 `[deps]`，没有 `[compat]`；准确版本由本地未提交的 Manifest 保存。

主要 package source 可读。depot 同时有 TensorKit 0.17.1 / MPSKit 0.13.13，不能把缓存中新版本当作本项目选定版本。本轮未 import，当前加载成功及端到端可运行性 **UNVERIFIED**。

### 9.2 实际运行依赖

- 根目录 `julia-env/Project.toml`、本地 Manifest：当前命令使用 `--project=julia-env`。
- 显式 import 的主要科研 packages：TensorKit、PEPSKit、MPSKit、Zygote、JLD2；ED 本身使用 Julia stdlib。
- `tracks/peps/results/` 路径布局：旧入口自动写入；`ad_staged_gd_continue.jl` / `slow_plaquette_diagnosis.jl` 依赖固定历史 run 路径；当前入口使用显式 checkpoint/outdir。
- 具体 package APIs，包括 `hook_pullback`、`_get_fixedspacetrunc`、`su_iter`，版本可移植性不能仅凭 package 名称保证。
- 未发现 Julia 源码 include 根目录 `scripts/`、读取 `.knowledge/` 或调用 `skills/` 的运行依赖。

### 9.3 工作流依赖

- 根目录 Makefile 的 `install-pepskit` / `install-mpskit` 使用 `Pkg.add`，不是冻结环境的重现脚本。
- 根目录 `make test` 仅测试 Python `scripts/tests/`，不覆盖本项目 Julia tests。
- `skills/using-pepskit`、`using-mpskit`、`setup-julia` 是设置/API 辅助，不是数值脚本运行必需。
- `AGENTS.md` 的 setup ratification、compute feasibility、opt-in verification、progress flush、artifact handoff 是协作规则，不是模型实现依赖。
- `PLAN.md` 保留 challenge scope、milestone gating/deadline，属于历史范围，不应自动当作永久科研限制。
- 源码/tests/文档引用 arXiv:0807.0487 Eq.8，但 `.knowledge/` 未找到该 ID 的本地条目/文件。series 内部实现一致；原论文公式及单位换算的独立 source audit **UNVERIFIED**。

## 10. 尚未验证的科研结论

以下均为 **UNVERIFIED**，不能由现有代码或测试声明替代：

- 任意非零场点已是可信 ground state；相变位置或对约 0.3285 benchmark 的数值确认。
- anyon condensation/confinement、e/m assignment、twist seam、endpoint form-factor selection。
- dense M2 state 已有 exact virtual-ℤ₂ symmetry，适合 sector-resolved production。
- A2 能解决收缩问题。
- “near-defective transfer operator”“factorizing fixed point”已被谱或秩数据确证；大步长/缺乏 virtual symmetry 是唯一根因。
- 当前 HEAD 所有 tests 通过，或可原样复现所有历史 CSV。

历史结果来自多个开发阶段；A2 已改 line-search objective，当前 pilot 不再含 0.10，旧 CSV 缺少后来添加字段。缺少逐 run 绑定 source commit，精确执行 provenance 不完整。

## 11. 私有仓库迁移建议

### 11.1 建议迁移

核心源码与当前入口：

- `P/scripts/tc_peps.jl`
- `P/scripts/ad_tied_core.jl`
- `P/scripts/ad_tied_gd.jl`
- `P/scripts/ed_checks.jl`
- `P/scripts/m3_series_validation.jl`
- `P/tests/runtests.jl`
- `P/tests/tied_ad_core_tests.jl`
- `P/tests/m3_series_tests.jl`
- `P/inspection/series_branch_divergence.jl`

证据与环境：

- 当前 `julia-env/Project.toml` 与本地 Manifest，先保留原环境快照。
- M2 steps001–086 的主要序列与 CSV/log，尤其 step086。
- 原始 series pilot / A1 的完整目录和顶层 logs。
- χ6 repair-point audit/summary/checkpoints；ED CSV；exact-smoke CSV。
- `M2_REPORT.md`、`M3_REPORT.md`、`PROGRESS.md`、`PLAN.md`、本审计文件。
- `figures/m2_energy_convergence.svg`；`CHALLENGE_SUMMARY.md` 作为历史 provenance。

应标为“有已知问题的科研基线”，不是 verified production。原始 pilot 的失败 tensors 与逐步 divergence 数据是后续诊断/回归的重要资产。

### 11.2 建议主要留在旧仓库作为历史

- 五个空 SU run dirs，以及只有部分初始化输出的 early smoke/probe runs。
- `groundstate_h0.jl` 旧 pipeline，尤其 optional spectrum 部分。
- `ad_ungraded_m2.jl`、`ad_staged_gd.jl`、`ad_staged_gd_continue.jl`、`ad_symmetrized_gd.jl`。
- `m3_hz_continuation.jl` / `m3_hz_tests.jl` 的旧阶段工作流；其中 physical-bound gates 仍有参考价值。
- 大多数一次性 SU/AD inspection probes、旧粗扫描、`m3_mz_vs_hz_invalid.svg`。
- challenge 注册 README、暑期学校 milestone/提交流程。
- harness 全套 Ion/skills/Zulip、其他方法环境与根目录工具链。

若长期研究需要保存旧失败实验，可作为 archive，而非新 repo 默认入口。本轮只记录建议，没有移动、删除、重构或修复这些文件。

**交接重点：保留 M2 零场 checkpoint 与证据、共用 Hamiltonian/exact tensor 代码和 M3 分支失败数据；A2、有限场物理结果和 anyon-sector 工作继续保持 UNVERIFIED。**

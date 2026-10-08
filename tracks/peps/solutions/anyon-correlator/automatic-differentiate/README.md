# Independent automatic-differentiation experiment

**Validated small-field AD approximations:** D=2 unrolled AD passes the three points
`(0.1,0)`, `(0,0.1)`, `(0.1,0.1)` with the declared energy/magnetization and
contraction accuracy. A fresh analytic-anchor suite also passes all four points.
No virtual-Z2 tensor constraint is imposed. Gaussian global search and generic
nonsymmetric-start robustness remain unresolved; the failed experiments are retained.
See [REPORT.md](REPORT.md) for the precise evidence and limits. This directory
contains a new real V–λ–P optimizer and its derivative/physical audit tools, developed
with OpenCode / `openai/gpt-6.1-sol`. See [DESIGN.md](DESIGN.md) and [PLAN.md](PLAN.md).

H=−ΣA−ΣB−hₓΣX−h_zΣZ, J=1, Pauli convention; infinite square lattice, 2×2
vertex repeat, physical d=2 on edges. The independent mode has four rank-4 V tensors,
eight rank-3 P tensors and sixteen positive diagonal λ connections at fixed D.
The user-approved uniform mode shares one V, P_E, P_N and four directional weights
across the repeat (40 parameters at D=2, rather than 160), preserving the geometry.
Blocking is exact and only provides the PEPS measurement/differentiation interface.

## Implementation

- `network.jl`: normalized real parameterization, balanced λ, exact/random starts,
  portable loop-state loading, independently constructed Hamiltonian/observables.
- `objective.jl`: fresh CTMRG contraction, two-stage AD chain rule, finite-difference
  probe, physical bounds and multi-seed/χ acceptance.
- `optimize.jl`: safeguarded L-BFGS/Armijo, independent trial veto, incremental
  trajectories and checkpoints; explicit stationarity/budget/failure status.
- `run.jl`: CLI and versions/seed/source-hash provenance.
- `tests.jl`: weighted-wavefunction reconstruction, parameter/TensorMap AD chain,
  independent Pauli action, quadratic optimizer and false-success regressions.
- `gradient_tests.jl`, `failure_control_test.jl`: reject failed adjoint solves even
  with correct observables; bind validation to method and unroll count.
- `diagnose.jl`, `repair.jl`: frozen-state unrolled-AD diagnostics and continuation
  of the three actual finite-field candidates.
- `boundary_check.jl`, `dominance.jl`: independent boundary measurements and
  norm-dominance diagnostics, without physical optimization.
- `perturb.jl`, `noisy_recover.jl`: explicitly nonsymmetric start/recovery experiments.
- `frozen_states/`: portable accepted states; `export_states.jl` checks JSON round trips.
- `analyze.py`: collected-data summary and plots, no additional physics compute.

The main AD path (`run.jl`, `network.jl`, `objective.jl`, `optimize.jl`) imports only
`finite-field-loop-update/reference.jl` from sibling scientific code, for published-series
coefficients and analytic derivatives. The separately labelled `boundary_check.jl`
diagnostic also imports `core.jl` and `vumps_core.jl` from that sibling directory to
adapt the frozen split state and solve/measure its boundaries. It never calls SU/ALS
physical updates; no old M2/M3 optimizer is used. Existing loop-update snapshots are
loaded only when explicitly requested as separately labelled controls.

## Running

Recommended fresh reproduction (no historical checkpoint dependency):

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/run.jl suite --out=tracks/peps/solutions/anyon-correlator/automatic-differentiate/results/reproduce-unrolled --uniform=true --exact=true --gradient=unrolled --unroll=20 --ctm-tol=1e-11 --ctm-maxiter=1000 --chi=8 --chi-check=16 --seed=19 --maxsteps=400 --budget=3600
```

The matching recorded run is `results/fresh-unrolled-suite01/` (exit 0, 22m19s).
The continuation cohort with final directional derivative audits is listed in
`selected_fields.json`; its benchmark figure is `figures/finite_field_benchmark.svg`.

Use the existing root `julia-env`; no package update is required. Run from the
repository root with personal startup files disabled. First CTMRG/AD compilation
can exceed short budgets; `session` reuses compiled code in one process. The user
now permits multi-hour local runs with progress, checkpoints and memory monitoring.
Obsolete cluster profiles must not be used. Do not treat cold-compilation timeouts
as physical failure.

Short implementation checks:

```bash
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/tests.jl
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/gradient_tests.jl
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/boundary_adapter_test.jl
OPENBLAS_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/failure_control_test.jl
```

Bounded zero-field diagnostic (D=2, χ=4; not a substitute for the finite-field goal):

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/run.jl session --out=tracks/peps/solutions/anyon-correlator/automatic-differentiate/results/new-zero --hx=0 --hz=0 --chi=4 --chi-check=8 --seed=19 --maxsteps=200 --budget=300
```

`--budget` bounds the optimizer loop after the first derivative probe, not total
process time or Julia compilation. Use an external `timeout` for a hard wall cap.
Output directories must be empty. Failed and interrupted runs are retained.

Approved uniform suite: zero field and all three weak fields, each starting from
the same random parameters, with compilation reused in one process:

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/run.jl suite --out=tracks/peps/solutions/anyon-correlator/automatic-differentiate/results/new-uniform-suite --uniform=true --chi=8 --chi-check=16 --seed=19 --maxsteps=400 --budget=120
```

The recorded suite uses an external 25-minute cap. A nonzero exit code means
at least one point was not accepted. See each point's `summary.json` for the reason.

Separately labelled analytic zero-field anchor controls (not random starts):

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/run.jl suite --out=tracks/peps/solutions/anyon-correlator/automatic-differentiate/results/new-anchor-suite --uniform=true --exact=true --chi=8 --chi-check=16 --seed=19 --maxsteps=400 --budget=240
```

This changes only initialization: the state begins at the analytic h=0 parameters,
and AD must optimize the finite-field Hamiltonian. The h=0 state itself is never
reported as the finite-field answer.

Current reliable-gradient workflow (unrolled N=20, final N=40 comparison):

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/repair.jl tracks/peps/solutions/anyon-correlator/automatic-differentiate/results/new-unrolled-repair 3600
```

This reads `analytic-anchor-suite/{z,xz,x}/final.jld2` and requires those checkpoints.
For a fresh analytic-anchor suite use `run.jl suite --uniform=true --exact=true`
with `--gradient=unrolled --unroll=20 --ctm-tol=1e-11 --ctm-maxiter=1000 --budget=3600`.
The fresh suite includes initial finite-graph FD and final N=20/40 checks; the selected
continuation states additionally have final N=40 finite-graph FD, N=40/80 comparison
and physical-direction fresh FD. Fixed-point AD remains available and now explicitly
rejects logged adjoint nonconvergence.

Finite-field random-start probe + optimization (run after forecasting local cost):

```bash
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --startup-file=no -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/automatic-differentiate/run.jl session --out=tracks/peps/solutions/anyon-correlator/automatic-differentiate/results/new-z --hx=0 --hz=0.1 --chi=8 --chi-check=16 --seed=19 --maxsteps=200 --budget=300
```

Replace the field by `(0.1,0)` or `(0.1,0.1)` for the other approved points.
For a labelled control, add
`--snapshot=tracks/peps/solutions/anyon-correlator/finite-field-loop-update/frozen_states/d2-xz.json`
at `(0.1,0.1)`. This does not count as random-start success.

Analyze completed artifacts:

```bash
.venv/bin/python tracks/peps/solutions/anyon-correlator/automatic-differentiate/analyze.py
```

## Interpretation

The first-stage targets are energy within 1e-5/spin and nonzero magnetization within
1% of the low-field series, plus optimizer stationarity and physical/contraction
checks. The series is not a rigorous oracle at arbitrary field. Agreement of multiple
environment seeds is not a full transfer-spectrum dominance proof. A small residual,
an iteration budget ending, or a lower energy from a different contraction branch
does not establish a ground state.

`results/` is local/generated and gitignored. Settings and source hashes are saved
per run. No code or data outside this experiment is overwritten. The user has
explicitly excluded the obsolete SCNet profile from this experiment; local-only
work remains bounded and larger-resource decisions must be revisited with the user.

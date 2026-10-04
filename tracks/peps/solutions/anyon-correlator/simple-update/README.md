# Independent D=2, d=2 toric-code update audit

**Result:** The old random-start SU stall reproduces. A new split-network path-SU also fails on the tested trajectory. Keeping the plaquette loop in a local **loop/cluster update** reaches the ground state from the same random tensors: **e = −0.99999999991334**, max stabilizer error **4.43×10⁻¹⁰**, χ=16.

This is not evidence that ordinary path-SU converges, nor proof of a bug in the old implementation. The algorithm distinction matters. See **[REPORT.md](REPORT.md)** for the conclusion and limitations.

The update state always contains separate rank-4 V tensors, rank-3 P tensors (one d=2 physical leg), and explicit D=2 λ on every V–P connection. The only physical control is the exact state. Physical legs are fused to d=4 **only for CTMRG measurement**, without approximation.

## Reproduce

From the repository root, with the existing `julia-env` (Julia 1.12.6, PEPSKit 0.8.0, TensorKit 0.16.5):

```bash
OPENBLAS_NUM_THREADS=1 julia -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/reproduce.jl
```

This reruns the recorded cases and overwrites their generated files under `results/`. First-call Julia compilation dominates the runtime; the updates themselves take seconds locally.

To run only the successful random-start calculation plus exact control:

```bash
OPENBLAS_NUM_THREADS=1 julia -O1 --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/run.jl --init=both --plaquette=loop --seed=1 --schedule=0.05:200,0.01:200,0.002:200 --measure-every=40 --out=tracks/peps/solutions/anyon-correlator/simple-update/results/loop-seed1
```

Implementation checks and plot generation:

```bash
julia --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/tests.jl
.venv/bin/python tracks/peps/solutions/anyon-correlator/simple-update/plot.py
```

## Files

| File | Role |
|---|---|
| `core.jl` | Independent Γ–λ storage, star HOSVD, plaquette path-SU, measurement blocking |
| `loop.jl` | Closed-loop ALS compression with diagonal external λ and internal cavity-message gauge |
| `measure.jl` | Independent operator construction and infinite-lattice CTMRG measurements |
| `run.jl` | Random/exact runs, incremental CSV and binary checkpoints |
| `final_audit.jl` | χ=8/16 checks, exact-control contraction diagnosis, original M2 replay |
| `reproduce.jl` | Consolidated single-process reproduction |
| `tests.jl` | Compact gate, weight, reconstruction and loop-fit checks |
| `results/final-audit/audit.toml` | Full final stabilizers and verification results |
| `figures/convergence.svg` | Recorded random-start trajectories |

No old M2 numerical code is imported by the new update or measurement implementation. `final_audit.jl` imports the old implementation in a separate module **only to replay it**.

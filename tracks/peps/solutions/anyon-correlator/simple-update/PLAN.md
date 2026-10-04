# Split V–λ–P simple-update audit

**Goal:** Independently test full-circuit random-start SU for the zero-field toric code, retaining rank-4 vertex tensors, rank-3 edge tensors with d=2, and explicit D=2 diagonal weights.

**Approved scope:** Random initialization is the main calculation; the exact ground state is the sole physical control. No product-state/staged-SU campaigns. H = −Σ A −Σ B, Pauli X stars and Z plaquettes, infinite square lattice, 2×2 vertex cell (4 V, 8 P, 16 λ). No imposed virtual symmetry or selected global topological sector. Acceptance: every stabilizer within 1e-6 of +1 and e within 1e-6 of −1, from converged infinite-network contraction.

## Implementation

1. `core.jl`: independent Julia/LinearAlgebra implementation. Store Γ tensors and one λ per V–P connection (state contains each λ **once**). All external cluster bonds are weighted with λ when constructing the local SU metric; remove those weights after decomposition.
   - Star: contract V and its four P neighbors, apply cosh(dt) I + sinh(dt) X⊗4, HOSVD the four (physical, external virtual) groups to D=2, recover V, P, and λ.
   - Plaquette: seven-tensor path P–V–P–V–P–V–P, omitting one corner V. The four physical P tensors carry cosh(dt) I + sinh(dt) Z⊗4. Contract the path, TT-SVD to D=2, recanonicalize the resulting MPS, extract Γ and λ, and remove external weights. Rotate the omitted corner each sweep. This is explicitly a **path SU**, not a loop/full-environment update; diagonal external weights neglect the connection through the omitted corner.
   - Singular-value normalization changes only overall state scale. Pseudoinverse cutoff 1e-14 relative to the largest λ, reported with settings.
2. `tests.jl`: compact algebra checks for four-spin gates and weighted path reconstruction, including nonuniform λ. These are implementation checks, not additional physical control runs. Exact-state preservation is assessed with physical observables, not equality of gauge-dependent tensors.
3. `measure.jl`: independently block V with outgoing E/N projections for measurement only, absorbing each λ exactly once; build stabilizers independently and contract using installed PEPSKit 0.8.0. Fresh CTMRG per measurement. Record all eight stabilizers, energy/spin, residual and iterations. Do not call the old `tc_peps.jl`.
4. `run.jl`: seeded random and exact modes, explicit dt schedule, full star/plaquette sweep, incremental CSV + checkpoints and flushed progress. Time a short local pilot before scheduling production. Start χ=8; use χ=16 only to resolve or validate the final contraction.
5. `REPORT.md` and convergence figure: evidence, commands, cost, numerical values, implementation comparison, and scope of conclusions. A different-layout success alone does not prove an old-code bug. Failure of finitely many SU trajectories does not prove impossibility.

## Resource estimate

Star cluster: 4⁴=256 scalar entries. Plaquette path: 4×4×2×4×2×4×4=4096 entries; SVD matrices are small after sequential truncation. Float64 core, no AD. CTMRG uses complex double-layer D²=4 and χ=8 initially. Expected local pilot under a few minutes and memory below 2 GB excluding Julia compilation. Time actual SU and CTMRG separately; use cluster for any subsequent job estimated above 10 minutes/16 GB.

## Execution

Run from repository root with `julia --project=julia-env tracks/peps/solutions/anyon-correlator/simple-update/tests.jl` and `.../run.jl`. All new scripts and outputs remain under this directory. Existing M2 artifacts are the comparison record. No commits requested.

## Evidence-driven amendment and completion

The path pilot preserved the exact state but did not find the ground state from random initialization. The 600-sweep continuation also failed the physical target; it was still changing and is **not** asserted to be a converged fixed point. The exact control had nonzero plaquette path discarded weight despite physical-state preservation, motivating a single algorithmic comparison:

- `loop.jl` retains all eight plaquette tensors and minimizes the full local weighted-cluster residual by ALS (12 sweeps maximum), with the same diagonal external λ environment. Internal cavity messages choose a gauge and new λ; they are not a global CTMRG environment. This is explicitly **loop/cluster update**, not a relabeling of ordinary path-SU.
- Same random initialization and same star update: the loop method reaches the target at step 120. χ=8/16 remeasurement and the sole exact-state control are recorded in `results/final-audit/audit.toml`.
- The exact control encountered a failed χ=8 CTMRG branch. Only this failure triggered an alternate environment seed and an exact small-torus contraction; all outcomes, including the failed contraction, are retained.
- Original M2 SU was replayed with its original seeded initialization and reproduces the reported non-ground fixed point. No concrete old-code bug was established.
- `reproduce.jl` ran end-to-end successfully; 11 targeted algebra checks passed. `REPORT.md` states what is established and what remains unproven.

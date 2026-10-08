# M2 single-step audit implementation plan

**Goal:** Locate the physical effect of each original SU update, rather than infer a cause from tensor parameterization.

**Architecture:** Reuse the original Hamiltonian/circuit and PEPSKit updates; independently contract the eight-spin periodic wavefunction and apply bit-level Pauli gates. Compare finite periodic and infinite CTMRG observations explicitly.

**Constraints:** H=-sum A-sum B, zero field, D=2, seed=1, dt=0.05, 2x2 cell. No old files changed, no commits or pushes. Local pilot capped at 600 seconds. No cluster access.

1. `audit.jl`: reproduce 175 full sweeps, retaining both state and weights. Exact periodic contraction uses eight virtual bonds and 256 amplitudes. Test exact-state stabilizers and the analytic single-gate response before auditing.
2. For each of eight original gates, save pre-update expectations, exact single-gate target expectations, actual SU expectations, wavefunction infidelity and library error. Finite periodic gate placement must use the original incidence and distinct site indices.
3. Branch from the identical stalled state: original sweep, plaquette-only sweep, identity updates, and selected isolated star/plaquette updates. Measure key states with fresh chi=8/16 CTMRG; record convergence rather than treating residual alone as proof.
4. Save incremental TOML and serialized checkpoint. Derive conclusions only from actual gate-resolved data. If necessary, extend with single-bond/no-truncation reconstruction controls after the pilot.
5. `REPORT.md`: formulas, observations, boundary limitations, remaining hypotheses, and reproduction command.

Analytic oracle: b_after=(b+tanh(2dt))/(1+b*tanh(2dt)); for any normalized state and one stabilizer B with B^2=I. Identity gates must preserve the physical state up to numerical precision if compression/regauging is harmless.

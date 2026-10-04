# Run from repository root. One process amortizes PEPSKit's first-call compilation.
include("final_audit.jl")

root = joinpath(@__DIR__,"results")
main(["--init=both", "--plaquette=path", "--seed=1", "--schedule=0.05:40",
      "--measure-every=10", "--out=$(joinpath(root,"pilot"))", "--max-seconds=480"])
main(["--init=random", "--plaquette=path", "--seed=1",
      "--resume=$(joinpath(root,"pilot","random-seed1.jls"))",
      "--schedule=0.05:160,0.01:200,0.002:200", "--measure-every=40",
      "--out=$(joinpath(root,"path-seed1"))", "--max-seconds=480"])
main(["--init=both", "--plaquette=loop", "--als-sweeps=12", "--seed=1",
      "--schedule=0.05:200,0.01:200,0.002:200", "--measure-every=40",
      "--out=$(joinpath(root,"loop-seed1"))", "--max-seconds=480"])
main_audit()

record = TOML.parsefile(joinpath(root,"final-audit","audit.toml"))
@assert record["loop_random"]["chi16"]["converged"]
@assert record["loop_random"]["chi16"]["maxerr"] < 1e-6
@assert record["loop_exact"]["chi16"]["maxerr"] < 1e-6
@assert abs(record["legacy_replay"]["chi16"]["energy_per_spin"] + 0.7533036457687) < 1e-6
println("Reproduction complete: loop/cluster update reached the ground state; legacy stall reproduced.")
flush(stdout)

# Standalone tests for the SU stagnation study; separate from ground-state tests.
using LinearAlgebra
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "dynamics", "tests.jl"))

println("SU_STAGNATION_TESTS_COMPLETE"); flush(stdout)

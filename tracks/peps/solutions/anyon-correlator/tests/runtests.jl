# Ground-state algorithm tests, in separate namespaces to avoid name collisions.
using LinearAlgebra
BLAS.set_num_threads(1)
const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

for (name, file) in [
    (:ModelChecks, "model/tests.jl"),
    (:MergedADChecks, "ad/merged/tests.jl"),
    (:SplitSUChecks, "simple-update/split/tests.jl"),
    (:LoopChecks, "simple-update/loop/tests.jl"),
    (:StrictOneSiteChecks, "simple-update/one-site-su/strict_tests.jl"),
    (:Z2OneSiteChecks, "simple-update/one-site-z2-su/tests.jl"),
    (:TransferNetworkChecks, "transfer-spectrum/tests.jl"),
    (:BoundaryChecks, "simple-update/loop/vumps_tests.jl"),
    (:SplitADChecks, "ad/split/tests.jl"),
    (:GradientChecks, "ad/split/gradient_tests.jl"),
    (:AdapterChecks, "ad/split/boundary_adapter_test.jl"),
    (:FailureChecks, "ad/split/failure_control_test.jl"),
    (:SavedStateChecks, "tests/state_loading_tests.jl"),
]
    println("SUITE $name $file"); flush(stdout)
    suite = Module(name)
    Core.eval(suite, :(include(path::AbstractString) = Base.include(@__MODULE__, path)))
    Base.include(suite, joinpath(PROJECT_ROOT, file))
end
println("ALL_RETAINED_TESTS_COMPLETE"); flush(stdout)

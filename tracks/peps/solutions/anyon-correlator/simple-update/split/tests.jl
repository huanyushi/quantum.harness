using Test, LinearAlgebra, Random
include("core.jl")
using .SplitSU

# Catches Pauli basis/order/sign errors independently of decomposition.
@testset "four-spin imaginary-time gate" begin
    x = zeros(2, 2, 2, 2); x[1, 1, 1, 1] = 1
    y = pauli_evolve(x, [1, 2, 3, 4], :X, 0.17)
    want = cosh(0.17) .* x
    want[2, 2, 2, 2] = sinh(0.17)
    @test y ≈ want atol=1e-14
    @test pauli_evolve(x, [1, 2, 3, 4], :Z, 0.17) ≈ exp(0.17) .* x
end

# Catches incorrect ring closure or gauge transformations that alter the state.
@testset "closed-loop metric and gauge" begin
    st = initial_state(:random,91)
    gs,meta = ring_cluster(st,1,1)
    x = ring_tensor(gs)
    restore_ring!(st,1,1,gs,meta)
    hs,_ = ring_cluster(st,1,1)
    y = ring_tensor(hs)
    @test abs(dot(vec(x),vec(y)))/(norm(x)*norm(y)) ≈ 1 atol=1e-11
    exact = initial_state(:exact,1)
    es,_ = ring_cluster(exact,1,1)
    z = ring_tensor(es)
    @test pauli_evolve(z,[2,4,6,8],:Z,0.17) ≈ exp(0.17)*z atol=1e-14
    target = pauli_evolve(x,[2,4,6,8],:Z,0.05)
    initial_error = norm(x/norm(x)-target/norm(target))^2
    fit_error = SplitSU.fit_ring!(gs,target;maxiter=4)
    @test fit_error < initial_error
end

# Catches missing/doubled internal λ and inverse/permutation mistakes.
@testset "weighted path reconstruction" begin
    st = initial_state(:random, 123)
    for (i, key) in enumerate(sort(collect(keys(st.weights))))
        st.weights[key] = [1.0, 0.2 + 0.02i]
    end
    for corner in 0:3
        path = plaquette_path(1, 1, corner)
        before, meta = path_cluster(st, path)
        restore_path!(st, path, before, meta; maxdim=2)
        after, _ = path_cluster(st, path)
        @test abs(dot(vec(before), vec(after))) / (norm(before)*norm(after)) ≈ 1 atol=1e-12
    end
end

# Catches an incorrect Schmidt spectrum/Γ convention in the multi-bond split.
@testset "TT split without truncation" begin
    Random.seed!(42)
    x = randn(2, 3, 2, 2)
    gs, ls, err = split_chain(x; maxdim=20)
    y = join_chain(gs, ls)
    @test y ≈ x / norm(x) atol=1e-12
    @test err < 1e-25
end

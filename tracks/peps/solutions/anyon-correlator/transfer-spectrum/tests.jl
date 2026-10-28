using Test, LinearAlgebra, TensorKit
BLAS.set_num_threads(1)
include("core.jl")
using .ToricTransferSpectrum
const S=ToricTransferSpectrum.OneSiteZ2SU
group=isempty(ARGS) ? "all" : only(ARGS)
group in ("all","--group=network","--group=boundary","--group=spectrum","--group=output") || error("unknown group $group")
if group in ("all","--group=network")
@testset "test_composite_lambda_counting" begin
    st=S.random_state(5); a=S.dense_arrays(st)
    st=S.state_from_arrays(a.V,a.P,[0.8,0.6]); a=S.dense_arrays(st)
    net=build_transfer(st); expected=zeros(4,2,2,2,2)
    for p in 0:1,q in 0:1,n in 1:2,e in 1:2,s in 1:2,w in 1:2,b in 1:2,c in 1:2
        expected[p+2q+1,n,e,s,w]+=a.V[c,b,s,w]*a.P[p+1,b,e]*a.P[q+1,c,n]*a.lambda[b]*a.lambda[e]*a.lambda[c]*a.lambda[n]
    end
    @test net.A ≈ expected atol=1e-12
    @test net.step == 1
    @test length(net.mpo) == 1
end
@testset "test_double_layer_direct_sum" begin
    net=build_transfer(S.random_state(5)); raw=zeros(4,4,4,4)
    for n in 1:2,N in 1:2,e in 1:2,E in 1:2,s in 1:2,S0 in 1:2,w in 1:2,W0 in 1:2
        raw[n+2(N-1),e+2(E-1),s+2(S0-1),w+2(W0-1)]=sum(net.A[p,n,e,s,w]*conj(net.A[p,N,E,S0,W0]) for p in 1:4)
    end
    @test net.Q_raw ≈ raw atol=1e-12
end
@testset "test_fused_symmetry_actions" begin
    net=build_transfer(S.random_state(5)); z=net.ket_action; b=net.bra_action
    @test z*z ≈ id(space(z,1)) atol=1e-12
    @test b*b ≈ id(space(b,1)) atol=1e-12
    @test norm(z*b-b*z) < 1e-12
    @test dim(space(z,1),Z2Irrep(0)) == 2
    @test dim(space(z,1),Z2Irrep(1)) == 2
    for a in (z,b)
        q=copy(net.Q)
        for i in 1:4; q=mode_array(q,reshape(convert(Array,a),4,4),i); end
        @test norm(q-net.Q)/norm(net.Q) < 1e-12
    end
    t=closed_row(net,2); g=kron(convert(Array,z),convert(Array,z)); h=kron(convert(Array,b),convert(Array,b))
    @test norm(t*g-g*t)/norm(t) < 1e-12
    @test norm(t*h-h*t)/norm(t) < 1e-12
end
@testset "test_transfer_adjoint" begin
    net=build_transfer(S.random_state(5)); t=closed_row(net,2)
    x=ComplexF64.(1:16); y=ComplexF64.(16:-1:1).+im
    @test dot(y,t*x) ≈ dot(t'*y,x) atol=1e-12
end
@testset "test_exact_reference_fixedpoint_structure" begin
    net=build_transfer(S.exact_reference_state()); t=closed_row(net,2)
    ev=eigvals(t); scale=maximum(abs,ev)
    @test count(x->abs(x/scale-1)<1e-12,ev) == 2
    @test count(x->abs(x/scale)<1e-12,ev) == 14
    @test norm((t/scale)^2-t/scale) < 1e-12
end
end
if group in ("all","--group=boundary")
    include("boundary_tests.jl")
end
if group in ("all","--group=spectrum")
    include("spectrum_tests.jl")
end
if group in ("all","--group=output")
    include("output_tests.jl")
end

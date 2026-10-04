using Test, LinearAlgebra, Random, TensorKit
BLAS.set_num_threads(1)
include("core.jl")
using .FieldLoop
include("vumps_core.jl")
using .BoundaryAudit

@testset "two-row column map and adjoint" begin
    @test isdefined(BoundaryAudit,:column_apply)
    if isdefined(BoundaryAudit,:column_apply)
        Random.seed!(17)
        q,chi=2,2
        a=randn(ComplexF64,chi,q,chi);b=randn(ComplexF64,chi,q,chi)
        u=randn(ComplexF64,q,q,q,q);v=randn(ComplexF64,q,q,q,q)
        x=randn(ComplexF64,chi,q,q,chi);y=randn(ComplexF64,chi,q,q,chi)
        got=column_apply(x,a,b,u,v);want=zeros(ComplexF64,size(x))
        for i in 1:chi,j in 1:chi,k in 1:q,l in 1:q,
            I in 1:chi,J in 1:chi,K in 1:q,L in 1:q,n in 1:q,t in 1:q,s in 1:q
            want[j,k,l,J]+=x[i,K,L,I]*a[i,n,j]*conj(b[I,s,J])*u[n,k,t,K]*v[t,l,s,L]
        end
        @test got ≈ want atol=1e-11
        @test dot(y,got) ≈ dot(column_adjoint(y,a,b,u,v),x) atol=1e-11
    end
end

@testset "measurement bond-gauge preserves a periodic wavefunction" begin
    st=initial_state(:random,7;D=3)
    for (i,k) in enumerate(sort(collect(keys(st.weights))))
        st.weights[k] .= [1.0,0.4+0.01i,0.02+0.001i]
    end
    function torus_wave(aa)
        a,b,c,d=aa[1,1],aa[1,2],aa[2,1],aa[2,2]
        @tensor psi[p,q,r,s] := a[p,i,j,k,l]*b[q,m,l,n,j]*c[r,k,o,i,t]*d[s,n,t,m,o]
        return psi/norm(psi)
    end
    raw,_,_=prepare_layers(st)
    balanced,_,_=prepare_layers(st;balanced=true)
    @test torus_wave(raw) ≈ torus_wave(balanced) atol=1e-12
end

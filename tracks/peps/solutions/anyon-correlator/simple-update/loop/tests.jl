using Test, LinearAlgebra, Random
BLAS.set_num_threads(1)
include("core.jl")
using .FieldLoop

@testset "field gate sign, axis and normalization" begin
    @test isdefined(FieldLoop,:field_gate)
    if isdefined(FieldLoop,:field_gate)
        x,z = [0.0 1;1 0],[1.0 0;0 -1]
        for (hx,hz) in ((0.0,0.1),(0.1,0.0),(0.07,0.13),(0.0,0.0))
            @test field_gate(hx,hz,0.17) ≈ exp(0.17*(hx*x+hz*z)) atol=1e-14
        end
        st=initial_state(:random,91;D=3)
        p=copy(st.tensors[(:E,1,1)])
        g=exp(0.17*(0.07x+0.13z))
        expected=reshape(g*reshape(p,2,:),size(p))
        field_step!(st,0.07,0.13,0.17)
        @test st.tensors[(:E,1,1)] ≈ expected/norm(expected) atol=1e-14
    end
end

@testset "QR-compressed ALS is the same local least-squares problem" begin
    @test isdefined(FieldLoop,:environment_qr)
    if isdefined(FieldLoop,:environment_qr)
        for D in (2,3)
            st=initial_state(:random,55;D);gs,_=ring_cluster(st,1,1)
            full=FieldLoop.open_chain(gs)
            explicit=reshape(sum(full[i,:,i] for i in axes(full,1)),(size(g,2) for g in gs)...)
            @test ring_tensor(gs) ≈ explicit atol=1e-14
            old=deepcopy(gs);transformed=deepcopy(gs)
            for i in (2,4,6,8)
                transformed[i]=FieldLoop.mode_product(transformed[i],[1.0 0;0 -1],2)
            end
            target=cosh(0.05)*ring_tensor(old)+sinh(0.05)*ring_tensor(transformed)
            dense=deepcopy(gs);compressed=deepcopy(gs)
            a=FieldLoop.fit_ring!(dense,target;maxiter=2)
            b=FieldLoop.fit_ring!(compressed,target;maxiter=2,target_cores=(old,transformed),coefficients=(cosh(0.05),sinh(0.05)))
            @test a.fit_after ≈ b.fit_after atol=1e-11
            @test norm(ring_tensor(dense)-ring_tensor(compressed)) < 1e-9
        end
    end
end

@testset "D-dependent closed-ring regauging" begin
    for D in (2,3)
        st=initial_state(:random,123;D)
        for w in values(st.weights)
            w .= collect(range(1.0,0.4;length=D))
        end
        gs,meta=ring_cluster(st,1,1)
        x=ring_tensor(gs)
        info=restore_ring!(st,1,1,gs,meta)
        hs,_=ring_cluster(st,1,1)
        y=ring_tensor(hs)
        @test norm(x/norm(x)-y/norm(y)) < 1e-10
        @test info.gauge_error < 1e-10
        @test isfinite(info.bp_residual)
        fit=loop_update!(st,1,1,0.05;als_sweeps=4)
        @test fit.fit_after <= fit.fit_before+1e-12
        @test 1 <= fit.als_iterations <= 4
    end
end

@testset "published series and physical derivatives" begin
    r=reference_values(0.0,0.1)
    @test r.energy ≈ -1.0025240347293854 atol=1e-12
    @test r.mx == 0.0
    @test r.mz > 0.05
    for (x,z) in ((0.05,0.1),(0.1,0.1))
        r=reference_values(x,z); x,z=BigFloat(x),BigFloat(z); d=big"1e-12"
        @test r.mx ≈ -(reference_values(x+d,z).energy-reference_values(x-d,z).energy)/(2d) atol=2e-10
        @test r.mz ≈ -(reference_values(x,z+d).energy-reference_values(x,z-d).energy)/(2d) atol=2e-10
        dual=reference_values(z,x)
        @test r.energy ≈ dual.energy atol=1e-14
        @test r.mx ≈ dual.mz atol=1e-14
    end
end

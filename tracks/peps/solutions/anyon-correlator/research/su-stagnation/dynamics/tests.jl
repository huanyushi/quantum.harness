using Test, LinearAlgebra
isfile(joinpath(@__DIR__,"core.jl")) && include("core.jl")

@testset "controlled dual-plaquette family matches physical filters" begin
    @test isdefined(@__MODULE__,:filtered_plaquette_state)
    if isdefined(@__MODULE__,:filtered_plaquette_state)
        for K in (zeros(2,2),fill(0.25,2,2),[0.13 0.25;0.35 0.42])
            state=filtered_plaquette_state(K)
            actual=AC.wave(state)
            oracle=fill(ComplexF64(1/16),256)
            for r in 1:2,c in 1:2
                oracle=cosh(K[r,c])*oracle+sinh(K[r,c])*AC.action(oracle,:B,r,c)
            end
            oracle=normalize(oracle)
            @test abs2(dot(actual,oracle)) ≈ 1 atol=1e-12
            @test AC.vals(actual)[1:2:end] ≈ ones(4) atol=1e-12
            rho=[tanh(2K[r,c]) for r in 1:2 for c in 1:2]
            expected=[(rho[i]+prod(rho[j] for j in 1:4 if j!=i))/(1+prod(rho)) for i in 1:4]
            @test AC.vals(actual)[2:2:end] ≈ expected atol=1e-12
            shifted=copy(K);shifted[1,1]+=0.07
            target=normalize(cosh(0.07)*actual+sinh(0.07)*AC.action(actual,:B,1,1))
            @test abs2(dot(target,AC.wave(filtered_plaquette_state(shifted)))) ≈ 1 atol=1e-12
        end
    end
end

@testset "explicit SU gauge preserves the family and predicts star dynamics" begin
    @test isdefined(@__MODULE__,:canonical_filtered_state)
    if isdefined(@__MODULE__,:canonical_filtered_state)
        K=[0.21 0.37;0.58 0.51]
        psi,w=canonical_filtered_state(K)
        @test abs2(dot(AC.wave(psi),AC.wave(filtered_plaquette_state(K)))) ≈ 1 atol=1e-12
        after,_,_=AC.step!(deepcopy(psi),deepcopy(w),make_alg(psi),:A,1,1;dt=0.017)
        expected=copy(K);expected[2,2]=star_filter_map(K[2,2],0.017)
        @test fit_filter_state(AC.wave(after)).K ≈ expected atol=1e-12
        f=stationary_profile(0.05)
        @test f.b_low ≈ 0.46949459791665 atol=1e-13
        @test f.b_high ≈ 0.54371998516605 atol=1e-13
        @test stationary_profile(0.0).energy == -0.75
    end
end

@testset "analytic reduced map predicts held-out SU steps" begin
    @test isdefined(@__MODULE__,:star_filter_map)
    if isdefined(@__MODULE__,:star_filter_map)
        for (K,dt) in (([0.18 0.31;0.43 0.22],0.003),([0.21 0.37;0.58 0.51],0.017),
                        (fill(0.45,2,2),0.08))
            psi,w,_=identity_gauge(filtered_plaquette_state(K))
            after,_,_=AC.step!(deepcopy(psi),deepcopy(w),make_alg(psi),:A,1,1;dt)
            fitted=fit_filter_state(AC.wave(after))
            expected=copy(K);expected[2,2]=star_filter_map(K[2,2],dt)
            @test fitted.K ≈ expected atol=2e-12
            @test fitted.support ≈ 1 atol=1e-12
            @test fitted.residual < 1e-12
        end
        @test star_filter_map(0.0,0.05)==0
        @test star_filter_map(0.35,0.0) ≈ 0.35 atol=1e-14
    end
end

@testset "B-eigenbasis fit detects the prescribed K and non-family interactions" begin
    @test isdefined(@__MODULE__,:fit_filter_state)
    if isdefined(@__MODULE__,:fit_filter_state)
        K=[0.13 0.25;0.35 0.42]
        v=AC.wave(filtered_plaquette_state(K))
        fit=fit_filter_state(v)
        @test fit.valid
        @test fit.K ≈ K atol=1e-12
        @test fit.support ≈ 1 atol=1e-12
        @test fit.residual < 1e-12
        @test fit_filter_state(exp(0.4im)*v).K ≈ K atol=1e-12
        # A two-plaquette interaction is not a sum of the four single B terms.
        interacting=normalize(cosh(0.1)*v+sinh(0.1)*AC.action(AC.action(v,:B,1,1),:B,1,2))
        @test fit_filter_state(interacting).residual > 1e-3
    end
end

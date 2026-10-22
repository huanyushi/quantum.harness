using Test, Logging
include("network.jl")
include("objective.jl")

@testset "a finite vector from a failed adjoint solve must be rejected" begin
    @test isdefined(@__MODULE__, :checked_gradient_call)
    if isdefined(@__MODULE__, :checked_gradient_call)
        # Real logging, not a mocked solver: reproduces PEPSKit 0.8's API behavior
        # of returning a finite result after a nonconvergence warning.
        failed() = begin
            @warn "gradient fixed-point iteration reached maximal number of iterations:"
            (-1.0025,[4.93,0.0])
        end
        @test_throws GradientFailure checked_gradient_call(failed)
        good() = (-1.0025,[1e-6,0.0])
        value,messages = checked_gradient_call(good)
        @test value==(-1.0025,[1e-6,0.0])
        @test isempty(messages)
        fallback_ok() = begin
            @warn "Fixed-point gradient computation using Arnoldi failed: fallback converged"
            (-1.0025,[1e-6,0.0])
        end
        @test first(checked_gradient_call(fallback_ok))==(-1.0025,[1e-6,0.0])
    end
end

@testset "norm dominance is not minimum measured energy" begin
    rows = [(;normcell=1.0,energy=-1.1),(;normcell=1.01,energy=-1.0),
            (;normcell=1.01+1e-11,energy=-1.0)]
    selected = dominant_cluster(rows)
    @test selected.best in (2,3)
    @test selected.members==[2,3]
    @test_throws ContractionFailure dominant_cluster([(;normcell=NaN,energy=-10.0)])
end

@testset "generic perturbations actually break the virtual symmetry" begin
    x = initial_parameters(2,1;uniform=true,exact=true)
    @test virtual_symmetry_defect(x,2).maximum<1e-14
    y = perturb_parameters(x,2,71;amplitude=1e-3)
    @test length(y)==length(x)
    @test y==perturb_parameters(x,2,71;amplitude=1e-3)
    @test virtual_symmetry_defect(y,2).maximum>1e-5
    for inds in (1:16,17:24,25:32)
        @test norm(y[inds]-x[inds])/norm(x[inds]) ≈ 1e-3 atol=1e-14
    end
    st = split_parameters(y,2)
    @test all(w->all(>(0),w),st.weights)
end

@testset "derivative validation binds the method and finite graph" begin
    p = Dict("passed"=>true,"D"=>2,"chi"=>8,"hx"=>0.0,"hz"=>0.1,
             "parameter_count"=>40,"gradient_method"=>"unrolled","unroll_steps"=>20)
    @test probe_matches(p,2,8,0.0,0.1,40;method=:unrolled,unroll_steps=20)
    @test !probe_matches(p,2,8,0.0,0.1,40;method=:fixedpoint)
    @test !probe_matches(p,2,8,0.0,0.1,40;method=:unrolled,unroll_steps=40)
end

@testset "unrolled AD rejects undefined iteration graphs" begin
    @test_throws ArgumentError energy_gradient(zeros(40),2,nothing,nothing;method=:unrolled,unroll_steps=0)
    @test_throws ArgumentError energy_gradient(zeros(40),2,nothing,nothing;method=:unknown)
end

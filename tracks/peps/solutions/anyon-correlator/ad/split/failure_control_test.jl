using Test
include("run.jl")

@testset "correct observables do not excuse a failed derivative" begin
    mktempdir("/tmp/opencode") do out
        calls = Ref(0)
        failed_gradient(args...;kwargs...) = begin
            calls[]+=1
            @test kwargs[:maxiter]==600
            checked_gradient_call() do
                @warn "gradient fixed-point iteration reached maximal number of iterations:"
                (-1.0,ones(40))
            end
        end
        x = initial_parameters(2,1;exact=true,uniform=true)
        summary = optimize(x,2,0.0,0.0;chi=4,chi_check=8,maxsteps=2,budget=60,ctm_maxiter=600,
                           gradient_provider=failed_gradient,outdir=out)
        @test summary.status=="gradient_failed"
        @test !summary.audit.verdict.accepted
        @test calls[]==1
        @test summary.gradnorm===nothing
        saved = load(joinpath(out,"final.jld2"))
        @test saved["parameters"]==x
        @test saved["steps"]==0
    end
end

@testset "an inconclusive final audit still preserves an explicit failure summary" begin
    mktempdir("/tmp/opencode") do out
        x=initial_parameters(2,1;exact=true,uniform=true)
        failed_gradient(args...;kwargs...)=throw(GradientFailure("deliberate adjoint failure"))
        failed_audit(args...;kwargs...)=throw(ContractionFailure("deliberate dominant-branch audit failure"))
        summary=optimize(x,2,0.0,0.0;chi=4,chi_check=8,maxsteps=2,budget=60,
                          gradient_provider=failed_gradient,audit_provider=failed_audit,outdir=out)
        @test summary.status=="gradient_failed"
        @test !summary.audit.verdict.accepted
        @test "audit_failed" in summary.audit.verdict.reasons
        @test isfile(joinpath(out,"summary.json"))
    end
end

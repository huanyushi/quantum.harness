using Test
include("network.jl")
isfile(joinpath(@__DIR__,"boundary_check.jl")) && include("boundary_check.jl")

@testset "independent boundary adapter preserves the measured PEPS" begin
    @test isdefined(@__MODULE__, :as_boundary_state)
    if isdefined(@__MODULE__, :as_boundary_state)
        x = initial_parameters(2,19;uniform=true)
        x[end-7:end] .= range(-0.2,0.4;length=8)
        st = as_boundary_state(x,2)
        aa,_,_ = BoundaryAudit.prepare_layers(st;balanced=true)
        expected = blocked_arrays(x,2)
        for i in eachindex(aa)
            @test aa[i] ≈ expected[i]/norm(expected[i]) atol=1e-13
        end
    end
end

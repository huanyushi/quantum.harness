include("core.jl")
using Test
AC.BLAS.set_num_threads(1)
state=legacy_state(;force_replay="--replay" in ARGS)
before=AC.wave(state.psi);initial=fit_filter_state(before)
@test initial.support>1-1e-12 && initial.residual<1e-12
@test abs2(dot(before,AC.wave(filtered_plaquette_state(initial.K)))) ≈ 1 atol=1e-12
rows=Any[]
for dt in (0.05,0.02,0.01)
    p,_,_=AC.step!(deepcopy(state.psi),deepcopy(state.w),make_alg(state.psi),:A,1,1;dt)
    after=AC.wave(p)
    eta=initial.K[2,2]-star_filter_map(initial.K[2,2],dt)
    inverse=normalize(cosh(eta)*before-sinh(eta)*AC.action(before,:B,2,2))
    fidelity=abs2(dot(inverse,after))
    @test fidelity ≈ 1 atol=1e-12
    row=Dict("dt"=>dt,"effective_inverse_B22_step"=>eta,"inverse_B22_infidelity"=>max(0.,1-fidelity),
        "initial_K"=>[initial.K[r,c] for r in 1:2 for c in 1:2])
    push!(rows,row);println(row);flush(stdout)
end
out=joinpath(@__DIR__,"results","legacy-check");mkpath(out)
open(joinpath(out,"inverse.json"),"w") do io;JSON.print(io,rows,2);end

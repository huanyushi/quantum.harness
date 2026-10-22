include("core.jl")
using Test
AC.BLAS.set_num_threads(1)
out=joinpath(@__DIR__,"results","branch-check");mkpath(out)
rows=Any[]
for K in ([0.2 0.0001;0.0001 1.0],[0.2 0.0001;0.4 1.0],
          [0.2 0.3;0.0001 1.0],[0.2 0.3;0.4 1.0])
    dt=0.05
    psi,w=canonical_filtered_state(K)
    t=tanh(K[2,2]);M=[exp(dt) t*exp(-dt);t*exp(-dt) t^2*exp(dt)]
    sm=svdvals(M);ratio=sm[2]/sm[1]
    spectators=[tanh(K[1,2]),tanh(K[2,1])]
    retained=all(x->x>ratio,spectators)
    after,_,_=AC.step!(deepcopy(psi),deepcopy(w),make_alg(psi),:A,1,1;dt)
    f=fit_filter_state(AC.wave(after));expected=copy(K);expected[2,2]=star_filter_map(K[2,2],dt)
    row=Dict("K"=>[K[r,c] for r in 1:2 for c in 1:2],"dt"=>dt,
        "channel_ratio"=>ratio,"spectator_ratios"=>spectators,"retained_dominant_channel"=>retained,
        "fit_valid"=>f.valid,"fit_support"=>f.support,"fit_residual"=>f.residual,
        "fitted_K"=>[f.K[r,c] for r in 1:2 for c in 1:2],
        "single_map_error"=>maximum(abs,f.K-expected),"stabilizers"=>AC.vals(AC.wave(after)))
    piecewise=copy(K)
    spectators[1]<ratio && (piecewise[1,2]=0.0)
    spectators[2]<ratio && (piecewise[2,1]=0.0)
    any(x->x>ratio,spectators) && (piecewise[2,2]=star_filter_map(K[2,2],dt))
    row["piecewise_prediction"]=[piecewise[r,c] for r in 1:2 for c in 1:2]
    row["piecewise_error"]=maximum(abs,f.K-piecewise)
    @test row["piecewise_error"]<1e-11
    push!(rows,row)
    if retained
        @test row["single_map_error"]<1e-11
    else
        @test row["single_map_error"]>1e-5 || !f.valid || f.residual>1e-4
    end
    println(row);flush(stdout)
end
open(joinpath(out,"branches.json"),"w") do io;JSON.print(io,rows,2);end

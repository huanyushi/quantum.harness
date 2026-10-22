# Frozen-state follow-up to actual failed CTMRG branches at D=3 and D=4.
include("run.jl")
root=joinpath(@__DIR__,"results")
for (name,hx,hz,chis) in (("d3-xz",0.1,0.1,(8,12,16)),("d3-x",0.1,0.0,(8,12,16)),("d4-xz",0.1,0.1,(8,12)))
    st=deserialize(joinpath(root,name,"state.jls")).state
    rows=[]
    for chi in chis
        m=measure_state(st,hx,hz;chi,envseed=-1,maxiter=150)
        push!(rows,m)
        println("POSITIVE BOUNDARY $name ",(;chi,m.energy,m.mx,m.mz,m.ctm_converged,m.ctm_residual,m.ctm_seconds));flush(stdout)
        open(joinpath(root,name,"positive-boundary.json"),"w") do io;JSON.print(io,rows);end
    end
end

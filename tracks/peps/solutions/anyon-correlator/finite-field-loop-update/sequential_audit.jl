# Same frozen states/observables, alternative supported CTMRG growth ordering.
include("run.jl")
root=joinpath(@__DIR__,"results")
for (name,hx,hz) in (("d3-xz",0.1,0.1),("d3-x",0.1,0.0),("d4-xz",0.1,0.1))
    st=deserialize(joinpath(root,name,"state.jls")).state
    rows=[]
    for chi in (8,12)
        m=measure_state(st,hx,hz;chi,envseed=17,maxiter=150,alg=:SequentialCTMRG)
        push!(rows,m)
        println("SEQUENTIAL $name ",(;chi,m.energy,m.mx,m.mz,m.ctm_converged,m.ctm_residual,m.ctm_seconds));flush(stdout)
        open(joinpath(root,name,"sequential.json"),"w") do io;JSON.print(io,rows);end
    end
end

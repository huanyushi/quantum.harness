# CTMRG on the identical measurement-only balanced gauge used by VUMPS.
include("vumps_run.jl")
include("measure.jl")
root=joinpath(@__DIR__,"results")
for name in ("d3-xz","d4-xz")
    chk=deserialize(joinpath(root,name,"state.jls"));st=chk.state
    aa,_,logs=prepare_layers(st;balanced=true)
    p,v=ComplexSpace(4),ComplexSpace(st.D)
    psi=InfinitePEPS([TensorMap(ComplexF64.(reshape(a,4,st.D^4)),p,v⊗v⊗v'⊗v') for a in aa])
    rows=[]
    for chi in (8,12,16)
        m=measure_state(st,chk.hx,chk.hz;chi,envseed=17,maxiter=180,prepared=psi)
        push!(rows,m)
        println("BALANCED CTMRG $name ",(;chi,m.energy,m.mx,m.mz,m.ctm_converged,m.ctm_residual,m.ctm_seconds));flush(stdout)
        open(joinpath(root,name,"ctm-balanced.json"),"w") do io;JSON.print(io,(;balanced=true,blocking_log_scales=logs,measurements=rows));end
    end
end

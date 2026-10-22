# Exact finite-network identity check on the actual frozen checkpoints.
using LinearAlgebra, Serialization, TensorKit, JSON, SHA
BLAS.set_num_threads(1)
include("core.jl")
using .FieldLoop
include("vumps_core.jl")
using .BoundaryAudit
function torus_wave(aa)
    a,b,c,d=aa[1,1],aa[1,2],aa[2,1],aa[2,2]
    @tensor psi[p,q,r,s] := a[p,i,j,k,l]*b[q,m,l,n,j]*c[r,k,o,i,t]*d[s,n,t,m,o]
    return psi/norm(psi)
end
results=[]
for name in ("d3-xz","d4-xz")
    path=joinpath(@__DIR__,"results",name,"state.jls")
    st=deserialize(path).state
    raw,_,_=prepare_layers(st);balanced,_,_=prepare_layers(st;balanced=true)
    err=norm(torus_wave(raw)-torus_wave(balanced))
    minratio=minimum(minimum(sqrt.(w))/maximum(sqrt.(w)) for w in values(st.weights))
    @assert err<1e-10 && minratio>FieldLoop.PINV_RTOL
    r=(;state=name,checkpoint_sha256=bytes2hex(open(sha256,path)),wavefunction_error=err,
       minimum_sqrt_lambda_ratio=minratio,pseudoinverse_cutoff_active=false)
    push!(results,r);println(r);flush(stdout)
end
open(joinpath(@__DIR__,"vumps_gauge_audit.json"),"w") do io;JSON.print(io,results);end

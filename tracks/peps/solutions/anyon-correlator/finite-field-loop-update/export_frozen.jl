# Portable, compact copies of the physical checkpoints used by the VUMPS audit.
using Serialization, JSON, SHA
include("core.jl")
using .FieldLoop
out=joinpath(@__DIR__,"frozen_states");mkpath(out)
for name in ("d2-xz","d3-xz","d4-xz")
    path=joinpath(@__DIR__,"results",name,"state.jls");chk=deserialize(path);st=chk.state
    tensors=[Dict("key"=>[String(k[1]),k[2],k[3]],"shape"=>collect(size(st.tensors[k])),"data"=>vec(st.tensors[k])) for k in sort(collect(keys(st.tensors)))]
    weights=[Dict("key"=>collect(k),"data"=>st.weights[k]) for k in sort(collect(keys(st.weights)))]
    record=Dict("format"=>"FieldLoop split V-lambda-P, column-major Float64 arrays","state"=>name,
        "D"=>st.D,"L"=>st.L,"hx"=>chk.hx,"hz"=>chk.hz,"step"=>chk.step,"tau"=>chk.tau,
        "original_checkpoint_sha256"=>bytes2hex(open(sha256,path)),"tensors"=>tensors,"weights"=>weights)
    target=joinpath(out,name*".json")
    open(io->JSON.print(io,record),target,"w")
    reread=JSON.parsefile(target)
    for entry in reread["tensors"]
        k=entry["key"];node=(Symbol(k[1]),Int(k[2]),Int(k[3]))
        @assert reshape(Float64.(entry["data"]),Tuple(Int.(entry["shape"])))==st.tensors[node]
    end
    println("Portable checkpoint verified: ",target);flush(stdout)
end

# Cheap zero-field diagnostics only; final claims require the CTMRG runner.
include("strict.jl")
using .StrictOneSiteSU, LinearAlgebra, JSON, Serialization
BLAS.set_num_threads(1)
const S=StrictOneSiteSU
function pilot(args=ARGS)
    out=isempty(args) ? joinpath(@__DIR__,"results","strict-pilot") : args[1]
    ispath(out) && error("output already exists: $out")
    mkpath(out);started=time();deadline=started+100
    seeds=length(args)>1 ? parse.(Int,split(args[2],',')) : collect(1:6)
    summaries=[]
    for seed in seeds
        time()<deadline || break
        st=S.random_state(seed);step=0;success=false
        open(joinpath(out,"seed$(seed).jsonl"),"w") do io
            for i in 1:320
                diag=S.sweep!(st,0.05);step=i
                if i%20==0
                    r=S.local_residuals(st)
                    row=(;seed,step=i,dt=0.05,residuals=r,symmetry=S.defect(st),lambda=st.lambda,
                          diagnostics=diag,seconds=time()-started)
                    JSON.print(io,row);println(io);flush(io)
                    serialize(joinpath(out,"seed$(seed).jls"),(;state=st,seed,step))
                    println((;seed,step,residuals=r,symmetry=S.defect(st),seconds=time()-started));flush(stdout)
                    if max(r.star,r.plaquette)<1e-5;success=true;break;end
                end
                time()<deadline || break
            end
        end
        r=S.local_residuals(st)
        push!(summaries,(;seed,step,local_candidate=success,residuals=r,symmetry=S.defect(st)))
        open(joinpath(out,"summary.json"),"w") do io
            JSON.print(io,(;status=time()<deadline ? "running" : "budget",summaries,seconds=time()-started));println(io)
        end
        success && break
    end
    open(joinpath(out,"summary.json"),"w") do io
        JSON.print(io,(;status=time()<deadline ? "finished" : "budget",summaries,seconds=time()-started));println(io)
    end
end
if abspath(PROGRAM_FILE)==@__FILE__;pilot();end

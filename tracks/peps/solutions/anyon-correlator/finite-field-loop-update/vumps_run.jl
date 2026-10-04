using LinearAlgebra, Serialization, JSON, TOML, SHA, Dates
BLAS.set_num_threads(1)
include("core.jl")
using .FieldLoop
include("vumps_core.jl")
using .BoundaryAudit

function main(args=ARGS)
    opts=Dict("state"=>"exact","chi"=>"4","seed"=>"1","maxiter"=>"80","tol"=>"1e-8","balanced"=>"false","budget"=>"300","resume-boundaries"=>"","tag"=>"","portable"=>"false")
    for a in args
        k,v=split(replace(a,r"^--"=>""),'=';limit=2);haskey(opts,k)||error("Unknown option $k");opts[k]=v
    end
    name=opts["state"];chi=parse(Int,opts["chi"]);seed=parse(Int,opts["seed"]);balanced=parse(Bool,opts["balanced"])
    if name=="exact"
        st=initial_state(:exact,1);hx=0.0;hz=0.0;source="analytic exact tensor";hash="analytic"
    else
        source=joinpath(@__DIR__,"results",name,"state.jls")
        if isfile(source) && !parse(Bool,opts["portable"])
            chk=deserialize(source);st=chk.state;hx=chk.hx;hz=chk.hz
        else
            source=joinpath(@__DIR__,"frozen_states",name*".json");j=JSON.parsefile(source)
            ts=Dict{FieldLoop.Node,Array{Float64}}()
            for entry in j["tensors"]
                k=entry["key"];node=(Symbol(k[1]),Int(k[2]),Int(k[3]))
                ts[node]=reshape(Float64.(entry["data"]),Tuple(Int.(entry["shape"])))
            end
            ws=Dict{FieldLoop.Bond,Vector{Float64}}(Tuple(Int.(w["key"]))=>Float64.(w["data"]) for w in j["weights"])
            st=FieldLoop.SplitState(ts,ws,Int(j["L"]),Int(j["D"]));hx=j["hx"];hz=j["hz"]
        end
        hash=bytes2hex(open(sha256,source))
    end
    out=joinpath(@__DIR__,"results","vumps","$name-chi$chi-seed$seed-balanced$balanced"*opts["tag"]);mkpath(out)
    result=Dict{String,Any}("state"=>name,"D"=>st.D,"chi"=>chi,"seed"=>seed,"balanced"=>balanced,
        "hx"=>hx,"hz"=>hz,"source"=>source,"checkpoint_sha256"=>hash,"settings"=>opts,"status"=>"running",
        "julia"=>string(VERSION),"mpskit"=>string(pkgversion(BoundaryAudit.MPSKit)))
    save()=open(io->JSON.print(io,result),joinpath(out,"result.json"),"w")
    save();t=time()
    try
        aa,qs,logs=prepare_layers(st;balanced)
        result["blocking_log_scales"]=logs
        println("START frozen=$name D=$(st.D) χ=$chi seed=$seed balanced=$balanced");flush(stdout)
        initial=isempty(opts["resume-boundaries"]) ? nothing : deserialize(opts["resume-boundaries"])
        down,up,infos=solve_boundaries(qs;chi,seed,tol=parse(Float64,opts["tol"]),maxiter=parse(Int,opts["maxiter"]),maxseconds=parse(Float64,opts["budget"]),initial)
        result["boundaries"]=infos;save()
        result["norm_direction_relative_difference"]=abs(infos[1].normcell-infos[2].normcell)/max(infos[1].normcell,infos[2].normcell)
        serialize(joinpath(out,"boundaries.jls"),(;down,up))
        m=measure_boundary(st,aa,qs,down,up,hx,hz)
        result["measurement"]=m
        result["status"]=all(i->i.converged,infos) && all(s->s.converged,m.strip_info) && m.imagmax<1e-7 && m.physical_excess<1e-6 ? "converged_candidate" : "not_converged_or_unphysical"
        if name=="exact"
            result["exact_control_passed"]=abs(m.energy+1)<1e-7 && maximum(abs(m.observables["$k$r$c"]-1) for k in ("A","B"),r in 1:2,c in 1:2)<1e-7 && abs(m.mx)<1e-7 && abs(m.mz)<1e-7
        end
        name=="exact" || (@assert bytes2hex(open(sha256,source))==hash "Physical checkpoint changed")
        println("RESULT ",m);flush(stdout)
    catch err
        result["status"]="error";result["error"]=sprint(showerror,err)
        rethrow()
    finally
        result["seconds"]=time()-t;save()
    end
    println("END ",result["status"]," seconds=",result["seconds"]);flush(stdout)
end
if abspath(PROGRAM_FILE)==@__FILE__;main();end

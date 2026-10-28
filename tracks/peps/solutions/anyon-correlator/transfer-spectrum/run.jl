module TransferRun
const PROCESS_START=time()
include("core.jl")
using .ToricTransferSpectrum, SHA, LinearAlgebra
const T=ToricTransferSpectrum
const S=T.OneSiteZ2SU
function options(args)
    d=Dict("state"=>"","num"=>"1","nk"=>"32","budget"=>"600","out"=>"","tol"=>"1e-10","resolution"=>"1e-12","diagnostic"=>"false")
    for arg in args
        kv=split(replace(arg,r"^--"=>""),'=';limit=2)
        length(kv)==2 && haskey(d,kv[1]) || throw(ArgumentError("unknown option $arg"))
        d[kv[1]]=kv[2]
    end
    isempty(d["state"]) && throw(ArgumentError("--state is required"))
    isempty(d["out"]) && throw(ArgumentError("--out is required"))
    0<parse(Float64,d["budget"])<=600 || throw(ArgumentError("budget must be in (0,600]"))
    parse(Int,d["nk"])>0 && parse(Int,d["num"])>0 || throw(ArgumentError("nk and num must be positive"))
    d
end
function worker_main(args)
    opts=options(args);out=opts["out"];budget=parse(Float64,opts["budget"]);deadline=PROCESS_START+budget
    println("TRANSFER START state=$(opts["state"]) ordinary down-row MPO, step=1, budget=$budget seconds");flush(stdout)
    st,metadata=Base.invokelatest(T.load_production_state,opts["state"])
    S.prepare_output(out)
    jsonl=joinpath(out,"spectrum.jsonl");open(jsonl,"w") do io;end
    source_files=[f for f in readdir(@__DIR__;join=true) if endswith(f,".jl") || endswith(f,".py")]
    nk=parse(Int,opts["nk"]);num=parse(Int,opts["num"]);tol=parse(Float64,opts["tol"]);resolution=parse(Float64,opts["resolution"])
    diagnostic=opts["diagnostic"]=="true"
    ks=diagnostic ? [-pi,-pi/2,0.,pi/2,pi] : collect(range(-pi;step=2pi/nk,length=nk))
    settings=(;opts,input_sha256=bytes2hex(sha256(read(opts["state"]))),input_metadata=metadata,
        direction="down",step=1,transfer_type="ordinary; no twist",momenta=ks,num,tol,resolution,
        julia=string(VERSION),tensorkit=string(pkgversion(T.TensorKit)),mpskit=string(pkgversion(T.MPSKit)),
        source_sha256=Dict(basename(f)=>bytes2hex(sha256(read(f))) for f in source_files))
    S.write_json(joinpath(out,"settings.json"),settings)
    points=SectorPoint[]; status="finished";boundary_diagnostics=nothing
    try
        net=Base.invokelatest(build_transfer,st)
        pair=Base.invokelatest(solve_boundaries,net;options=(;tol,maxiter=100,deadline))
        Base.invokelatest(T.save_boundaries,joinpath(out,"boundaries.json"),pair)
        boundary_diagnostics=pair.diagnostics
        S.write_json(joinpath(out,"fixedpoints.json"),(;normalized_eigenvalues=[1.,1.],diagnostics=boundary_diagnostics,
            note="Two symmetry-related boundary representatives; separate from orthogonal quasiparticle modes"))
        stop=false
        for g in 0:1,q in 0:1
            label=SectorLabel(g,q)
            for k in ks
                current=Base.invokelatest(solve_sector,net,pair,label;momenta=[k],num,options=(;tol,resolution,deadline))
                for p in current
                    T.append_point(jsonl,p);push!(points,p)
                    println("SPECTRUM g=$g q=$q k=$k branch=$(p.branch) |r|=$(p.rho) residual=$(p.projected_residual) status=$(p.status)");flush(stdout)
                    if p.status in ("budget","not_converged"); status=p.status;stop=true;end
                end
                T.write_csv(jsonl,joinpath(out,"spectrum.csv"))
                stop && break
            end
            stop && break
        end
        residuals=[p.projected_residual for p in points if p.projected_residual!==nothing]
        rhos=[p.rho for p in points if p.rho!==nothing]
        counts=Dict(s=>count(p->p.status==s,points) for s in ("converged","zero_mode","below_resolution","empty_sector","not_converged","budget"))
        S.write_json(joinpath(out,"summary.json"),(;status,passed=status=="finished",count=length(points),counts,
            max_projected_residual=isempty(residuals) ? nothing : maximum(residuals),max_rho=isempty(rhos) ? nothing : maximum(rhos),
            boundary_diagnostics,normalization=net.normalization,string_dictionary=T.string_dictionary(),
            wall_seconds=time()-PROCESS_START,peak_rss_bytes=Sys.maxrss()))
    catch err
        S.write_json(joinpath(out,"summary.json"),(;status="error",passed=false,error=sprint(showerror,err),boundary_diagnostics))
        T.write_csv(jsonl,joinpath(out,"spectrum.csv"))
        rethrow()
    end
    println("TRANSFER END status=$status points=$(length(points))");flush(stdout)
    return status
end
function main(args=ARGS)
    "--worker" in args && return Base.invokelatest(worker_main,filter(!=("--worker"),args))
    opts=options(args);out=opts["out"];budget=parse(Float64,opts["budget"])
    ispath(out) && throw(ArgumentError("output already exists: $out"))
    project=dirname(Base.active_project())
    cmd=`$(Base.julia_cmd()) --startup-file=no -O1 --project=$project $(@__FILE__) --worker $args`
    deadline=PROCESS_START+budget-min(10.,budget/10)
    remaining=deadline-time();remaining>0 || error("imports exhausted budget")
    S.supervise(cmd,out;budget=remaining,deadline)
end
end
if abspath(PROGRAM_FILE)==@__FILE__;TransferRun.main();end

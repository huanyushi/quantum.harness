using Printf, Dates, SHA, JLD2, Pkg
include("network.jl")
include("objective.jl")
include("optimize.jl")

function save_json(path,data)
    open(path,"w") do io
        write(io,JSON.json(data));write(io,'\n');flush(io)
    end
end

function main(args=ARGS)
    isempty(args) && error("usage: run.jl calibrate|probe|session|suite|run|audit --out=DIR [--hx=0.1 --hz=0.1 --D=2 --chi=8 --seed=19 --uniform=true]")
    mode = first(args)
    opts = Dict((split(a[3:end],"=";limit=2)...,) for a in args[2:end])
    number(k,default,T) = parse(T,get(opts,k,string(default)))
    D,chi,seed = number("D",2,Int),number("chi",8,Int),number("seed",19,Int)
    hx,hz = number("hx",0.0,Float64),number("hz",0.1,Float64)
    uniform = parse(Bool,get(opts,"uniform","false"))
    exact_start = parse(Bool,get(opts,"exact","false"))
    gradient_method = Symbol(get(opts,"gradient","fixedpoint"))
    unroll_steps = number("unroll",20,Int)
    ctm_tol = number("ctm-tol",1e-11,Float64)
    ctm_maxiter = number("ctm-maxiter",1000,Int)
    gradient_algorithms(gradient_method,unroll_steps,ctm_tol,1000,1e-8)
    out = get(opts,"out",joinpath(@__DIR__,"results",mode))
    isdir(out) && !isempty(readdir(out)) && error("output directory is not empty: $out")
    mkpath(out)
    versions = Dict(d.name=>string(d.version) for d in values(Pkg.dependencies()) if d.name in ("PEPSKit","TensorKit","Zygote","JLD2","JSON","MPSKit"))
    hashes = Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in ("network.jl","objective.jl","optimize.jl","run.jl") )
    settings = Dict("mode"=>mode,"D"=>D,"chi"=>chi,"seed"=>seed,"hx"=>hx,"hz"=>hz,
                    "gradient_method"=>String(gradient_method),"unroll_steps"=>unroll_steps,
                    "ctm_tol"=>ctm_tol,"ctm_maxiter"=>ctm_maxiter,"options"=>opts,"date"=>string(now()),"julia"=>string(VERSION),
                    "versions"=>versions,"source_hashes"=>hashes,"agent"=>"openai/gpt-6.1-sol")
    save_json(joinpath(out,"settings.json"),settings)
    mkpath(joinpath(out,"source"))
    for f in keys(hashes)
        write(joinpath(out,"source",f),read(joinpath(@__DIR__,f)))
    end
    write(joinpath(out,"source","reference.jl"),read(joinpath(@__DIR__,"..","..","model","reference.jl")))
    x = initial_parameters(D,seed;uniform,exact=exact_start)
    if haskey(opts,"checkpoint")
        mode=="suite" && error("use a single-field mode for checkpoint initialization")
        saved = load(opts["checkpoint"])
        Int(saved["D"])==D && saved["hx"]==hx && saved["hz"]==hz || error("checkpoint setup mismatch")
        x = Vector{Float64}(saved["parameters"])
        uniform = length(x)==parameter_count(D;uniform=true)
        settings["initialization"]="checkpoint-continuation"
        settings["checkpoint_sha256"]=bytes2hex(sha256(read(opts["checkpoint"])))
    elseif haskey(opts,"snapshot")
        exact_start && error("--exact and --snapshot are mutually exclusive")
        uniform && error("uniform snapshot sharing requires explicit gauge matching; no tensor averaging")
        x,D,data = snapshot_parameters(opts["snapshot"])
        D==settings["D"] || error("snapshot D differs from requested D")
        hx==data["hx"] && hz==data["hz"] || error("snapshot field differs from requested field")
        settings["initialization"] = "loop-control"
        settings["snapshot_sha256"] = bytes2hex(sha256(read(opts["snapshot"])))
    else
        settings["initialization"] = exact_start ? (uniform ? "analytic-anchor-uniform" : "analytic-anchor-independent") : (uniform ? "random-uniform" : "random-independent")
    end
    settings["uniform"] = uniform
    settings["exact_start"] = exact_start
    settings["parameter_count"] = length(x)
    save_json(joinpath(out,"settings.json"),settings)
    jldsave(joinpath(out,"input.jld2");parameters=x,D,hx,hz,chi,uniform,seed)
    if mode=="calibrate"
        hx==0 && hz==0 || error("exact calibration is only defined at zero field")
        x = initial_parameters(D,seed;exact=true)
        settings["initialization"] = "exact-control"
        save_json(joinpath(out,"settings.json"),settings)
        result = if haskey(opts,"replay")
            data = JSON.parsefile(opts["replay"])
            rows = [row["status"]=="converged" ? (;status="converged",measurement=(;energy=Float64(row["measurement"]["energy"]),imagmax=Float64(row["measurement"]["imagmax"]),observables=Dict{String,Float64}(row["measurement"]["observables"]))) : (;status="failed") for row in data["rows"]]
            (;rows,source=opts["replay"],source_sha256=bytes2hex(sha256(read(opts["replay"]))))
        else
            audit(x,D,hx,hz;chi,chi_check=number("chi-check",2chi,Int),gradnorm=Inf,status="exact-control")
        end
        save_json(joinpath(out,"calibration.json"),result)
        # This is a contraction calibration, explicitly not an AD optimization.
        success = calibration_ok(result.rows)
        save_json(joinpath(out,"calibration_verdict.json"),(;contraction_calibrated=success,ad_gradient_verified=false))
        success || error("exact-state contraction calibration failed")
        return success
    elseif mode=="suite"
        verdicts = Bool[]
        for (label,fx,fz) in (("zero",0.0,0.0),("z",0.0,0.1),("x",0.1,0.0),("xz",0.1,0.1))
            pointdir = joinpath(out,label);mkpath(pointdir)
            pointsettings = merge(settings,Dict("hx"=>fx,"hz"=>fz,"source_snapshot"=>joinpath(out,"source")))
            save_json(joinpath(pointdir,"settings.json"),pointsettings)
            println("SUITE point=$label hx=$fx hz=$fz D=$D chi=$chi uniform=$uniform");flush(stdout)
            try
                probe = derivative_probe(x,D,fx,fz;chi,tol=ctm_tol,maxiter=ctm_maxiter,method=gradient_method,unroll_steps)
                probe = merge(probe,(;gradient_sources=Dict(f=>hashes[f] for f in ("network.jl","objective.jl"))))
                save_json(joinpath(pointdir,"probe.json"),probe)
                if !probe.passed
                    push!(verdicts,false);continue
                end
                summary = optimize(copy(x),D,fx,fz;chi,chi_check=number("chi-check",2chi,Int),
                    maxsteps=number("maxsteps",400,Int),budget=number("budget",3600,Float64),
                    tol=ctm_tol,ctm_maxiter,gradient_method,unroll_steps,outdir=pointdir)
                push!(verdicts,summary.audit.verdict.accepted)
            catch err
                err isa Union{ContractionFailure,GradientFailure} || rethrow()
                save_json(joinpath(pointdir,"failure.json"),(;error=sprint(showerror,err)))
                push!(verdicts,false)
            end
            save_json(joinpath(out,"suite_status.json"),(;completed=length(verdicts),accepted=verdicts))
        end
        return all(verdicts)
    elseif mode=="probe" || mode=="session"
        probe = derivative_probe(x,D,hx,hz;chi,tol=ctm_tol,maxiter=ctm_maxiter,method=gradient_method,unroll_steps)
        probe = merge(probe,(;gradient_sources=Dict(f=>hashes[f] for f in ("network.jl","objective.jl"))))
        save_json(joinpath(out,"probe.json"),probe)
        exact = initial_parameters(2,1;exact=true)
        H,ops = operators(0.0,0.0)
        result = contract(exact,2,H;chi=8,tol=1e-10)
        measurement = measure(result.psi,result.env,ops,0.0,0.0)
        _,gradient = energy_gradient(exact,2,H,result;tol=1e-10,method=gradient_method,unroll_steps)
        calibration = (;measurement,gradnorm=norm(gradient),residual=result.info.convergence_error)
        save_json(joinpath(out,"exact.json"),calibration)
        maxerr = maximum(abs(measurement.observables["$k$r$c"]-1) for k in ("A","B"),r in 1:2,c in 1:2)
        println("EXACT e=$(measurement.energy) max_stabilizer_error=$maxerr gradnorm=$(norm(gradient))");flush(stdout)
        probe.passed || error("AD derivative probe failed")
        if mode=="session"
            summary = optimize(x,D,hx,hz;chi,chi_check=number("chi-check",2chi,Int),
                maxsteps=number("maxsteps",400,Int),budget=number("budget",3600,Float64),
                tol=ctm_tol,ctm_maxiter,gradient_method,unroll_steps,outdir=out)
            return summary.audit.verdict.accepted
        end
        return probe.passed
    elseif mode=="run"
        # A production run must cite a passed AD probe from this implementation.
        haskey(opts,"probe") || error("run requires --probe=passed-probe.json")
        p = JSON.parsefile(opts["probe"])
        probe_matches(p,D,chi,hx,hz,length(x);method=gradient_method,unroll_steps) || error("probe has a different field/D/chi/parameterization/gradient method")
        all(f->get(get(p,"gradient_sources",Dict()),f,nothing)==hashes[f],("network.jl","objective.jl")) || error("probe was generated by a different gradient implementation")
        summary = optimize(x,D,hx,hz;chi,chi_check=number("chi-check",2chi,Int),
            maxsteps=number("maxsteps",400,Int),budget=number("budget",3600,Float64),
            tol=ctm_tol,ctm_maxiter,gradient_method,unroll_steps,outdir=out)
        return summary.audit.verdict.accepted
    elseif mode=="audit"
        haskey(opts,"checkpoint") || error("audit requires --checkpoint=final.jld2")
        saved = load(opts["checkpoint"])
        Int(saved["D"])==D && saved["hx"]==hx && saved["hz"]==hz || error("checkpoint setup mismatch")
        result = audit(saved["parameters"],D,hx,hz;chi,chi_check=number("chi-check",2chi,Int),gradnorm=saved["gradnorm"],status=saved["status"])
        save_json(joinpath(out,"audit.json"),result)
        return result.verdict.accepted
    else
        error("unknown mode $mode")
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    exit(main() ? 0 : 1)
end

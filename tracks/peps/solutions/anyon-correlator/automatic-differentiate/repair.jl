# Reliable-gradient continuation of the three actual finite-field checkpoints.
include("diagnose.jl")

function physical_fd_check(x,D,hx,hz;chi=8,tol=1e-11,steps=40)
    H,_ = operators(hx,hz)
    reference = contract(x,D,H;chi,tol,maxiter=1000)
    _,g = energy_gradient(x,D,H,reference;tol,method=:unrolled,unroll_steps=steps)
    rows = NamedTuple[]
    for kind in (:X,:Z)
        (kind==:X ? hx : hz)==0 && continue
        d = field_filter_direction(x,D,kind)
        for epsilon in (1e-4,1e-5)
            row = try
                plus = contract(x+epsilon*d,D,H;chi,tol,maxiter=1000).energy
                minus = contract(x-epsilon*d,D,H;chi,tol,maxiter=1000).energy
                fd = (plus-minus)/(2epsilon)
                ad = dot(g,d)
                (;direction=String(kind),epsilon,fd,ad,error=abs(fd-ad),passed=abs(fd-ad)<=2e-6)
            catch err
                err isa ContractionFailure || rethrow()
                (;direction=String(kind),epsilon,passed=false,error_message=sprint(showerror,err))
            end
            push!(rows,row)
            println("最终物理方向 FD：$(kind) ε=$(epsilon) passed=$(row.passed)");flush(stdout)
        end
    end
    return (;rows,passed=all(r->r.passed,rows))
end

function repair_suite(out;chi=8,chi_check=16,steps=20,budget=3600,maxsteps=400,tol=1e-11,labels=("z","xz","x"))
    isdir(out) && !isempty(readdir(out)) && error("repair output directory is not empty")
    mkpath(out)
    mkpath(joinpath(out,"source"))
    files = ("network.jl","objective.jl","optimize.jl","run.jl","diagnose.jl","repair.jl")
    hashes = Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files)
    for f in files;write(joinpath(out,"source",f),read(joinpath(@__DIR__,f)));end
    write(joinpath(out,"source","reference.jl"),read(joinpath(@__DIR__,"..","finite-field-loop-update","reference.jl")))
    verdicts = Bool[]
    for label in labels
        checkpoint = joinpath(@__DIR__,"results","analytic-anchor-suite",label,"final.jld2")
        saved = load(checkpoint)
        x,D,hx,hz = Vector{Float64}(saved["parameters"]),Int(saved["D"]),Float64(saved["hx"]),Float64(saved["hz"])
        pointdir = joinpath(out,label);mkpath(pointdir)
        settings = (;D,hx,hz,chi,chi_check,ctm_tol=tol,ctm_maxiter=1000,gradient_method="unrolled",unroll_steps=steps,
                     initialization="checkpoint-continuation",source_checkpoint=checkpoint,
                     checkpoint_sha256=bytes2hex(sha256(read(checkpoint))),parameter_count=length(x),
                     virtual_z2_constraint=false,source_hashes=hashes,julia=string(VERSION),budget,maxsteps)
        save_json(joinpath(pointdir,"settings.json"),settings)
        println("继续有限场优化：$(label)，hₓ=$(hx) h_z=$(hz)，D=$(D) χ=$(chi) N=$(steps)");flush(stdout)
        try
            probe = derivative_probe(x,D,hx,hz;chi,tol,method=:unrolled,unroll_steps=steps)
            save_json(joinpath(pointdir,"probe.json"),probe)
            if !probe.passed
                save_json(joinpath(pointdir,"failure.json"),(;reason="derivative_probe_failed"))
                push!(verdicts,false);continue
            end
            summary = optimize(x,D,hx,hz;chi,chi_check,maxsteps,budget,tol,
                               gradient_method=:unrolled,unroll_steps=steps,outdir=pointdir)
            finalx = Vector{Float64}(load(joinpath(pointdir,"final.jld2"))["parameters"])
            finalprobe = derivative_probe(finalx,D,hx,hz;chi,tol,method=:unrolled,unroll_steps=2steps)
            save_json(joinpath(pointdir,"final_probe.json"),finalprobe)
            fd = physical_fd_check(finalx,D,hx,hz;chi,tol,steps=2steps)
            save_json(joinpath(pointdir,"physical_fd.json"),fd)
            reasons = copy(summary.audit.verdict.reasons)
            finalprobe.passed || push!(reasons,"final_derivative_probe")
            fd.passed || push!(reasons,"physical_direction_fd")
            verdict = (;accepted=isempty(reasons),reasons)
            summary = merge(summary,(;final_probe_passed=finalprobe.passed,physical_fd_passed=fd.passed,
                                     audit=merge(summary.audit,(;verdict))))
            save_json(joinpath(pointdir,"summary.json"),summary)
            push!(verdicts,verdict.accepted)
            println("$(label) 最终验收：$(verdict.accepted)，原因=$(reasons)");flush(stdout)
        catch err
            err isa Union{ContractionFailure,GradientFailure} || rethrow()
            save_json(joinpath(pointdir,"failure.json"),(;error=sprint(showerror,err)))
            push!(verdicts,false)
        end
        save_json(joinpath(out,"suite_status.json"),(;points=labels,completed=length(verdicts),accepted=verdicts))
    end
    return all(verdicts)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=1 || error("usage: repair.jl OUTDIR [BUDGET_SECONDS] [LABELS_CSV]")
    labels = length(ARGS)>2 ? Tuple(split(ARGS[3],',')) : ("z","xz","x")
    all(k->k in ("z","xz","x"),labels) || error("unknown field label")
    exit(repair_suite(ARGS[1];budget=length(ARGS)>1 ? parse(Float64,ARGS[2]) : 3600,labels) ? 0 : 1)
end

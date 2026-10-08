# Frozen-checkpoint derivative diagnostic. No optimization or symmetry projection.
include("run.jl")

function field_filter_direction(x,D,kind)
    width = D^4+4D^2
    count = length(x)==parameter_count(D;uniform=true) ? 1 : 4
    d = zeros(length(x))
    op = kind==:Z ? [1.0 0;0 -1] : [0.0 1;1 0]
    for cell in 0:count-1, offset in (D^4,D^4+2D^2)
        inds = cell*width+offset+1:cell*width+offset+2D^2
        p = reshape(x[inds],2,D^2)
        d[inds] = vec(op*p)
    end
    return normalize(d)
end

function frozen_diagnostic(path,out;chi=8,counts=(10,20,40),tol=1e-11)
    isdir(out) && !isempty(readdir(out)) && error("diagnostic output directory is not empty")
    mkpath(out)
    saved = load(path)
    x,D,hx,hz = Vector{Float64}(saved["parameters"]),Int(saved["D"]),Float64(saved["hx"]),Float64(saved["hz"])
    sources = ("network.jl","objective.jl","diagnose.jl")
    mkpath(joinpath(out,"source"))
    for f in sources;write(joinpath(out,"source",f),read(joinpath(@__DIR__,f)));end
    save_json(joinpath(out,"settings.json"),(;checkpoint=path,checkpoint_sha256=bytes2hex(sha256(read(path))),D,hx,hz,chi,counts,tol,parameter_count=length(x),virtual_z2_constraint=false,julia=string(VERSION)))
    H,ops = operators(hx,hz)
    println("冻结态诊断开始：D=$(D) χ=$(chi) hₓ=$(hx) h_z=$(hz)，参数数目=$(length(x))");flush(stdout)
    reference = contract(x,D,H;chi,tol,maxiter=1000)
    measurement = measure(reference.psi,reference.env,ops,hx,hz)
    save_json(joinpath(out,"reference.json"),(;measurement,residual=Float64(reference.info.convergence_error)))
    println("参考收缩完成：e=$(reference.energy)，residual=$(reference.info.convergence_error)");flush(stdout)

    println("检查原 fixed-point AD（非收敛解将拒绝）");flush(stdout)
    legacy = try
        value,g = energy_gradient(x,D,H,reference;tol,maxiter=1000)
        (;status="valid",energy=value,gradnorm=norm(g),gradient=g)
    catch err
        err isa GradientFailure || rethrow()
        (;status="rejected",message=sprint(showerror,err))
    end
    save_json(joinpath(out,"fixedpoint.json"),legacy)
    println("原 fixed-point AD 状态：$(legacy.status)");flush(stdout)

    directions = [("random_$s",normalize(randn(MersenneTwister(s),length(x)))) for s in (17,31)]
    hx!=0 && push!(directions,("physical_X",field_filter_direction(x,D,:X)))
    hz!=0 && push!(directions,("physical_Z",field_filter_direction(x,D,:Z)))
    fresh = NamedTuple[]
    for (label,d) in directions,epsilon in (1e-3,1e-4,1e-5)
        samples = NamedTuple[]
        for sign in (-1,1),seed in (2026,1)
            row = try
                result = contract(x+sign*epsilon*d,D,H;chi,tol,maxiter=1000,seed)
                (;sign,seed,status="converged",energy=result.energy,residual=Float64(result.info.convergence_error))
            catch err
                err isa ContractionFailure || rethrow()
                (;sign,seed,status="failed",message=sprint(showerror,err))
            end
            push!(samples,row)
        end
        good = all(r->r.status=="converged",samples)
        slope = good ? (samples[3].energy-samples[1].energy)/(2epsilon) : nothing
        spread = good ? max(abs(samples[1].energy-samples[2].energy),abs(samples[3].energy-samples[4].energy)) : nothing
        push!(fresh,(;direction=label,epsilon,slope,energy_seed_spread=spread,samples))
        save_json(joinpath(out,"fresh_fd.json"),fresh)
        println("fresh FD：$label ε=$epsilon slope=$slope seed_spread=$spread");flush(stdout)
    end

    previous_gradient = nothing
    rows = NamedTuple[]
    for steps in counts
        println("开始展开 AD：N=$steps");flush(stdout)
        start = time()
        value,g = energy_gradient(x,D,H,reference;tol,method=:unrolled,unroll_steps=steps)
        elapsed = time()-start
        delta = isnothing(previous_gradient) ? nothing : norm(g-previous_gradient)
        checks = NamedTuple[]
        for (label,d) in directions,epsilon in (1e-4,1e-5)
            plus = unrolled_energy(x+epsilon*d,D,H,reference;steps,tol)
            minus = unrolled_energy(x-epsilon*d,D,H,reference;steps,tol)
            fd = (plus-minus)/(2epsilon)
            ad = dot(g,d)
            push!(checks,(;direction=label,epsilon,fd,ad,error=abs(fd-ad),passed=abs(fd-ad)<=max(2e-6,0.005abs(ad))))
            println("展开 FD：N=$steps $label ε=$epsilon AD=$ad FD=$fd");flush(stdout)
        end
        row = (;steps,energy=value,gradnorm=norm(g),gradient=g,gradient_delta=delta,seconds=elapsed,checks)
        push!(rows,row)
        previous_gradient = g
        save_json(joinpath(out,"unrolled.json"),rows)
        println("展开 AD 完成：N=$steps |g|=$(norm(g)) Δg=$delta wall=$(round(elapsed;digits=1))s");flush(stdout)
    end
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=2 || error("usage: diagnose.jl CHECKPOINT OUTDIR [CHI]")
    frozen_diagnostic(ARGS[1],ARGS[2];chi=length(ARGS)>2 ? parse(Int,ARGS[3]) : 8)
end

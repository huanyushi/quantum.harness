using Zygote, Printf, Logging

struct ContractionFailure <: Exception
    message::String
end
Base.showerror(io::IO,e::ContractionFailure) = print(io,e.message)

struct GradientFailure <: Exception
    message::String
end
Base.showerror(io::IO,e::GradientFailure) = print(io,e.message)

function checked_gradient_call(f)
    buffer = IOBuffer()
    value = with_logger(SimpleLogger(buffer,Logging.Warn)) do
        f()
    end
    messages = String(take!(buffer))
    failures = ("gradient fixed-point iteration reached maximal number",
                "linsolve stopped without converging", "eigensolve stopped without converging")
    any(m->occursin(m,messages),failures) && throw(GradientFailure(
        "AD adjoint did not converge; returned vector is rejected:\n"*messages))
    return value,messages
end

function contract(x,D,H;chi=8,tol=1e-9,maxiter=300,seed=2026,initial_env=nothing)
    psi = make_peps(x,D)
    Random.seed!(max(seed,0))
    env0 = isnothing(initial_env) ? CTMRGEnv(randn,ComplexF64,psi,ComplexSpace(chi)) : initial_env
    alg = SimultaneousCTMRG(;tol,maxiter,verbosity=0)
    start = time()
    env,info = leading_boundary(env0,psi,alg)
    info.converged && isfinite(info.convergence_error) || throw(ContractionFailure("CTMRG chi=$chi seed=$seed residual=$(info.convergence_error)"))
    value = expectation_value(psi,H,env)/8
    isfinite(real(value)) && abs(imag(value))<=1e-7 || throw(ContractionFailure("nonreal/nonfinite energy $value"))
    return (;energy=real(value),psi,env,info,seconds=time()-start,chi,seed)
end

function dominant_cluster(rows;relative_tol=1e-8)
    valid = findall(r->isfinite(r.normcell) && r.normcell>0,rows)
    isempty(valid) && throw(ContractionFailure("no positive finite norm candidate"))
    best = valid[argmax([rows[i].normcell for i in valid])]
    top = rows[best].normcell
    members = filter(i->abs(rows[i].normcell-top)/top<=relative_tol,valid)
    return (;best,members,top)
end

function dominant_contract(x,D,H;chi=16,tol=1e-10,maxiter=1000,seed=2026,
                           seeds=(2026,17,1,2,3,42,71,7,11),minimum_members=2,initial_env=nothing)
    candidates = NamedTuple[]
    excluded = NamedTuple[]
    if !isnothing(initial_env)
        try
            r=contract(x,D,H;chi,tol,maxiter,seed=-1,initial_env)
            n=network_value(r.psi,r.env)
            isfinite(real(n)) && real(n)>0 && abs(imag(n))<=1e-7*real(n) || throw(ContractionFailure("invalid warm network norm"))
            push!(candidates,merge(r,(;normcell=Float64(real(n)))))
        catch err
            err isa ContractionFailure || rethrow()
            push!(excluded,(;seed=-1,status="failed",message=sprint(showerror,err)))
        end
    end
    for s in unique((seed,seeds...))
        try
            r = contract(x,D,H;chi,tol,maxiter,seed=s)
            n = network_value(r.psi,r.env)
            isfinite(real(n)) && real(n)>0 && abs(imag(n))<=1e-7*real(n) || throw(ContractionFailure("invalid network norm $n"))
            push!(candidates,merge(r,(;normcell=Float64(real(n)))))
        catch err
            err isa ContractionFailure || rethrow()
            push!(excluded,(;seed=s,status="failed",message=sprint(showerror,err)))
        end
    end
    choice = dominant_cluster(candidates)
    length(choice.members)>=minimum_members || throw(ContractionFailure(
        "dominant norm has insufficient independent initializations; "*
        string([(seed=c.seed,normcell=c.normcell,energy=c.energy) for c in candidates])))
    values = [candidates[i].energy for i in choice.members]
    maximum(values)-minimum(values)<=1e-6 || throw(ContractionFailure("dominant-norm energy disagreement"))
    for (i,c) in enumerate(candidates)
        i in choice.members || push!(excluded,(;seed=c.seed,status="subdominant",normcell=c.normcell,energy=c.energy))
    end
    best = choice.members[argmin([candidates[i].info.convergence_error for i in choice.members])]
    return merge(candidates[best],(;dominance=(;normcell=choice.top,
                    leading_seeds=[candidates[i].seed for i in choice.members],excluded)))
end

function dominant_audit(x,D,hx,hz;chi=16,chi_check=32,tol=1e-10,maxiter=1000,gradnorm=Inf,status="unknown")
    H,ops = operators(hx,hz)
    rows = NamedTuple[]
    diagnostics = NamedTuple[]
    for ch in (chi,chi_check)
        selected = dominant_contract(x,D,H;chi=ch,tol,maxiter,minimum_members=3,
                    seeds=(2026,17,1,2,3,42,71,7))
        push!(diagnostics,(;chi=ch,selected.dominance...))
        # Each retained seed is independently recontracted, never an averaged result.
        for s in selected.dominance.leading_seeds
            r = contract(x,D,H;chi=ch,tol,maxiter,seed=s)
            m = measure(r.psi,r.env,ops,hx,hz)
            push!(rows,(;chi=ch,seed=s,status="converged",residual=r.info.convergence_error,measurement=m))
        end
    end
    return (;rows,diagnostics,verdict=assess_audit(rows,hx,hz;gradnorm,status),
             selection_rule="largest norm, with independent boundary-growth validation on the input")
end

function gradient_algorithms(method,steps,tol,maxiter,grad_tol)
    method in (:fixedpoint,:unrolled) || throw(ArgumentError("unknown gradient method $method"))
    if method==:unrolled
        steps>0 || throw(ArgumentError("unrolled AD needs a positive fixed iteration count"))
        # miniter=maxiter defines the same finite computation graph for the
        # derivative and its finite-difference reference. No implicit rrule.
        return SimultaneousCTMRG(;tol,miniter=steps,maxiter=steps,verbosity=0),nothing
    end
    return SimultaneousCTMRG(;tol,maxiter,verbosity=0),
        PEPSKit.GradientAlgorithm(;alg=:FixedPointGradient,tol=grad_tol,maxiter=30,verbosity=1)
end

function parameter_pullback(x,D,peps_gradient)
    return Vector{Float64}(only(Zygote.gradient(x) do y
        psi = make_peps(y,D)
        real(sum(dot(peps_gradient.A[i],psi.A[i]) for i in eachindex(psi.A)))
    end))
end

function energy_gradient(x,D,H,result;tol=1e-9,maxiter=300,grad_tol=1e-8,method=:fixedpoint,unroll_steps=20)
    alg,gad = gradient_algorithms(method,unroll_steps,tol,maxiter,grad_tol)
    # Two AD stages implement exactly the chain rule but avoid asking Zygote
    # to compile the entire CTMRG pullback and split-network graph together.
    (value,grads),messages = checked_gradient_call() do
        Zygote.withgradient(result.psi) do psi
            env,_ = PEPSKit.hook_pullback(leading_boundary,result.env,psi,alg;alg_rrule=gad)
            real(PEPSKit.cost_function(psi,env,H))/8
        end
    end
    isempty(messages) || print(stderr,messages)
    peps_gradient = only(grads)
    gradient = parameter_pullback(x,D,peps_gradient)
    all(isfinite,gradient) || error("AD parameter gradient is nonfinite")
    abs(value-result.energy)<=1e-7 || throw(ContractionFailure("AD forward objective changed by $(value-result.energy)"))
    return Float64(value),Vector{Float64}(gradient)
end

function unrolled_energy(x,D,H,reference;steps=20,tol=1e-9)
    alg,_ = gradient_algorithms(:unrolled,steps,tol,steps,1e-8)
    psi = make_peps(x,D)
    env,_ = leading_boundary(reference.env,psi,alg)
    value = real(PEPSKit.cost_function(psi,env,H))/8
    isfinite(value) || throw(ContractionFailure("unrolled objective is nonfinite"))
    return Float64(value)
end

function measure(psi,env,ops,hx,hz)
    values = Dict(name=>expectation_value(psi,op,env) for (name,op) in ops)
    imagmax = maximum(abs(imag(v)) for v in Base.values(values))
    vals = Dict(k=>Float64(real(v)) for (k,v) in values)
    astar = sum(vals["A$r$c"] for r in 1:2,c in 1:2)/4
    bplaq = sum(vals["B$r$c"] for r in 1:2,c in 1:2)/4
    mx = sum(vals["$k$r$c"] for k in ("XE","XN"),r in 1:2,c in 1:2)/8
    mz = sum(vals["$k$r$c"] for k in ("ZE","ZN"),r in 1:2,c in 1:2)/8
    return (;energy=-(astar+bplaq)/2-hx*mx-hz*mz,mx,mz,astar,bplaq,imagmax,observables=vals)
end

function physical_ok(m,hx,hz;tol=1e-6)
    all(isfinite,Base.values(m.observables)) && isfinite(m.energy) && m.imagmax<=1e-7 &&
        maximum(abs,Base.values(m.observables))<=1+tol && m.energy>=-1-abs(hx)-abs(hz)-tol
end

function calibration_ok(rows)
    length(rows)>=4 && all(row->row.status=="converged",rows) || return false
    return all(rows) do row
        m = row.measurement
        abs(m.energy+1)<=1e-8 && m.imagmax<=1e-7 &&
            maximum(abs(m.observables["$k$r$c"]-1) for k in ("A","B"),r in 1:2,c in 1:2)<=1e-8
    end
end

function probe_matches(p,D,chi,hx,hz,nparameters;method=nothing,unroll_steps=nothing)
    setup = get(p,"passed",false)==true && get(p,"D",nothing)==D &&
        get(p,"chi",nothing)==chi && get(p,"hx",nothing)==hx &&
        get(p,"hz",nothing)==hz && get(p,"parameter_count",nothing)==nparameters
    setup || return false
    isnothing(method) && return true
    get(p,"gradient_method",nothing)==String(method) || return false
    return method!=:unrolled || get(p,"unroll_steps",nothing)==unroll_steps
end

function assess_audit(rows,hx,hz;gradnorm,status)
    reasons = String[]
    status=="stationary" && gradnorm<=1e-5 || push!(reasons,"optimizer_not_stationary")
    good = filter(r->r.status=="converged",rows)
    length(good)==length(rows) || push!(reasons,"contraction_failed")
    isempty(good) && return (;accepted=false,reasons=vcat(reasons,"no_contractions"))
    chi = minimum(r.chi for r in rows)
    base = filter(r->r.chi==chi,good)
    length(unique(r.seed for r in base))>=3 || push!(reasons,"insufficient_initializations")
    any(r->r.chi>chi,good) || push!(reasons,"missing_chi_check")
    all(r->physical_ok(r.measurement,hx,hz),good) || push!(reasons,"physical_bounds")
    measurements = [r.measurement for r in good]
    maximum(m.energy for m in measurements)-minimum(m.energy for m in measurements)<=1e-6 || push!(reasons,"energy_disagreement")
    keys0 = keys(first(measurements).observables)
    spread = maximum(maximum(m.observables[k] for m in measurements)-minimum(m.observables[k] for m in measurements) for k in keys0)
    spread<=1e-5 || push!(reasons,"observable_disagreement")
    ref = reference_values(hx,hz)
    all(m->abs(m.energy-ref.energy)<=1e-5,measurements) || push!(reasons,"energy_benchmark")
    for name in (:mx,:mz)
        target = getproperty(ref,name)
        limit = abs(target)>1e-12 ? 0.01abs(target) : 1e-6
        all(m->abs(getproperty(m,name)-target)<=limit,measurements) || push!(reasons,"$(name)_benchmark")
    end
    if hx==0 || hz==0
        prefix = hx==0 ? "B" : "A"
        all(m->maximum(abs(m.observables["$prefix$r$c"]-1) for r in 1:2,c in 1:2)<=1e-6,measurements) || push!(reasons,"conserved_sector")
    end
    if hx==0 && hz==0
        all(m->maximum(abs(m.observables["$k$r$c"]-1) for k in ("A","B"),r in 1:2,c in 1:2)<=1e-6,measurements) || push!(reasons,"zero_field_stabilizers")
    end
    return (;accepted=isempty(reasons),reasons)
end

function audit(x,D,hx,hz;chi=8,chi_check=16,tol=1e-9,maxiter=300,gradnorm=Inf,status="unknown")
    chi_check>chi || throw(ArgumentError("chi_check must exceed chi"))
    H,ops = operators(hx,hz)
    rows = NamedTuple[]
    for (ch,seed) in ((chi,2026),(chi,1),(chi,2),(chi_check,2026))
        row = try
            result = contract(x,D,H;chi=ch,tol,maxiter,seed)
            measurement = measure(result.psi,result.env,ops,hx,hz)
            (;chi=ch,seed,status="converged",residual=Float64(result.info.convergence_error),measurement)
        catch err
            err isa ContractionFailure || rethrow()
            (;chi=ch,seed,status="failed",error=sprint(showerror,err))
        end
        push!(rows,row)
        println("audit chi=$ch seed=$seed status=$(row.status)");flush(stdout)
    end
    return (;rows,verdict=assess_audit(rows,hx,hz;gradnorm,status))
end

function derivative_probe(x,D,hx,hz;chi=8,tol=1e-11,maxiter=1000,method=:fixedpoint,unroll_steps=20)
    H,ops = operators(hx,hz)
    t0 = time()
    result = contract(x,D,H;chi,tol,maxiter)
    compile_objective = time()-t0
    println("probe objective ready energy=$(result.energy) residual=$(result.info.convergence_error) wall=$(round(compile_objective;digits=2))s; starting two-stage AD");flush(stdout)
    t0 = time()
    energy,g = energy_gradient(x,D,H,result;tol,maxiter,method,unroll_steps)
    compile_gradient = time()-t0
    println("probe first energy=$energy gradnorm=$(norm(g)) objective=$(round(compile_objective;digits=2))s gradient=$(round(compile_gradient;digits=2))s");flush(stdout)
    t0 = time(); repeat = contract(x,D,H;chi,tol,maxiter); objective_seconds = time()-t0
    t0 = time(); _,g2 = energy_gradient(x,D,H,repeat;tol,maxiter,method,unroll_steps); gradient_seconds = time()-t0
    convergence = if method==:unrolled
        _,gcheck = energy_gradient(x,D,H,result;tol,maxiter,method,unroll_steps=2unroll_steps)
        (;steps=unroll_steps,check_steps=2unroll_steps,delta=norm(gcheck-g),stable=norm(gcheck-g)<=2e-6)
    else
        (;steps=0,check_steps=0,delta=0.0,stable=true)
    end
    dirs = [normalize(randn(MersenneTwister(s),length(x))) for s in (17,31)]
    norm(g)>1e-12 && push!(dirs,-g/norm(g))
    checks = NamedTuple[]
    for (i,d) in enumerate(dirs),epsilon in (1e-4,1e-5)
        plus = method==:unrolled ? unrolled_energy(x+epsilon*d,D,H,result;steps=unroll_steps,tol) : contract(x+epsilon*d,D,H;chi,tol,maxiter).energy
        minus = method==:unrolled ? unrolled_energy(x-epsilon*d,D,H,result;steps=unroll_steps,tol) : contract(x-epsilon*d,D,H;chi,tol,maxiter).energy
        fd = (plus-minus)/(2epsilon)
        ad = dot(g,d)
        passed = abs(fd-ad)<=max(2e-6,0.005abs(ad))
        push!(checks,(;direction=i,epsilon,fd,ad,passed))
        println("FD direction=$i eps=$epsilon ad=$ad fd=$fd passed=$passed");flush(stdout)
    end
    return (;D,chi,hx,hz,parameter_count=length(x),gradient_method=String(method),unroll_steps,
            objective_kind=method==:unrolled ? "fixed_initial_environment_finite_graph" : "fresh_fixedpoint",
            energy,gradnorm=norm(g),measurement=measure(result.psi,result.env,ops,hx,hz),compile_objective,
            compile_gradient,objective_seconds,gradient_seconds,repeat_gradient_difference=norm(g-g2),
            convergence,checks,passed=convergence.stable && all(c->c.passed,checks))
end

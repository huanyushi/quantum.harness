# Generic nonsymmetric start: norm-dominant objective, fully recorded selection.
include("repair.jl")

function noisy_recovery(out;chi=16,tol=1e-10,budget=3600)
    isdir(out) && !isempty(readdir(out)) && error("noisy output directory is not empty")
    mkpath(out)
    input = joinpath(@__DIR__,"results","perturb-preflight01","z-a0p001","input.jld2")
    saved = load(input)
    x,D,hx,hz = Vector{Float64}(saved["parameters"]),Int(saved["D"]),Float64(saved["hx"]),Float64(saved["hz"])
    files=("network.jl","objective.jl","optimize.jl","run.jl","diagnose.jl","repair.jl","noisy_recover.jl")
    mkpath(joinpath(out,"source"))
    hashes=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files)
    for f in files;write(joinpath(out,"source",f),read(joinpath(@__DIR__,f)));end
    settings=(;D,hx,hz,chi,chi_check=32,ctm_tol=tol,initialization="generic-perturbation",
                amplitude=0.001,seed=71,parameter_count=length(x),virtual_z2_constraint=false,
                selection_rule="largest-norm cluster, not lowest-energy branch",source_hashes=hashes,
                input_sha256=bytes2hex(sha256(read(input))),gradient_method="fixedpoint",budget)
    save_json(joinpath(out,"settings.json"),settings)
    save_json(joinpath(out,"initial_symmetry.json"),virtual_symmetry_defect(x,D))
    H,_=operators(hx,hz)
    println("非对称起点恢复：D=$(D) χ=$(chi) hₓ=$(hx) h_z=$(hz)，使用主导 norm 分支");flush(stdout)
    reference=dominant_contract(x,D,H;chi,tol)
    save_json(joinpath(out,"initial_dominance.json"),reference.dominance)
    println("主导 norm=$(reference.normcell)，e=$(reference.energy)，开始验证 fixed-point 导数");flush(stdout)
    value,g=energy_gradient(x,D,H,reference;tol,maxiter=1000)
    save_json(joinpath(out,"initial_gradient.json"),(;energy=value,gradient=g,gradnorm=norm(g),checks=NamedTuple[]))
    directions=[("random_$s",normalize(randn(MersenneTwister(s),length(x)))) for s in (17,31)]
    push!(directions,("negative_gradient",-g/norm(g)))
    checks=NamedTuple[]
    for (label,d) in directions,epsilon in (1e-4,1e-5)
        plus=dominant_contract(x+epsilon*d,D,H;chi,tol,initial_env=reference.env).energy
        minus=dominant_contract(x-epsilon*d,D,H;chi,tol,initial_env=reference.env).energy
        fd=(plus-minus)/(2epsilon)
        ad=dot(g,d)
        push!(checks,(;direction=label,epsilon,ad,fd,passed=abs(ad-fd)<=max(2e-6,0.005abs(ad))))
        save_json(joinpath(out,"initial_gradient.json"),(;energy=value,gradient=g,checks))
        println("主导分支 FD：$(label) ε=$(epsilon) AD=$(ad) FD=$(fd)");flush(stdout)
    end
    all(c->c.passed,checks) || throw(GradientFailure("norm-dominant derivative failed finite differences"))
    result=optimize(x,D,hx,hz;chi,chi_check=32,tol,budget,maxsteps=200,
                    gradient_method=:fixedpoint,contraction_provider=dominant_contract,
                    audit_provider=dominant_audit,carry_env=true,outdir=out)
    finalx=Vector{Float64}(load(joinpath(out,"final.jld2"))["parameters"])
    save_json(joinpath(out,"final_symmetry.json"),virtual_symmetry_defect(finalx,D))
    return result.audit.verdict.accepted
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: noisy_recover.jl OUTDIR")
    try
        exit(noisy_recovery(ARGS[1]) ? 0 : 1)
    catch err
        if err isa Union{ContractionFailure,GradientFailure}
            save_json(joinpath(ARGS[1],"failure.json"),(;message=sprint(showerror,err)))
        end
        rethrow()
    end
end

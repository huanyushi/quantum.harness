# Forward preflight and optional recovery from genuinely nonsymmetric tensors.
include("repair.jl")

function perturb_preflight(out;labels=("z",),amplitudes=(1e-3,1e-2),tol=1e-10)
    isdir(out) && !isempty(readdir(out)) && error("perturb output directory is not empty")
    mkpath(out)
    mkpath(joinpath(out,"source"))
    files = ("network.jl","objective.jl","optimize.jl","run.jl","diagnose.jl","repair.jl","perturb.jl")
    hashes = Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in files)
    for f in files;write(joinpath(out,"source",f),read(joinpath(@__DIR__,f)));end
    rows = NamedTuple[]
    for label in labels,amplitude in amplitudes
        checkpoint = joinpath(@__DIR__,"results","unrolled-repair01",label,"final.jld2")
        saved = load(checkpoint)
        x,D,hx,hz = Vector{Float64}(saved["parameters"]),Int(saved["D"]),Float64(saved["hx"]),Float64(saved["hz"])
        seed=71
        y = perturb_parameters(x,D,seed;amplitude)
        name = label*"-a"*replace(string(amplitude),"."=>"p")
        dir = joinpath(out,name);mkpath(dir)
        settings = (;D,hx,hz,chi=8,ctm_tol=tol,amplitude,seed,initialization="generic-perturbation",
                     source_checkpoint=checkpoint,checkpoint_sha256=bytes2hex(sha256(read(checkpoint))),
                     virtual_z2_constraint=false,parameter_count=length(y),stage="forward-preflight",
                     gradient_evaluated=false,source_hashes=hashes)
        save_json(joinpath(dir,"settings.json"),settings)
        defect = virtual_symmetry_defect(y,D)
        save_json(joinpath(dir,"symmetry.json"),(;initial=defect,parent=virtual_symmetry_defect(x,D)))
        jldsave(joinpath(dir,"input.jld2");parameters=y,D,hx,hz,seed,amplitude)
        println("通用扰动前向检查：$(label) amplitude=$(amplitude) Z₂ defect=$(defect.maximum)");flush(stdout)
        check = audit(y,D,hx,hz;chi=8,chi_check=16,tol,maxiter=1000)
        save_json(joinpath(dir,"forward_audit.json"),check)
        critical = filter(r->r in ("contraction_failed","insufficient_initializations","missing_chi_check",
                                   "physical_bounds","energy_disagreement","observable_disagreement"),check.verdict.reasons)
        row = (;label,amplitude,dir,defect=defect.maximum,forward_consistent=isempty(critical),reasons=critical)
        push!(rows,row)
        save_json(joinpath(out,"preflight.json"),rows)
        println("扰动 forward consistency：$(row.forward_consistent) $(critical)");flush(stdout)
    end
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=1 || error("usage: perturb.jl OUTDIR")
    perturb_preflight(ARGS[1])
end

# Diagnose CTMRG branch dominance against the independently grown boundary norm.
include("run.jl")

function dominance_preflight(out)
    isdir(out) && !isempty(readdir(out)) && error("dominance output directory is not empty")
    mkpath(out)
    rows = NamedTuple[]
    for subdir in ("z-a0p001","z-a0p01")
        path = joinpath(@__DIR__,"results","perturb-preflight01",subdir,"input.jld2")
        saved = load(path)
        x,D,hx,hz = Vector{Float64}(saved["parameters"]),Int(saved["D"]),Float64(saved["hx"]),Float64(saved["hz"])
        H,ops = operators(hx,hz)
        for chi in (8,16),seed in (2026,1,2,17,3)
            row = try
                r = contract(x,D,H;chi,tol=1e-10,maxiter=1000,seed)
                normcell = network_value(r.psi,r.env)
                m = measure(r.psi,r.env,ops,hx,hz)
                (;case=subdir,chi,seed,status="converged",normcell=real(normcell),
                  norm_imag=imag(normcell),measurement=m,residual=r.info.convergence_error)
            catch err
                err isa ContractionFailure || rethrow()
                (;case=subdir,chi,seed,status="failed",message=sprint(showerror,err))
            end
            push!(rows,row)
            save_json(joinpath(out,"rows.json"),rows)
            println("norm 分支检查：$(subdir) χ=$(chi) seed=$(seed) $(row.status)");flush(stdout)
        end
    end
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: dominance.jl OUTDIR")
    dominance_preflight(ARGS[1])
end

# Independent measurement only: no physical update, AD optimizer or symmetry mask.
using JLD2, SHA, Dates
isdefined(@__MODULE__,:make_peps) || include("network.jl")
isdefined(@__MODULE__,:FieldLoop) || include(joinpath(@__DIR__,"..","..","simple-update","loop","core.jl"))
isdefined(@__MODULE__,:BoundaryAudit) || include(joinpath(@__DIR__,"..","..","simple-update","loop","vumps_core.jl"))

function as_boundary_state(x,D)
    split = split_parameters(x,D)
    tensors = Dict{FieldLoop.Node,Array{Float64}}()
    weights = Dict{FieldLoop.Bond,Vector{Float64}}()
    for r in 1:2,c in 1:2
        tensors[(:V,r,c)]=copy(split.V[r,c])
        tensors[(:E,r,c)]=copy(split.E[r,c])
        tensors[(:N,r,c)]=copy(split.N[r,c])
        for d in 1:4;weights[(r,c,d)]=copy(split.weights[r,c,d]);end
    end
    return FieldLoop.SplitState(tensors,weights,2,D)
end

function boundary_json(path,data)
    open(path,"w") do io;write(io,JSON.json(data));write(io,'\n');flush(io);end
end

function boundary_checks(out)
    isdir(out) && !isempty(readdir(out)) && error("boundary output directory is not empty")
    mkpath(out)
    inputs = (
        ("control",joinpath(@__DIR__,"results","unrolled-repair01","z","final.jld2")),
        ("noise001",joinpath(@__DIR__,"results","perturb-preflight01","z-a0p001","input.jld2")),
        ("noise01",joinpath(@__DIR__,"results","perturb-preflight01","z-a0p01","input.jld2")))
    allrows = NamedTuple[]
    for (label,path) in inputs
        saved = load(path)
        x,D,hx,hz = Vector{Float64}(saved["parameters"]),Int(saved["D"]),Float64(saved["hx"]),Float64(saved["hz"])
        st = as_boundary_state(x,D)
        aa,qs,_ = BoundaryAudit.prepare_layers(st;balanced=true)
        expected = blocked_arrays(x,D)
        adapter_error = maximum(norm(aa[i]-expected[i]/norm(expected[i])) for i in eachindex(aa))
        adapter_error<=1e-12 || error("boundary adapter changed the PEPS: $adapter_error")
        for chi in (8,16),seed in (1,2)
            println("冻结态 boundary-VUMPS：$(label) χ=$(chi) seed=$(seed)");flush(stdout)
            row = try
                down,up,info = BoundaryAudit.solve_boundaries(qs;chi,seed,tol=1e-9,maxiter=150,maxseconds=900)
                measured = BoundaryAudit.measure_boundary(st,aa,qs,down,up,hx,hz)
                dn,un = info[1].normcell,info[2].normcell
                norm_difference = abs(dn-un)/max(dn,un)
                valid = all(i->i.converged,info) && measured.imagmax<=1e-7 && measured.physical_excess<=1e-6 &&
                        norm_difference<=1e-6 && all(s->s.converged && max(s.right_residual,s.left_residual)<=1e-8,measured.strip_info)
                (;label,chi,seed,status=valid ? "converged_candidate" : "rejected",D,hx,hz,
                  source_sha256=bytes2hex(sha256(read(path))),adapter_error,info,norm_difference,measurement=measured)
            catch err
                (;label,chi,seed,status="failed",message=sprint(showerror,err))
            end
            push!(allrows,row)
            boundary_json(joinpath(out,"boundary_results.json"),allrows)
            println("boundary 结果：$(label) χ=$(chi) seed=$(seed) status=$(row.status)");flush(stdout)
        end
    end
    return allrows
end

function boundary_growth(out)
    isdir(out) && !isempty(readdir(out)) && error("growth output directory is not empty")
    mkpath(out)
    rows = NamedTuple[]
    for (label,subdir) in (("noise001","z-a0p001"),("noise01","z-a0p01"))
        path = joinpath(@__DIR__,"results","perturb-preflight01",subdir,"input.jld2")
        saved = load(path)
        x,D,hx,hz = Vector{Float64}(saved["parameters"]),Int(saved["D"]),Float64(saved["hx"]),Float64(saved["hz"])
        st = as_boundary_state(x,D)
        aa,qs,_ = BoundaryAudit.prepare_layers(st;balanced=true)
        initial = nothing
        for chi in (8,16,32)
            println("同一冻结态 boundary growth：$(label) χ=$(chi)");flush(stdout)
            down,up,info = BoundaryAudit.solve_boundaries(qs;chi,seed=1,tol=1e-9,maxiter=400,maxseconds=900,initial)
            measured = BoundaryAudit.measure_boundary(st,aa,qs,down,up,hx,hz)
            dn,un = info[1].normcell,info[2].normcell
            norm_difference=abs(dn-un)/max(dn,un)
            valid = all(i->i.converged,info) && norm_difference<=1e-6 && measured.imagmax<=1e-7 && measured.physical_excess<=1e-6
            row = (;label,chi,seed=1,status=valid ? "converged_candidate" : "rejected",
                     D,hx,hz,source_sha256=bytes2hex(sha256(read(path))),info,norm_difference,
                     initial_kind=chi==8 ? "random_seed1" : "grown_previous_boundary",measurement=measured)
            push!(rows,row)
            boundary_json(joinpath(out,"boundary_results.json"),rows)
            jldsave(joinpath(out,"$(label)_chi$(chi).jld2");down,up,chi,source_checkpoint=path)
            valid && (initial=(down,up))
            println("growth 结果：$(label) χ=$(chi) norm=$(dn) e=$(measured.energy) valid=$(valid)");flush(stdout)
        end
    end
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)>=1 || error("usage: boundary_check.jl OUTDIR [growth]")
    length(ARGS)>1 && ARGS[2]=="growth" ? boundary_growth(ARGS[1]) : boundary_checks(ARGS[1])
end

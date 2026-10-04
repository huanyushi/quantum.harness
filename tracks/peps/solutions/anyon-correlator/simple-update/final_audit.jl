# Focused follow-up to the path/loop difference and the exact-control CTMRG failure.
include("run.jl")

"Exact 2×2 periodic contraction, independent of CTMRG; not an infinite-size result."
function torus_stabilizers(st)
    @assert st.L == 2
    ts = [blocked_tensor(st,r,c) for r in 1:2 for c in 1:2]
    psi = zeros(256)
    for cfg in 0:255
        a,b,c,d,e,f,g,h = [1+((cfg>>k)&1) for k in 0:7]
        for p in 0:255
            p1,p2,p3,p4 = [1+((p>>(2k))&3) for k in 0:3]
            psi[p+1] += ts[1][p1,a,b,c,d]*ts[2][p2,e,d,f,b]*
                        ts[3][p3,c,g,a,h]*ts[4][p4,f,h,e,g]
        end
    end
    psi ./= norm(psi)
    ebit(r,c) = 2*(2*(mod1(r,2)-1)+mod1(c,2)-1)
    nbit(r,c) = ebit(r,c)+1
    vals = Float64[]
    for r in 1:2,c in 1:2
        amask = sum(1<<k for k in (ebit(r,c),nbit(r,c),ebit(r,c-1),nbit(r+1,c)))
        bmask = sum(1<<k for k in (ebit(r,c),nbit(r,c),ebit(r-1,c),nbit(r,c+1)))
        av = sum(psi[p+1]*psi[(p⊻amask)+1] for p in 0:255)
        bv = sum(psi[p+1]^2 * (iseven(count_ones(p&bmask)) ? 1 : -1) for p in 0:255)
        push!(vals,av,bv)
    end
    return vals
end

module LegacyReplay
using Random, LinearAlgebra, Printf
include("../scripts/tc_peps.jl")

function replay(;seed=1,nstep=600)
    Random.seed!(seed)
    v,pe,pn = randn(2,2,2,2),randn(2,2,2),randn(2,2,2)
    a = random_merged_tensor_VP(v,pe,pn,UPSPACE,uspace(2))
    psi = InfinitePEPS(fill(a,2,2))
    env = init_suweight(psi,uspace(2)); prev = deepcopy(env)
    alg = SimpleUpdate(;trunc=PEPSKit._get_fixedspacetrunc(psi),imaginary_time=true)
    circuit = build_su_circuit(0.05;P=UPSPACE)
    steps = 0; diff=Inf; t=time()
    for i in 1:nstep
        psi,env,err = PEPSKit.su_iter(psi,circuit,alg,env)
        steps=i
        if i%25 == 0
            diff = PEPSKit.compare_weights(prev,env)
            prev=deepcopy(env)
            @printf("[legacy replay] step=%d weight_delta=%.3e\n",i,diff);flush(stdout)
            diff<1e-10 && break
        end
    end
    return psi,(;steps,diff,seconds=time()-t)
end
end

function main_audit()
    out=joinpath(@__DIR__,"results","final-audit");mkpath(out)
    legacy_only = "--legacy-only" in ARGS
    rows = legacy_only ? TOML.parsefile(joinpath(out,"audit.toml")) : Dict{String,Any}()
    states = [("path_random","path-seed1","random"),
              ("loop_exact","loop-seed1","exact"),
              ("loop_random","loop-seed1","random")]
    legacy_only && empty!(states)
    for (name,folder,kind) in states
        chk = deserialize(joinpath(@__DIR__,"results",folder,"$(kind)-seed1.jls"))
        st = chk.state
        localres = local_stabilizer_residuals(st)
        finite = torus_stabilizers(st)
        rows[name] = Dict("step"=>chk.step,"tau"=>chk.tau,"local_tensor_residuals"=>localres,
                          "torus_stabilizers"=>finite,"torus_energy_per_spin"=>-sum(finite)/8)
        println("[$name] max local eigen residual=$(maximum(localres)) torus e=$(-sum(finite)/8)");flush(stdout)
        for chi in (8,16)
            m=measure_state(st;chi)
            rows[name]["chi$(chi)"] = Dict("energy_per_spin"=>m.energy,"stabilizers"=>m.vals,
                                           "residual"=>m.residual,"converged"=>m.converged,"maxerr"=>m.maxerr)
            println("[$name χ=$chi] e=$(m.energy) maxerr=$(m.maxerr) residual=$(m.residual) converged=$(m.converged)");flush(stdout)
            # Only retry a failed contraction; retain both outcomes in the record.
            if !m.converged || maximum(abs.(m.vals))>1+1e-6
                m2=measure_state(st;chi,envseed=17)
                rows[name]["chi$(chi)_seed17"] = Dict("energy_per_spin"=>m2.energy,"stabilizers"=>m2.vals,
                                           "residual"=>m2.residual,"converged"=>m2.converged,"maxerr"=>m2.maxerr)
                println("[$name χ=$chi envseed17] e=$(m2.energy) maxerr=$(m2.maxerr) residual=$(m2.residual) converged=$(m2.converged)");flush(stdout)
            end
            open(joinpath(out,"audit.toml"),"w") do io;TOML.print(io,rows);end
        end
    end
    psi,meta = LegacyReplay.replay()
    rows["legacy_replay"] = Dict{String,Any}("steps"=>meta.steps,"weight_delta"=>meta.diff,"su_seconds"=>meta.seconds)
    for chi in (8,16)
        m=measure_peps(psi,2;chi)
        rows["legacy_replay"]["chi$(chi)"] = Dict("energy_per_spin"=>m.energy,"stabilizers"=>m.vals,
                                               "residual"=>m.residual,"converged"=>m.converged)
        println("[legacy χ=$chi] e=$(m.energy) residual=$(m.residual) converged=$(m.converged)");flush(stdout)
    end
    open(joinpath(out,"audit.toml"),"w") do io;TOML.print(io,rows);end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_audit()
end

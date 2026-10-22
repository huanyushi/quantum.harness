using LinearAlgebra, Printf, Serialization, TOML, Dates
BLAS.set_num_threads(1)
include("core.jl")
using .SplitSU
include("measure.jl")

function options(args)
    d = Dict("init"=>"both", "seed"=>"1", "schedule"=>"0.05:200,0.01:200,0.002:200",
             "chi"=>"8", "measure-every"=>"20", "out"=>joinpath(@__DIR__,"results","pilot"),
             "max-seconds"=>"480", "rotate"=>"true", "resume"=>"",
             "plaquette"=>"path", "als-sweeps"=>"12")
    for arg in args
        k,v = split(replace(arg,r"^--"=>""),'=';limit=2)
        haskey(d,k) || error("Unknown option $k")
        d[k] = v
    end
    return d
end

function record!(io, st, kind, seed, step, tau, dt, es, ep, dw, elapsed; chi)
    m = measure_state(st;chi)
    @printf(io,"%s,%d,%d,%.12g,%.12g,%.16g,%.6g,%.6g,%.6g,%d,%.6g,%s,%d,%.6f,%.6f",
            kind,seed,step,tau,dt,m.energy,es,ep,dw,chi,m.residual,string(m.converged),
            m.iterations,elapsed,m.seconds)
    for v in m.vals
        @printf(io,",%.16g",v)
    end
    println(io); flush(io)
    @printf("[%s seed=%d] step=%d τ=%.3f dt=%.3g e=%+.12f max|stab−1|=%.3e CTM=%.2e (%s, %.2fs) local_error(A/B)=%.2e/%.2e\n",
            kind,seed,step,tau,dt,m.energy,m.maxerr,m.residual,string(m.converged),m.seconds,es,ep)
    flush(stdout)
    return m
end

function run_one(opts, kind, schedule)
    seed,chi = parse(Int,opts["seed"]),parse(Int,opts["chi"])
    every = parse(Int,opts["measure-every"])
    limit = parse(Float64,opts["max-seconds"])
    rotate = parse(Bool,opts["rotate"])
    method = Symbol(opts["plaquette"])
    method in (:path,:loop) || error("plaquette must be path or loop")
    als_sweeps = parse(Int,opts["als-sweeps"])
    out = opts["out"]; mkpath(out)
    if isempty(opts["resume"])
        st = initial_state(Symbol(kind),seed)
        step = 0; tau = 0.0
    else
        chk = deserialize(opts["resume"])
        st,step,tau = chk.state,chk.step,chk.tau
    end
    csvpath = joinpath(out,"$(kind)-seed$(seed).csv")
    checkpoint = joinpath(out,"$(kind)-seed$(seed).jls")
    t = time(); su_seconds = 0.0
    println("START $kind seed=$seed D=2 d=2 χ=$chi plaquette=$method path_rotation=$rotate Julia=$(VERSION)");flush(stdout)
    status = "finished_schedule"
    open(csvpath,"w") do io
        println(io,"init,seed,step,tau,dt,energy_per_spin,discard_star,discard_plaquette,weight_delta,chi,ctm_residual,ctm_converged,ctm_iterations,elapsed_before_measure,ctm_seconds,A11,B11,A12,B12,A21,B21,A22,B22")
        record!(io,st,kind,seed,step,tau,0.0,0.0,0.0,0.0,time()-t;chi)
        serialize(checkpoint,(;state=st,step,tau))
        for (dt,nstep) in schedule
            for j in 1:nstep
                old = state_weights(st)
                t0 = time()
                step += 1; tau += dt
                es,ep = sweep!(st,dt,step;rotate,plaquette=method,als_sweeps)
                su_seconds += time()-t0
                dw = maximum(abs.(old-state_weights(st)))
                if j == nstep || step % every == 0 || time()-t > limit
                    m = record!(io,st,kind,seed,step,tau,dt,es,ep,dw,time()-t;chi)
                    serialize(checkpoint,(;state=st,step,tau))
                    if kind == "random" && m.converged && m.maxerr < 1e-7 && abs(m.energy+1) < 1e-7
                        status = "ground_state_at_chi$(chi)"
                        break
                    end
                    if time()-t > limit
                        status = "time_limit"
                        break
                    end
                end
            end
            status != "finished_schedule" && break
        end
    end
    open(joinpath(out,"$(kind)-seed$(seed)-summary.toml"),"w") do io
        TOML.print(io,Dict("status"=>status,"steps"=>step,"tau"=>tau,
                          "su_seconds"=>su_seconds,"wall_seconds"=>time()-t,
                          "julia"=>string(VERSION),"pepskit"=>string(pkgversion(PEPSKit)),
                          "tensorkit"=>string(pkgversion(TensorKit))))
    end
    println("END $kind status=$status SU seconds=$su_seconds wall=$(time()-t)");flush(stdout)
end

function main(args=ARGS)
    opts = options(args)
    mkpath(opts["out"])
    open(joinpath(opts["out"],"settings.toml"),"w") do io
        TOML.print(io,merge(opts,Dict("D"=>"2","d"=>"2","Je"=>"1","Jm"=>"1","hx"=>"0","hz"=>"0",
                                    "timestamp"=>string(now()),"pinv_rtol"=>"1e-14")))
    end
    schedule = [(parse(Float64,split(s,':')[1]),parse(Int,split(s,':')[2])) for s in split(opts["schedule"],',')]
    if opts["init"] == "both"
        run_one(opts,"exact",[(0.05,8)])
        run_one(opts,"random",schedule)
    else
        run_one(opts,opts["init"],schedule)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

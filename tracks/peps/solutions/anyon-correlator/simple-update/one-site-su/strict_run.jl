module StrictZeroRun
const PROCESS_START=time()
using LinearAlgebra, Printf, JSON, Serialization, SHA, Dates
BLAS.set_num_threads(1)
include("strict.jl")
const S=StrictOneSiteSU
const blocked_tensor=S.K.blocked_tensor
# Identical operators and infinite-plane CTMRG measurements to the old run.
include("../split/measure.jl")

function write_json(path,data)
    temp=path*".tmp"
    open(temp,"w") do io;JSON.print(io,data);println(io);flush(io);end
    mv(temp,path;force=true)
end
function prepare_output(out)
    ispath(out) && throw(ArgumentError("output already exists: $out"))
    mkpath(out)
end
function save_state(path,st;seed,step)
    write_json(path,(;format="strict-one-site-zero-v1",D=2,V=vec(st.V),P=vec(st.P),lambda=st.lambda,
              metadata=(;seed,step,symmetry_defect=S.defect(st),virtual_Z2_imposed=false,
                         independent_tensors="one V, one P, one lambda")))
end
function load_state(path)
    data=JSON.parsefile(path)
    data["format"]=="strict-one-site-zero-v1" && data["D"]==2 || throw(ArgumentError("invalid state format"))
    st=S.State(reshape(Float64.(data["V"]),2,2,2,2),reshape(Float64.(data["P"]),2,2,2),Float64.(data["lambda"]))
    all(isfinite,st.V) && all(isfinite,st.P) && all(x->isfinite(x)&&x>0,st.lambda) && S.defect(st)<1e-13 ||
        throw(ArgumentError("invalid or nonsymmetric state"))
    return st,data["metadata"]
end
function audit(ms,st,status)
    checks=[!isnothing(m) && all(isfinite,(m.energy,m.residual,m.maxerr)) &&
            all(isfinite,m.vals) && m.converged && m.residual<=1e-9 && abs(m.energy+1)<=1e-6 &&
            maximum(abs.(m.vals.-1))<=1e-6 && maximum(abs.(m.vals))<=1+1e-6 for m in ms]
    valid=[m for m in ms if !isnothing(m)]
    distinct_chi=length(unique(m.chi for m in valid))>=2
    spread=isempty(valid) ? Inf : maximum(abs(m.energy-first(valid).energy) for m in valid)
    symmetry=S.defect(st)
    return (;passed=status=="finished" && length(ms)>=2 && all(checks) && distinct_chi && spread<=1e-6 && symmetry<1e-13,
             checks,distinct_chi,energy_difference=isfinite(spread) ? spread : nothing,symmetry_defect=symmetry)
end
function options(args)
    d=Dict("seed"=>"5","steps"=>"700","dt"=>"0.05","als-sweeps"=>"12","every"=>"40",
           "measure-every"=>"100","budget"=>"480","out"=>joinpath(@__DIR__,"results","strict-zero-seed5"))
    for arg in args
        kv=split(replace(arg,r"^--"=>""),'=';limit=2)
        length(kv)==2 && haskey(d,kv[1]) || throw(ArgumentError("unknown option: $arg"))
        d[kv[1]]=kv[2]
    end
    return d
end
function main(args=ARGS)
    opts=options(args)
    seed=parse(Int,opts["seed"]);steps=parse(Int,opts["steps"]);dt=parse(Float64,opts["dt"])
    als=parse(Int,opts["als-sweeps"]);every=parse(Int,opts["every"]);mevery=parse(Int,opts["measure-every"])
    budget=parse(Float64,opts["budget"]);out=opts["out"]
    0<budget<=600 && steps>=0 && isfinite(dt) && dt>=0 && als>0 && every>0 && mevery>0 ||
        throw(ArgumentError("invalid steps, time step, cadence or budget"))
    deadline=PROCESS_START+budget
    prepare_output(out)
    st=S.random_state(seed);step=0;status="finished";start=time();update_seconds=0.0
    max_symmetry=0.0;max_gauge=0.0;finals=Any[]
    sources=[joinpath(@__DIR__,f) for f in ("strict.jl","core.jl","strict_run.jl")]
    append!(sources,[joinpath(@__DIR__,"..","split",f) for f in ("core.jl","loop.jl","measure.jl")])
    write_json(joinpath(out,"settings.json"),(;options=opts,timestamp=string(now()),julia=string(VERSION),
        model="H=-sum(A)-sum(B), Pauli J=1, infinite square edge spins, energy/edge spin",
        initialization="dense Gaussian random, C4 and endpoint projection only; seed selected from recorded bounded pilot",
        storage="only State(V,P,lambda); local SU scratch and measurement replicas discarded",
        source_sha256=Dict(relpath(f,@__DIR__)=>bytes2hex(sha256(read(f))) for f in sources)))
    save_state(joinpath(out,"initial.json"),st;seed,step)
    function checkpoint()
        save_state(joinpath(out,"latest.json"),st;seed,step)
        serialize(joinpath(out,"latest.jls"),(;state=st,seed,step,dt))
    end
    function logrow(row)
        open(joinpath(out,"trajectory.jsonl"),"a") do io;JSON.print(io,row);println(io);flush(io);end
    end
    function measure(ch)
        time()<deadline || return nothing
        checkpoint()
        m=try
            measure_state(S.workspace(st);chi=ch,tol=1e-10,maxiter=400,envseed=2026)
        catch err
            logrow((;event="measurement_error",step,chi=ch,message=sprint(showerror,err),seconds=time()-start))
            return nothing
        end
        logrow((;event="measurement",step,measurement=m,seconds=time()-start))
        @printf("MEASURE seed=%d step=%d chi=%d e=%+.12f max|stab-1|=%.3e CTM=%.2e/%s\n",
                seed,step,ch,m.energy,m.maxerr,m.residual,string(m.converged));flush(stdout)
        return m
    end
    println("START strict single V/P/lambda seed=$seed D=2 h=0; no virtual Z2; steps=$steps dt=$dt");flush(stdout)
    checkpoint();measure(8)
    for i in 1:steps
        if time()>=deadline;status="budget";break;end
        t=time()
        diag=try
            S.sweep!(st,dt;als_sweeps=als)
        catch err
            status="update_error"
            logrow((;event="update_error",step,message=sprint(showerror,err),seconds=time()-start))
            break
        end
        update_seconds+=time()-t;step=i
        max_symmetry=max(max_symmetry,diag.star.symmetry_defect,diag.plaquette.symmetry_defect)
        max_gauge=max(max_gauge,diag.star.gauge_error,diag.plaquette.gauge_error)
        max_symmetry<1e-13 || error("lost spatial symmetry")
        logrow((;event="update",step,dt,diagnostics=diag,seconds=time()-start))
        checkpoint()
        if step%every==0 || step==steps
            r=S.local_residuals(st)
            logrow((;event="local_residual",step,residuals=r,symmetry=S.defect(st),seconds=time()-start))
            @printf("STEP %d A/B residual=(%.3e,%.3e) symmetry=%.1e gauge=%.1e ALS fit=%.2e wall=%.1fs\n",
                    step,r.star,r.plaquette,max_symmetry,max_gauge,diag.plaquette.fit,time()-start);flush(stdout)
        end
        step%mevery==0 && measure(8)
    end
    checkpoint()
    for ch in (8,16)
        m=measure(ch);push!(finals,m)
        time()>=deadline && (status="budget")
    end
    result=audit(finals,st,status)
    save_state(joinpath(out,"final.json"),st;seed,step)
    serialize(joinpath(out,"final.jls"),(;state=st,seed,step,dt))
    summary=(;status,passed=result.passed,audit=result,seed,D=2,step,dt,measurements=finals,
              local_residuals=S.local_residuals(st),max_symmetry_defect=max_symmetry,max_gauge_error=max_gauge,
              strict_one_site_implemented=true,virtual_Z2_imposed=false,update_seconds,
              wall_seconds=time()-start,process_seconds=time()-PROCESS_START,peak_rss_bytes=Sys.maxrss())
    write_json(joinpath(out,"summary.json"),summary)
    println("END status=$status passed=$(result.passed) strict_one_site=true max_symmetry=$max_symmetry process=$(summary.process_seconds)s");flush(stdout)
    return summary
end
end
if abspath(PROGRAM_FILE)==@__FILE__;StrictZeroRun.main();end

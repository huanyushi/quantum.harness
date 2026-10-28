module Z2ZeroRun
const PROCESS_START=time()
include("core.jl")
using .OneSiteZ2SU, LinearAlgebra, JSON, SHA, Printf
const S=OneSiteZ2SU
include("measure.jl")
function worker_main(args=ARGS)
    opts=Dict("seed"=>"5","steps"=>"700","dt"=>"0.05","als-sweeps"=>"12","budget"=>"600","resume"=>"","out"=>joinpath(@__DIR__,"results","zero-z2-seed5"))
    for arg in args
        kv=split(replace(arg,r"^--"=>""),'=';limit=2)
        length(kv)==2 && haskey(opts,kv[1]) || throw(ArgumentError("invalid option $arg"))
        opts[kv[1]]=kv[2]
    end
    seed=parse(Int,opts["seed"]);steps=parse(Int,opts["steps"]);dt=parse(Float64,opts["dt"])
    als=parse(Int,opts["als-sweeps"]);budget=parse(Float64,opts["budget"]);out=opts["out"]
    0<budget<=600 && steps>=0 && isfinite(dt) && dt>=0 && als>0 || throw(ArgumentError("invalid settings"))
    S.prepare_output(out)
    st,start_step=isempty(opts["resume"]) ? (random_state(seed),0) : S.resume_state(opts["resume"];seed,dt,als_sweeps=als)
    step=start_step;status="finished";deadline=PROCESS_START+budget
    parent_hash=isempty(opts["resume"]) ? nothing : bytes2hex(sha256(read(opts["resume"])))
    sources=[joinpath(@__DIR__,f) for f in ("core.jl","tensors.jl","shared.jl","star.jl","plaquette.jl","state_io.jl","measure.jl","run.jl")]
    source_hashes=Dict(basename(f)=>bytes2hex(sha256(read(f))) for f in sources)
    settings=(;opts,model="H=-sum(A)-sum(B); Pauli; Je=Jm=1; infinite square; edge spins; energy per edge",D=2,
        constraints="one V/P/lambda; C4 V; symmetric P; horizontal=vertical P; native virtual Z2",start_step,parent_hash,julia=string(VERSION),tensorkit=string(pkgversion(S.TensorKit)),source_hashes)
    S.write_json(joinpath(out,"settings.json"),settings)
    metadata()=(;origin="numerical_random_optimization",seed,step,dt,als_sweeps=als,accepted=false,parent_hash,source_hashes)
    save_state(joinpath(out,"initial.json"),st;metadata=metadata())
    function checkpoint()
        p=joinpath(out,"latest.json"); tmp=joinpath(out,"checkpoint.json")
        save_state(tmp,st;metadata=metadata());mv(tmp,p;force=true)
    end
    function logrow(row)
        open(joinpath(out,"trajectory.jsonl"),"a") do io; JSON.print(io,row);println(io);flush(io);end
    end
    function measure(chi)
        time()<deadline || return nothing
        checkpoint()
        m=measure_state(st;chi)
        logrow((;event="measurement",step,m))
        @printf("MEASURE step=%d chi=%d e=%.12f stab_error=%.3e CTM=%.3e converged=%s\n",step,chi,m.energy,m.maxerr,m.residual,string(m.converged));flush(stdout)
        m
    end
    println("START seed=$seed start_step=$start_step additional_steps=$steps dt=$dt; native virtual Z2, one V/P/lambda; budget=$budget seconds");flush(stdout)
    checkpoint(); update_seconds=0.0
    try
        for i in 1:steps
            if time()>=deadline;status="budget";break;end
            t=time();diag=sweep!(st,dt;als_sweeps=als);update_seconds+=time()-t;step=start_step+i
            checkpoint();logrow((;event="update",step,diag,seconds=time()-PROCESS_START))
            if i==1 || step%40==0 || i==steps
                r=local_residuals(st);logrow((;event="residual",step,r))
                @printf("STEP %d A/B=(%.3e,%.3e) fit=%.3e wall=%.1fs update=%.1fs\n",step,r.star,r.plaquette,diag.plaquette.fit_residual,time()-PROCESS_START,update_seconds);flush(stdout)
            end
            step%100==0 && measure(8)
        end
        finals=[measure(chi) for chi in (8,16)]
        time()>=deadline && (status="budget")
        result=S.audit(finals,st,status)
        save_state(joinpath(out,"final.json"),st;metadata=merge(metadata(),(;accepted=result.passed)))
        S.write_json(joinpath(out,"summary.json"),(;status,passed=result.passed,audit=result,step,measurements=finals,update_seconds,process_seconds=time()-PROCESS_START,peak_rss_bytes=Sys.maxrss()))
        println("END status=$status passed=$(result.passed)");flush(stdout)
        return result
    catch err
        checkpoint()
        S.write_json(joinpath(out,"summary.json"),(;status="error",passed=false,step,error=sprint(showerror,err)))
        rethrow()
    end
end
function main(args=ARGS)
    "--worker" in args && return worker_main(filter(!=("--worker"),args))
    budget=600.0; out=joinpath(@__DIR__,"results","zero-z2-seed5")
    for a in args
        startswith(a,"--budget=") && (budget=parse(Float64,split(a,'=';limit=2)[2]))
        startswith(a,"--out=") && (out=split(a,'=';limit=2)[2])
    end
    0<budget<=600 || throw(ArgumentError("budget must be in (0,600]"))
    ispath(out) && throw(ArgumentError("output already exists: $out"))
    project=dirname(Base.active_project())
    cmd=`$(Base.julia_cmd()) --startup-file=no -O1 --project=$project $(@__FILE__) --worker $args`
    # Absolute deadline counts imports and supervisor JIT; leave time to save
    # the budget marker before an external command timeout at the same cap.
    deadline=PROCESS_START+budget-min(10.,budget/10)
    remaining=deadline-time()
    remaining>0 || error("budget exhausted during imports")
    status=S.supervise(cmd,out;budget=remaining,deadline)
    println("SUPERVISOR status=$status");flush(stdout)
    return status
end
end
if abspath(PROGRAM_FILE)==@__FILE__; Z2ZeroRun.main(); end

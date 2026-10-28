function write_json(path,data)
    open(path*".tmp","w") do io; JSON.print(io,data); println(io); flush(io); end
    mv(path*".tmp",path;force=true)
end
function prepare_output(out)
    ispath(out) && throw(ArgumentError("output already exists: $out"))
    mkpath(out)
end
"Hard outer wall limit: a child cannot overrun inside compilation, ALS or CTMRG."
function supervise(cmd,out;budget,deadline=Inf,output=stdout,error_output=stderr)
    t=time(); deadline=isfinite(deadline) ? deadline : t+budget
    function budget_summary()
        mkpath(out)
        write_json(joinpath(out,"summary.json"),(;status="budget",passed=false,wall_seconds=time()-t,
            checkpoint=isfile(joinpath(out,"latest.json")) ? "latest.json" : nothing))
        return "budget"
    end
    time()<deadline || return budget_summary()
    p=run(pipeline(cmd;stdout=output,stderr=error_output);wait=false)
    result=timedwait(()->process_exited(p),max(0.,deadline-time());pollint=min(0.01,budget))
    if result==:timed_out
        kill(p,Base.SIGKILL);wait(p)
        return budget_summary()
    end
    return success(p) ? "finished" : "error"
end
function save_state(path,st;metadata)
    ispath(path) && throw(ArgumentError("state already exists: $path"))
    a=dense_arrays(st)
    data=(;format="one-site-z2-v1",D=2,charges=[0,1],physical_charges=[0,0],
        V_legs=["N","E","S","W"],P_legs=["physical","head","tail"],
        partitions=["N E <- S W","physical head <- tail"],
        normalization="unit Frobenius V,P,lambda; lambda nonnegative",
        V=vec(a.V),P=vec(a.P),lambda=a.lambda,metadata)
    write_json(path,data)
end
function load_state(path)
    d=JSON.parsefile(path)
    get(d,"format",nothing)=="one-site-z2-v1" || throw(ArgumentError("invalid state format"))
    d["D"]==2 && d["charges"]==[0,1] && d["physical_charges"]==[0,0] &&
        d["V_legs"]==["N","E","S","W"] && d["P_legs"]==["physical","head","tail"] &&
        d["partitions"]==["N E <- S W","physical head <- tail"] || throw(ArgumentError("invalid state convention"))
    length(d["V"])==16 && length(d["P"])==8 && length(d["lambda"])==2 || throw(ArgumentError("invalid state dimension"))
    v=reshape(Float64.(d["V"]),2,2,2,2); p=reshape(Float64.(d["P"]),2,2,2); l=Float64.(d["lambda"])
    all(isfinite,v) && all(isfinite,p) && all(isfinite,l) && all(>=(0),l) || throw(ArgumentError("invalid state values"))
    q=virtual_space(); h=physical_space()
    st=Z2State(TensorMap(v,q⊗q←q⊗q;tol=0),TensorMap(p,h⊗q←q;tol=0),DiagonalTensorMap(TensorMap(Matrix(Diagonal(l)),q←q)))
    restored=dense_arrays(st)
    restored.V==v && restored.P==p && restored.lambda==l || throw(ArgumentError("charge-forbidden input component"))
    maximum(values(symmetry_residuals(st)))<=1e-12 && all(x->abs(norm(x)-1)<=1e-12,(st.V,st.P,st.lambda)) || throw(ArgumentError("invalid state symmetry or normalization"))
    return st,d["metadata"]
end
function audit(ms,st,status)
    r=local_residuals(st)
    checks=[m!==nothing && all(isfinite,(m.energy,m.residual)) && all(isfinite,m.vals) &&
        m.converged && m.residual<=1e-9 && abs(m.energy+1)<=1e-6 && maximum(abs.(m.vals.-1))<=1e-6 for m in ms]
    valid=filter(!isnothing,ms)
    distinct=length(unique(m.chi for m in valid))>=2
    spread=isempty(valid) ? Inf : maximum(abs(m.energy-first(valid).energy) for m in valid)
    passed=status=="finished" && length(ms)>=2 && all(checks) && distinct && spread<=1e-6 &&
        maximum(values(symmetry_residuals(st)))<=1e-12 && all(x->isfinite(x)&&x<=1e-6,values(r))
    return (;passed,checks,distinct_chi=distinct,energy_difference=isfinite(spread) ? spread : nothing,local_residuals=r)
end
function resume_state(path;seed,dt,als_sweeps)
    st,md=load_state(path)
    get(md,"origin",nothing)=="numerical_random_optimization" &&
        get(md,"seed",nothing)==seed && get(md,"dt",nothing)==dt &&
        get(md,"als_sweeps",nothing)==als_sweeps && get(md,"step",-1)>=0 ||
        throw(ArgumentError("resume requires the same numerical origin, seed and update settings"))
    return st,Int(md["step"])
end

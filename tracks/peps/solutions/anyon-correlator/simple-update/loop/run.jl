using LinearAlgebra, Printf, Serialization, TOML, Dates, JSON
BLAS.set_num_threads(1)
include("core.jl")
using .FieldLoop
include("measure.jl")

const EMPTY_DIAG=(star_error=0.0,fit_error=0.0,gauge_error=0.0,bp_residual=0.0,als_iterations=0,bp_iterations=0,als_capped=0,bp_capped=0,fit_increase=0.0)
const CSV_KEYS=[:step,:tau,:dt,:D,:hx,:hz,:energy,:mx,:mz,:astar,:bplaq,:sector_error,:site_spread,:physical_excess,
                :chi,:ctm_converged,:ctm_residual,:ctm_seconds,:weight_delta,:star_error,:fit_error,:gauge_error,
                :bp_residual,:als_iterations,:bp_iterations,:als_capped,:bp_capped,:fit_increase,
                :energy_error,:mx_error,:mz_error,:elapsed]
function options(args)
    d=Dict("D"=>"2","hx"=>"0.1","hz"=>"0.1","seed"=>"1","init"=>"random","chi"=>"8",
           "schedule"=>"0.05:200,0.01:200,0.002:200","als-sweeps"=>"12","every"=>"40",
           "out"=>joinpath(@__DIR__,"results","pilot"),"resume"=>"","max-seconds"=>"480",
           "final-chis"=>"8,16","grow-noise"=>"0.001","envseed"=>"2026")
    for arg in args
        k,v=split(replace(arg,r"^--"=>""),'=';limit=2);haskey(d,k)||error("Unknown option $k");d[k]=v
    end
    return d
end
function measure_record!(io,jsonio,st,hx,hz,step,tau,dt,diag,weight_delta,start;chi,envseed)
    m=measure_state(st,hx,hz;chi,envseed)
    ref=reference_values(hx,hz)
    row=(;step,tau,dt,D=st.D,hx,hz,m...,weight_delta,diag...,energy_error=m.energy-ref.energy,
          mx_error=m.mx-ref.mx,mz_error=m.mz-ref.mz,elapsed=time()-start)
    println(io,join([getproperty(row,k) for k in CSV_KEYS],','));flush(io)
    JSON.print(jsonio,row);println(jsonio);flush(jsonio)
    @printf("[D%d h=(%.3f,%.3f)] step=%d τ=%.3f e=%+.10f mx=%.8f mz=%.8f Δm=(%+.2e,%+.2e) CTM=%.1e/%s ALS=%d BP=%.1e gauge=%.1e t=%.1fs\n",
            st.D,hx,hz,step,tau,m.energy,m.mx,m.mz,row.mx_error,row.mz_error,m.ctm_residual,string(m.ctm_converged),diag.als_iterations,diag.bp_residual,diag.gauge_error,row.elapsed)
    flush(stdout)
    return row
end
function run_one(opts)
    D=parse(Int,opts["D"]);hx=parse(Float64,opts["hx"]);hz=parse(Float64,opts["hz"])
    chi=parse(Int,opts["chi"]);seed=parse(Int,opts["seed"]);every=parse(Int,opts["every"])
    als=parse(Int,opts["als-sweeps"]);limit=parse(Float64,opts["max-seconds"]);envseed=parse(Int,opts["envseed"])
    out=opts["out"];mkpath(out)
    if isempty(opts["resume"])
        st=initial_state(Symbol(opts["init"]),seed;D);step=0;tau=0.0
    else
        chk=deserialize(opts["resume"]);st=grow_state(chk.state,D;noise=parse(Float64,opts["grow-noise"]));step=chk.step;tau=chk.tau
        @assert chk.hx==hx && chk.hz==hz "Resume must use same fields"
    end
    open(joinpath(out,"settings.toml"),"w") do io
        TOML.print(io,merge(opts,Dict("J"=>"1","d"=>"2","timestamp"=>string(now()),"julia"=>string(VERSION),"pepskit"=>string(pkgversion(PEPSKit)))))
    end
    open(joinpath(out,"reference.json"),"w") do io
        JSON.print(io,Dict("source"=>"arXiv:1012.1740v2 supplement Eq.(1), J=1 conversion","hx"=>hx,"hz"=>hz,
                           "order10"=>reference_values(hx,hz),"order8"=>reference_values(hx,hz;order=8)))
    end
    schedule=[(parse(Float64,split(s,':')[1]),parse(Int,split(s,':')[2])) for s in split(opts["schedule"],',')]
    println("START D=$D fields=($hx,$hz) init=$(opts["init"]) resume=$(opts["resume"]) ALS=$als χ=$chi");flush(stdout)
    start=time();su_seconds=0.0;status="finished_schedule"
    diag=EMPTY_DIAG;dw=0.0;lastdt=0.0
    open(joinpath(out,"trajectory.csv"),"w") do io
        open(joinpath(out,"measurements.jsonl"),"w") do jsonio
            println(io,join(CSV_KEYS,','));flush(io)
            measure_record!(io,jsonio,st,hx,hz,step,tau,0.0,diag,dw,start;chi,envseed)
            serialize(joinpath(out,"state.jls"),(;state=st,step,tau,hx,hz))
            for (dt,nstep) in schedule
                for j in 1:nstep
                    old=state_weights(st);t0=time();step+=1;tau+=dt;lastdt=dt
                    diag=sweep!(st,hx,hz,dt;als_sweeps=als);su_seconds+=time()-t0
                    dw=maximum(abs.(old-state_weights(st)))
                    if j==nstep || step%every==0 || time()-start>limit
                        measure_record!(io,jsonio,st,hx,hz,step,tau,dt,diag,dw,start;chi,envseed)
                        serialize(joinpath(out,"state.jls"),(;state=st,step,tau,hx,hz))
                    end
                    if time()-start>limit;status="time_limit";break;end
                end
                status=="time_limit" && break
            end
        end
    end
    summary=Dict{String,Any}("status"=>status,"steps"=>step,"tau"=>tau,"last_dt"=>lastdt,
                             "su_seconds"=>su_seconds,"wall_seconds"=>time()-start,"final_audit_complete"=>false)
    open(joinpath(out,"summary.toml"),"w") do io;TOML.print(io,summary);end
    finals=[]
    for ch in parse.(Int,split(opts["final-chis"],','))
        m=measure_state(st,hx,hz;chi=ch,envseed)
        push!(finals,m)
        @printf("FINAL χ=%d e=%.12f mx=%.10f mz=%.10f CTM=%.2e/%s\n",ch,m.energy,m.mx,m.mz,m.ctm_residual,string(m.ctm_converged));flush(stdout)
        if !m.ctm_converged || m.physical_excess>1e-6
            retry=measure_state(st,hx,hz;chi=ch,envseed=17);push!(finals,retry)
            @printf("RETRY χ=%d seed17 e=%.12f mx=%.10f mz=%.10f CTM=%.2e/%s\n",ch,retry.energy,retry.mx,retry.mz,retry.ctm_residual,string(retry.ctm_converged));flush(stdout)
        end
        open(joinpath(out,"final.json"),"w") do io;JSON.print(io,finals);end
    end
    summary["wall_seconds"]=time()-start;summary["final_audit_complete"]=true
    open(joinpath(out,"summary.toml"),"w") do io;TOML.print(io,summary);end
    println("END status=$status SU=$(su_seconds)s wall=$(time()-start)s");flush(stdout)
    return st
end
main(args=ARGS)=run_one(options(args))
if abspath(PROGRAM_FILE)==@__FILE__;main();end

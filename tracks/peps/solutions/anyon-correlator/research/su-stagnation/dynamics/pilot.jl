include("core.jl")
AC.BLAS.set_num_threads(1)
const OUTPUT=joinpath(@__DIR__,"results","pilot")
mkpath(OUTPUT)
const RECORD=Dict{String,Any}("setup"=>Dict("D"=>2,"seed"=>1,"hx"=>0,"hz"=>0,"J"=>1,"julia"=>string(VERSION),"finite_boundary"=>"2x2 torus"))
function persist!()
    open(joinpath(OUTPUT,"pilot.json"),"w") do io
        JSON.print(io,RECORD,2)
    end
end
flat(K)=[K[r,c] for r in 1:2 for c in 1:2]
function probe(psi,w,kind,dt)
    before=AC.wave(psi);fb=fit_filter_state(before)
    after,_,err=AC.step!(deepcopy(psi),deepcopy(w),make_alg(psi),kind,1,1;dt)
    v=AC.wave(after);fa=fit_filter_state(v)
    phase=dot(before,v);aligned=v*conj(phase)/abs(phase)
    velocity=(aligned-before)/dt
    velocity-=before*dot(before,velocity)
    target=normalize(cosh(dt)*before+sinh(dt)*AC.action(before,kind,1,1))
    row=Dict("kind"=>String(kind),"dt"=>dt,"before"=>AC.vals(before),"after"=>AC.vals(v),
        "exact"=>AC.vals(target),"velocity_norm"=>norm(velocity),"library_error"=>Float64(err),
        "infidelity"=>max(0.,1-abs2(dot(target,v))),"K_before"=>flat(fb.K),"K_after"=>flat(fa.K),
        "fit_support"=>fa.support,"fit_residual"=>fa.residual)
    @printf("probe %s dt=%.4g velocity=%.5g fit=%.2e support=%.6f ΔK=%s\n",kind,dt,norm(velocity),fa.residual,fa.support,string(round.(flat(fa.K-fb.K);digits=7)));flush(stdout)
    return row
end
function main()
    started=time()
    stalled=legacy_state()
    RECORD["legacy"]=[probe(stalled.psi,stalled.w,:A,dt) for dt in (0.05,0.02,0.01,0.005,0.002,0.001)]
    persist!()
    for (label,K) in (("uniform02",fill(0.2,2,2)),("uniform035",fill(0.35,2,2)),
                       ("uniform06",fill(0.6,2,2)),("nonuniform",[0.13 0.25;0.35 0.42]))
        raw=filtered_plaquette_state(K)
        psi,w,meta=identity_gauge(raw)
        println("family $label identity gauge: $meta");flush(stdout)
        rows=Any[];RECORD[label]=Dict("K"=>flat(K),"identity_gauge"=>Dict(pairs(meta)),"probes"=>rows)
        for dt in (0.05,0.02,0.01,0.005,0.002,0.001)
            push!(rows,probe(psi,w,:A,dt));persist!()
        end
        push!(rows,probe(psi,w,:B,0.05));persist!()
        p,q=deepcopy(psi),deepcopy(w);cycle=Any[]
        for r in 1:2,c in 1:2,k in (:A,:B)
            p,q,_=AC.step!(p,q,make_alg(p),k,r,c;dt=0.05)
            f=fit_filter_state(AC.wave(p))
            push!(cycle,Dict("kind"=>String(k),"r"=>r,"c"=>c,"K"=>flat(f.K),"support"=>f.support,"residual"=>f.residual))
        end
        RECORD[label]["cycle"]=cycle
        serialize(joinpath(OUTPUT,label*".jls"),(;psi,w,K))
        persist!()
    end
    RECORD["wall_seconds"]=time()-started;persist!()
    println("pilot completed in $(time()-started) seconds");flush(stdout)
end
main()

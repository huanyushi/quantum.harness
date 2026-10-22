include("core.jl")
using Test
AC.BLAS.set_num_threads(1)
const OUTDIR=joinpath(@__DIR__,"results","followup")
mkpath(OUTDIR)
const DATA=Dict{String,Any}()
flat(K)=[K[r,c] for r in 1:2 for c in 1:2]
function save_data!()
    open(joinpath(OUTDIR,"followup.json"),"w") do io;JSON.print(io,DATA,2);end
end
function reduced_sweep(K,dt)
    K=copy(K)
    for r in 1:2,c in 1:2
        a,b=mod1(r+1,2),mod1(c-1,2)
        K[a,b]=star_filter_map(K[a,b],dt)
        K[r,c]+=dt
    end
    K
end
function normalized_star_spectra(psi,w,dt)
    sites=[CartesianIndex(1,0),CartesianIndex(1,1),CartesianIndex(2,1)]
    ms,_,_=AC.PEPSKit._get_cluster_with_weights(psi,sites,w)
    flips=[AC.isdual(AC.space(m,1)) for m in ms[2:end]]
    AC.PEPSKit._flip_virtuals!(ms,flips)
    gate=AC.PEPSKit.gate_to_mpo(AC.pauli_gate3(AC.phys_tmap(AC.STAR_A_mat(),3,AC.UPSPACE),dt,1.0))
    AC.PEPSKit._apply_gatempo!(ms,gate)
    rs,ls=AC.PEPSKit._get_allRLs(ms)
    spectra=[svdvals(reshape(AC.convert(Array,r*l),AC.dim(AC.codomain(r*l)),:)) for (r,l) in zip(rs,ls)]
    [s/norm(s) for s in spectra]
end
function verify_spectra()
    rows=Any[];DATA["spectra"]=rows
    for (K,dt) in (([0.18 0.31;0.43 0.22],0.003),([0.21 0.37;0.58 0.51],0.017),
                    (fill(0.45,2,2),0.08))
        psi,w,_=identity_gauge(filtered_plaquette_state(K))
        t=tanh(K[2,2]);M=[exp(dt) t*exp(-dt);t*exp(-dt) t^2*exp(dt)]
        sm=svdvals(M)
        actual=normalized_star_spectra(psi,w,dt)
        expected=[sort(kron([1.,tanh(k)],sm);rev=true) for k in (K[1,2],K[2,1])]
        expected=[s/norm(s) for s in expected]
        difference=[norm(a-b) for (a,b) in zip(actual,expected)]
        push!(rows,Dict("K"=>flat(K),"dt"=>dt,"actual"=>actual,"expected"=>expected,"difference"=>difference,
            "dominant_branch_ratio"=>sm[2]/sm[1],"spectator_ratios"=>[tanh(K[1,2]),tanh(K[2,1])]))
        println("spectra K=$(flat(K)) dt=$dt differences=$difference");flush(stdout);save_data!()
        @test maximum(difference)<1e-11
    end
end
function trajectories()
    rows=Any[];DATA["trajectories"]=rows
    for dt in (0.05,0.02,0.01,0.005),initial in (0.12,0.8)
        K=fill(initial,2,2);psi,w,_=identity_gauge(filtered_plaquette_state(K));alg=make_alg(psi)
        circuit=AC.build_su_circuit(dt;P=AC.UPSPACE);samples=Any[]
        steps=round(Int,3/dt);every=max(1,steps÷10)
        row=Dict("dt"=>dt,"initial"=>initial,"samples"=>samples);push!(rows,row)
        for i in 1:steps
            psi,w,_=AC.PEPSKit.su_iter(psi,circuit,alg,w)
            K=reduced_sweep(K,dt)
            if i%every==0 || i==steps
                f=fit_filter_state(AC.wave(psi))
                error=maximum(abs,f.K-K)
                push!(samples,Dict("step"=>i,"tau"=>i*dt,"fitted_K"=>flat(f.K),"predicted_K"=>flat(K),
                    "map_error"=>error,"fit_residual"=>f.residual,"support"=>f.support,
                    "torus_stabilizers"=>AC.vals(AC.wave(psi)),"predicted_infinite_energy"=>-0.5-sum(tanh.(2K))/8))
                println("trajectory dt=$dt init=$initial tau=$(i*dt) K=$(round.(flat(f.K);digits=7)) maperr=$error");flush(stdout);save_data!()
                @test error<1e-9
                @test f.support>1-1e-10 && f.residual<1e-10
            end
        end
        serialize(joinpath(OUTDIR,"trajectory-dt$(dt)-init$(initial).jls"),(;psi,w,K))
    end
end
function fixed_points()
    rows=Any[];DATA["fixed_points"]=rows
    for dt in (0.1,0.05,0.02,0.01,0.005)
        f=stationary_profile(dt)
        psi,w,_=identity_gauge(filtered_plaquette_state(f.K))
        before=AC.wave(psi)
        psi,w,_=AC.PEPSKit.su_iter(psi,AC.build_su_circuit(dt;P=AC.UPSPACE),make_alg(psi),w)
        after=AC.wave(psi);fit=fit_filter_state(after)
        infid=max(0.,1-abs2(dot(before,after)))
        row=Dict("dt"=>dt,"K"=>flat(f.K),"b_low"=>f.b_low,"b_high"=>f.b_high,"infinite_energy"=>f.energy,
            "sweep_infidelity"=>infid,"map_error"=>maximum(abs,fit.K-f.K),"fit_residual"=>fit.residual)
        push!(rows,row);save_data!()
        println("fixed dt=$dt b=($(f.b_low),$(f.b_high)) e=$(f.energy) infid=$infid");flush(stdout)
        @test infid<1e-12 && row["map_error"]<1e-11
        if dt in (0.05,0.01)
            for chi in (8,16)
                m=AC.measure_peps(psi,2;chi)
                row["chi$(chi)"]=Dict("energy"=>m.energy,"stabilizers"=>m.vals,"converged"=>m.converged,"residual"=>m.residual)
                println("fixed CTMRG dt=$dt chi=$chi e=$(m.energy) residual=$(m.residual)");flush(stdout);save_data!()
                @test m.converged && abs(m.energy-f.energy)<1e-9
            end
        end
    end
end
started=time()
verify_spectra()
trajectories()
fixed_points()
DATA["wall_seconds"]=time()-started;save_data!()
println("followup completed in $(time()-started) seconds");flush(stdout)

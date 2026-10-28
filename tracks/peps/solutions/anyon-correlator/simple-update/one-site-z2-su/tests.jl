using Test, LinearAlgebra, TensorKit
BLAS.set_num_threads(1)
include("core.jl")
using .OneSiteZ2SU

group = isempty(ARGS) ? "all" : only(ARGS)
group in ("all", "--group=tensors", "--group=updates", "--group=runner") || error("unknown group $group")
if group in ("all", "--group=runner")
    @testset "test_absolute_deadline_does_not_restart_budget" begin
        mktempdir() do dir
            marker=joinpath(dir,"started")
            cmd=Cmd(["python3","-c","from pathlib import Path; Path($(repr(marker))).touch()"])
            @test OneSiteZ2SU.supervise(cmd,dir;budget=2.,deadline=time()-1)=="budget"
            @test !isfile(marker)
            @test OneSiteZ2SU.JSON.parsefile(joinpath(dir,"summary.json"))["status"]=="budget"
        end
    end
    @testset "test_supervisor_streams_progress" begin
        mktempdir() do dir
            path=joinpath(dir,"worker.log")
            open(path,"w") do io
                @test OneSiteZ2SU.supervise(`printf progress`,dir;budget=2.,output=io)=="finished"
            end
            @test read(path,String)=="progress"
        end
    end
    @testset "test_resume_preserves_state_and_origin" begin
        mktempdir() do dir
            st=random_state(5); file=joinpath(dir,"latest.json")
            md=(;origin="numerical_random_optimization",seed=5,step=300,dt=0.05,als_sweeps=12,accepted=false)
            save_state(file,st;metadata=md)
            resumed,step=OneSiteZ2SU.resume_state(file;seed=5,dt=0.05,als_sweeps=12)
            @test dense_arrays(resumed)==dense_arrays(st)
            @test step==300
            @test_throws ArgumentError OneSiteZ2SU.resume_state(file;seed=6,dt=0.05,als_sweeps=12)
            @test_throws ArgumentError OneSiteZ2SU.resume_state(file;seed=5,dt=0.1,als_sweeps=12)
            ref=joinpath(dir,"ref.json");save_state(ref,exact_reference_state();metadata=merge(md,(origin="reference",)))
            @test_throws ArgumentError OneSiteZ2SU.resume_state(ref;seed=5,dt=0.05,als_sweeps=12)
        end
    end
    @testset "test_budget_terminates_worker" begin
        mktempdir() do dir
            save_state(joinpath(dir,"latest.json"),random_state(5);metadata=Dict())
            t=time()
            status=OneSiteZ2SU.supervise(`sleep 5`,dir;budget=0.05)
            @test status=="budget"
            @test time()-t < 3
            @test load_state(joinpath(dir,"latest.json"))[1] isa Z2State
            @test OneSiteZ2SU.JSON.parsefile(joinpath(dir,"summary.json"))["passed"]==false
        end
    end
    @testset "test_portable_state_roundtrip" begin
        mktempdir() do dir
            st=random_state(5); file=joinpath(dir,"state.json")
            save_state(file,st;metadata=Dict("origin"=>"random", "seed"=>5))
            restored,metadata=load_state(file)
            @test dense_arrays(restored) == dense_arrays(st)
            @test metadata["seed"] == 5
            @test norm(OneSiteZ2SU.star_tensor(restored)-OneSiteZ2SU.star_tensor(st)) < 1e-12
            @test_throws ArgumentError save_state(file,st;metadata=Dict())
        end
    end
    @testset "test_invalid_state_rejected" begin
        mktempdir() do dir
            file=joinpath(dir,"bad.json"); write(file,"{\"format\":\"invalid\"}")
            @test_throws ArgumentError load_state(file)
            good=joinpath(dir,"good.json");save_state(good,random_state(5);metadata=Dict())
            data=OneSiteZ2SU.JSON.parsefile(good);data["V"][9]=1e-9
            OneSiteZ2SU.write_json(file,data)
            @test_throws ArgumentError load_state(file)
        end
    end
    @testset "test_output_not_overwritten" begin
        mktempdir() do dir
            @test_throws ArgumentError OneSiteZ2SU.prepare_output(dir)
        end
    end
    @testset "test_failed_audit_stays_failed" begin
        good=(;energy=-1.0,vals=ones(8),converged=true,residual=0.0,chi=8)
        other=merge(good,(chi=16,))
        st=exact_reference_state()
        @test OneSiteZ2SU.audit([good,other],st,"finished").passed
        @test !OneSiteZ2SU.audit([good,other],st,"budget").passed
        @test !OneSiteZ2SU.audit([merge(good,(converged=false,)),other],st,"finished").passed
        @test !OneSiteZ2SU.audit([merge(good,(vals=zeros(8),)),other],st,"finished").passed
        @test !OneSiteZ2SU.audit([merge(good,(energy=NaN,)),other],st,"finished").passed
    end
end
if group in ("all", "--group=tensors")
    @testset "test_native_spaces_and_initialization" begin
        st = random_state(5)
        @test fieldnames(typeof(st)) == (:V, :P, :lambda)
        @test st.V isa AbstractTensorMap
        @test st.P isa AbstractTensorMap
        @test st.lambda isa DiagonalTensorMap
        @test dim(virtual_space(), Z2Irrep(0)) == 1
        @test dim(virtual_space(), Z2Irrep(1)) == 1
        @test dim(physical_space(), Z2Irrep(0)) == 2
        @test dim(physical_space(), Z2Irrep(1)) == 0
        @test maximum(values(symmetry_residuals(st))) <= 1e-12
        @test norm(dense_arrays(st).P-dense_arrays(exact_reference_state()).P) > 1e-3
        @test dense_arrays(random_state(5)) == dense_arrays(st)
    end
    @testset "test_array_conventions" begin
        st = exact_reference_state(); a = dense_arrays(st)
        @test size(a.V) == (2,2,2,2)
        @test size(a.P) == (2,2,2)
        for n in 0:1,e in 0:1,s in 0:1,w in 0:1
            @test a.V[n+1,e+1,s+1,w+1] ≈ (iseven(n+e+s+w) ? 1/sqrt(8) : 0) atol=1e-12
        end
        for p in 0:1,a0 in 0:1,b in 0:1
            @test a.P[p+1,a0+1,b+1] ≈ (a0==b ? (-1.0)^(p*a0)/2 : 0) atol=1e-12
        end
        @test a.lambda ≈ [1,1]/sqrt(2)
        # A cyclic invariant need not be invariant under arbitrary swaps.
        v=zeros(2,2,2,2); v[1,1,2,2]=1
        st2=state_from_arrays(v,a.P,a.lambda)
        @test maximum(values(symmetry_residuals(st2))) <= 1e-12
        @test norm(dense_arrays(st2).V-permutedims(dense_arrays(st2).V,(1,3,2,4))) > 0.1
    end
end
if group in ("all", "--group=updates")
    include("../split/core.jl")
    function dense_workspace(st)
        a=dense_arrays(st); w=SplitSU.initial_state(:exact,0)
        for (n,t) in w.tensors; w.tensors[n]=copy(n[1]==:V ? a.V : a.P); end
        for b in keys(w.weights); w.weights[b]=copy(a.lambda); end
        w
    end
    @testset "test_charge_preserving_star" begin
        st=random_state(5)
        x=OneSiteZ2SU.star_tensor(st)
        xd=first(SplitSU.star_cluster(dense_workspace(st),1,1))
        @test reshape(convert(Array,x),4,4,4,4) ≈ xd atol=1e-12
        d=star_step!(st,0.05)
        @test maximum(values(symmetry_residuals(st))) <= 1e-12
        @test isfinite(d.discarded_weight)
        @test d.gauge_residual <= 1e-8
    end
    @testset "test_constrained_ring_fit" begin
        st=random_state(5)
        x=OneSiteZ2SU.plaquette_tensor(st)
        xd=SplitSU.ring_tensor(first(SplitSU.ring_cluster(dense_workspace(st),1,1)))
        @test vec(convert(Array,x)) ≈ vec(xd) atol=1e-12
        d=plaquette_step!(st,0.05;als_sweeps=2)
        @test isfinite(d.fit_residual)
        @test maximum(values(symmetry_residuals(st))) <= 1e-12
        @test d.gauge_residual <= 1e-8
    end
    @testset "test_degenerate_gauge_alignment" begin
        st=random_state(5); q=virtual_space()
        z=TensorMap([1.0 0;0 -1],q←q)
        @test OneSiteZ2SU.gauge_pair_residual(st,z) < 1e-12
        @test_throws ArgumentError OneSiteZ2SU.validate_charge_gauge([0.0 1;1 0])
        ref=exact_reference_state();v=OneSiteZ2SU.mode(ref.V,z,1)
        ps=[OneSiteZ2SU.outgoing(ref.P) for _ in 1:4]
        ps[1]=OneSiteZ2SU.mode(ps[1],z,2)
        vv,pp=OneSiteZ2SU.align_star(ref,v,ps)
        @test norm(OneSiteZ2SU.canonical_v(vv)-ref.V)<1e-12
        @test all(p->norm(OneSiteZ2SU.canonical_p(p)-ref.P)<1e-12,pp)
    end
    @testset "test_zero_support_pseudoinverse" begin
        a=dense_arrays(exact_reference_state()); st=state_from_arrays(a.V,a.P,[1.0,0])
        @test all(isfinite,convert(Array,OneSiteZ2SU.support_inverse(st.lambda)))
    end
    @testset "test_exact_reference_update_is_fixedpoint" begin
        st=exact_reference_state()
        @test maximum(values(local_residuals(st))) < 1e-12
        sweep!(st,0.05;als_sweeps=12)
        @test maximum(values(local_residuals(st))) < 1e-12
    end
    @testset "test_product_plus_als_stationary" begin
        a=dense_arrays(random_state(5))
        p=[a0==b ? 0.5 : 0.0 for p in 0:1,a0 in 0:1,b in 0:1]
        st=state_from_arrays(a.V,p,a.lambda)
        d=plaquette_step!(st,0.05;als_sweeps=12)
        expected=1-cosh(0.05)^2/(cosh(0.05)^2+sinh(0.05)^2)
        @test d.fit_residual ≈ expected atol=1e-12
        @test local_residuals(st).plaquette ≈ sqrt(2) atol=1e-12
        # This diagnoses a stationary ALS fit, never an accepted ground state.
        good=(;energy=-1.,vals=ones(8),converged=true,residual=0.,chi=8)
        @test !OneSiteZ2SU.audit([good,merge(good,(chi=16,))],st,"finished").passed
    end
    @testset "test_atomic_sweep" begin
        st=random_state(5); a=dense_arrays(st)
        @test_throws ArgumentError sweep!(st,0.05;als_sweeps=0)
        @test dense_arrays(st)==a
        d=sweep!(st,0.05;als_sweeps=1)
        @test maximum(values(symmetry_residuals(st))) <= 1e-12
        @test all(isfinite,values(local_residuals(st)))
        for _ in 1:3; sweep!(st,0.05;als_sweeps=12); end
        @test eltype(st.V)==Float64
        @test all(isfinite,values(local_residuals(st)))
    end
end

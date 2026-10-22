using Test, LinearAlgebra, Random
BLAS.set_num_threads(1)
const FILE=joinpath(@__DIR__,"strict.jl")
isfile(FILE) && include(FILE)
@testset "strict single-state implementation exists" begin
    @test isdefined(@__MODULE__,:StrictOneSiteSU)
end

const RUNFILE=joinpath(@__DIR__,"strict_run.jl")
isfile(RUNFILE) && include(RUNFILE)
@testset "strict numerical runner is implemented" begin
    @test isdefined(@__MODULE__,:StrictZeroRun)
end
if isdefined(@__MODULE__,:StrictZeroRun)
    const R=StrictZeroRun
    @testset "checkpoint stores exactly one V/P/lambda and failed audits stay failed" begin
        st=R.S.random_state(5)
        mktempdir(@__DIR__) do dir
            path=joinpath(dir,"state.json")
            R.save_state(path,st;seed=5,step=10)
            other,meta=R.load_state(path)
            @test other.V==st.V && other.P==st.P && other.lambda==st.lambda
            @test fieldnames(typeof(other))==(:V,:P,:lambda)
            @test meta["step"]==10
            @test_throws ArgumentError R.prepare_output(dir)
        end
        m=(;energy=-1.0,vals=ones(8),maxerr=0.0,converged=true,residual=1e-12,chi=8)
        m16=merge(m,(chi=16,))
        @test R.audit([m,m16],st,"finished").passed
        @test !R.audit([m,m],st,"finished").passed
        @test !R.audit([m,m16],st,"budget").passed
        @test !R.audit([m,merge(m16,(converged=false,))],st,"finished").passed
        @test !R.audit([m,merge(m16,(energy=NaN,))],st,"finished").passed
        @test !R.audit([m,merge(m16,(vals=[ones(7);0.9],))],st,"finished").passed
    end
end
if isdefined(@__MODULE__,:StrictOneSiteSU)
    const T=StrictOneSiteSU
    @testset "only three arrays persist and spatial constraints are exact" begin
        st=T.random_state(1)
        @test fieldnames(typeof(st))==(:V,:P,:lambda)
        @test size(st.V)==(2,2,2,2) && size(st.P)==(2,2,2) && length(st.lambda)==2
        @test st.V==permutedims(st.V,(2,3,4,1))
        @test st.P==permutedims(st.P,(1,3,2))
        @test st.V[1,1,1,2]!=0 # No even-parity constraint or analytic initialization.
        @test st.P[1,1,2]!=0
        @test all(>(0),st.lambda)
    end
    @testset "virtual gauge and lambda compensation preserve a weighted star" begin
        st=T.random_state(3);work=T.workspace(st)
        before,_=T.K.star_cluster(work,1,1)
        q=[cos(0.37) -sin(0.37);sin(0.37) cos(0.37)]
        T.gauge_bond!(work,(1,1,1),q,[0.8,0.6])
        after,_=T.K.star_cluster(work,1,1)
        @test before ≈ after atol=1e-12
    end
    @testset "original updates commit one symmetric state at each substep" begin
        st=T.random_state(1)
        for i in 1:3
            a=T.star_step!(st,0.05)
            @test a.gauge_error<1e-10
            @test T.defect(st)<1e-14
            b=T.plaquette_step!(st,0.05;als_sweeps=12)
            @test b.gauge_error<1e-10
            @test T.defect(st)<1e-14
            @test all(isfinite,st.V) && all(isfinite,st.P)
        end
    end
    @testset "aligning a pure virtual gauge must not erase vertex information" begin
        st=T.random_state(7)
        p=zeros(2,2,2)
        p[:,1,1]=[1.0,1.0];p[:,2,2]=[1.0,-1.0]
        T.commit!(st,st.V,p,st.lambda)
        old=deepcopy(st);work=T.workspace(st)
        nodes=T.K.ring_nodes(1,1)
        _,meta=T.K.ring_cluster(work,1,1)
        for (i,b) in enumerate(meta[1].links)
            theta=0.23i
            q=[cos(theta) -sin(theta);sin(theta) cos(theta)]
            T.gauge_bond!(work,b,q,st.lambda)
        end
        info=T.restore_shared!(st,work,:plaquette)
        @test info.gauge_error<1e-10
        @test T.relative(st.V,old.V)<1e-8
        @test T.relative(st.P,old.P)<1e-8
    end
    @testset "reflection ambiguity of P must be resolved using V" begin
        for theta in (0.0,0.21,0.63)
            st=T.random_state(7)
            p=zeros(2,2,2)
            p[:,1,1]=[1.0,1.0];p[:,2,2]=[1.0,-1.0]
            q=[cos(theta) -sin(theta);sin(theta) cos(theta)]
            p=T.K.mode_product(T.K.mode_product(p,q,2),q,3)
            T.commit!(st,st.V,p,st.lambda)
            old=deepcopy(st);work=T.workspace(st)
            reflection=q*[1.0 0;0 -1]*q'
            T.gauge_bond!(work,(1,2,4),reflection,st.lambda)
            info=T.restore_shared!(st,work,:plaquette)
            @test info.gauge_error<1e-10
            @test T.relative(st.V,old.V)<1e-8
            @test T.relative(st.P,old.P)<1e-8
        end
    end
    @testset "a failed plaquette must not leave an uncounted star update" begin
        st=T.random_state(5);old=deepcopy(st)
        @test_throws ArgumentError T.sweep!(st,0.05;als_sweeps=0)
        @test st.V==old.V && st.P==old.P && st.lambda==old.lambda
    end
end

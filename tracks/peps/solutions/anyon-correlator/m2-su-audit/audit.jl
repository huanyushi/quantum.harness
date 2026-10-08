using LinearAlgebra, Random, Printf, TOML, Serialization, Test
include("../scripts/tc_peps.jl")
include("../simple-update/measure.jl")
BLAS.set_num_threads(1)
const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)
const ROWS = Dict{String,Any}()
function save!()
    open(joinpath(OUT,"audit.toml"),"w") do io
        TOML.print(io,ROWS)
    end
end

"Eight-spin, 2x2 torus only; never an infinite-plane wavefunction."
function wave(psi)
    ts = [reshape(convert(Array,psi[r,c]),4,2,2,2,2) for r in 1:2 for c in 1:2]
    v = zeros(ComplexF64,256)
    for cfg in 0:255
        a,b,c,d,e,f,g,h = [1+((cfg>>k)&1) for k in 0:7]
        for p in 0:255
            p1,p2,p3,p4 = [1+((p>>(2k))&3) for k in 0:3]
            v[p+1] += ts[1][p1,a,b,c,d]*ts[2][p2,e,d,f,b]*ts[3][p3,c,g,a,h]*ts[4][p4,f,h,e,g]
        end
    end
    return normalize(v)
end
ebit(r,c)=2*(2*(mod1(r,2)-1)+mod1(c,2)-1)
nbit(r,c)=ebit(r,c)+1
function action(v,kind,r,c)
    bits = kind==:A ? (ebit(r,c),nbit(r,c),ebit(r,c-1),nbit(r+1,c)) :
                       (ebit(r,c),nbit(r,c),ebit(r-1,c),nbit(r,c+1))
    mask = sum(1<<k for k in bits)
    return kind==:A ? [v[(p⊻mask)+1] for p in 0:255] :
        [v[p+1]*(iseven(count_ones(p&mask)) ? 1 : -1) for p in 0:255]
end
vals(v)=[real(dot(v,action(v,k,r,c))) for r in 1:2 for c in 1:2 for k in (:A,:B)]
function single(kind,r,c,dt)
    sites = kind==:A ? [CartesianIndex(r,c-1),CartesianIndex(r,c),CartesianIndex(r+1,c)] :
                       [CartesianIndex(r-1,c),CartesianIndex(r,c),CartesianIndex(r,c+1)]
    op=phys_tmap(kind==:A ? STAR_A_mat() : PLAQ_B_mat(),3,UPSPACE)
    gate=PEPSKit.gate_to_mpo(pauli_gate3(op,dt,1.0))
    return PEPSKit.LocalCircuit(fill(UPSPACE,2,2),[sites=>gate])
end
function step!(psi,w,alg,kind,r,c;dt=0.05)
    return PEPSKit.su_iter(psi,single(kind,r,c,dt),alg,w)
end
function record_step(psi,w,alg,kind,r,c;dt=0.05)
    before=wave(psi)
    target=normalize(cosh(dt)*before+sinh(dt)*action(before,kind,r,c))
    p2,w2,err=step!(psi,w,alg,kind,r,c;dt)
    after=wave(p2)
    row=Dict("kind"=>String(kind),"r"=>r,"c"=>c,"dt"=>dt,"before"=>vals(before),
        "exact"=>vals(target),"after"=>vals(after),"library_error"=>Float64(err),
        "target_infidelity"=>max(0.0,1-abs2(dot(target,after))),
        "input_infidelity"=>max(0.0,1-abs2(dot(before,after))))
    if kind==:A
        inverse_b=normalize(cosh(dt)*before-sinh(dt)*action(before,:B,r,c))
        row["inverse_plaquette_infidelity"]=max(0.0,1-abs2(dot(inverse_b,after)))
        row["inverse_plaquette_target"]=vals(inverse_b)
    end
    @printf("%s(%d,%d) dt=%.3f e: %.10f -> %.10f target %.10f infid %.3e\n",kind,r,c,dt,-sum(row["before"])/8,-sum(row["after"])/8,-sum(row["exact"])/8,row["target_infidelity"]);flush(stdout)
    return p2,w2,row
end
function main()
    if "--resume" in ARGS
        merge!(ROWS,TOML.parsefile(joinpath(OUT,"audit.toml")))
        states=deserialize(joinpath(OUT,"states.jls"))
        measurements(states)
        return
    end
    @testset "independent finite oracle" begin
        v=wave(exact_peps(UPSPACE,uspace(2)))
        @test vals(v) ≈ ones(8) atol=1e-13
        rng=MersenneTwister(71); v=normalize(randn(rng,ComplexF64,256))
        for k in (:A,:B)
            b=real(dot(v,action(v,k,1,1)))
            target=normalize(cosh(0.05)*v+sinh(0.05)*action(v,k,1,1))
            @test real(dot(target,action(target,k,1,1))) ≈ (b+tanh(0.1))/(1+b*tanh(0.1)) atol=1e-13
        end
    end
    Random.seed!(1)
    a=random_merged_tensor_VP(randn(2,2,2,2),randn(2,2,2),randn(2,2,2),UPSPACE,uspace(2))
    psi=InfinitePEPS(fill(a,2,2));w=init_suweight(psi,uspace(2))
    alg=SimpleUpdate(;trunc=PEPSKit._get_fixedspacetrunc(psi),imaginary_time=true)
    circuit=build_su_circuit(0.05;P=UPSPACE)
    for i in 1:175
        psi,w,_=PEPSKit.su_iter(psi,circuit,alg,w)
        i%25==0 && (println("replay $i/175");flush(stdout))
    end
    serialize(joinpath(OUT,"stalled.jls"),(;psi,w))
    ROWS["setup"]=Dict("D"=>2,"seed"=>1,"dt"=>0.05,"sweeps"=>175,"julia"=>string(VERSION))
    ROWS["stalled_torus"]=vals(wave(psi));save!()
    states=Dict("stalled"=>deepcopy(psi))
    for (name,kinds,dt) in (("full",(:A,:B),0.05),("plaq_only",(:B,),0.05),("identity",(:A,:B),0.0))
        p,q=deepcopy(psi),deepcopy(w); rows=Any[]
        for r in 1:2,c in 1:2,k in kinds
            p,q,row=record_step(p,q,alg,k,r,c;dt)
            push!(rows,row);ROWS[name]=rows;save!()
        end
        states[name]=deepcopy(p)
    end
    for kind in (:A,:B)
        p,q,row=record_step(deepcopy(psi),deepcopy(w),alg,kind,1,1)
        ROWS["isolated_$(kind)"]=row;states["isolated_$(kind)"]=p;save!()
    end
    serialize(joinpath(OUT,"states.jls"),states)
    measurements(states)
end
function measurements(states)
    for name in ("stalled","full","plaq_only","identity","isolated_A","isolated_B"),chi in (8,16)
        haskey(ROWS,"$(name)_chi$(chi)") && continue
        m=measure_peps(states[name],2;chi)
        ROWS["$(name)_chi$(chi)"]=Dict("energy"=>m.energy,"stabilizers"=>m.vals,"residual"=>m.residual,"converged"=>m.converged)
        println("CTMRG $name chi=$chi e=$(m.energy) residual=$(m.residual) converged=$(m.converged)");flush(stdout);save!()
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    main()
end

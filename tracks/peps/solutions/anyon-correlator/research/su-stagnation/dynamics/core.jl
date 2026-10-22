using LinearAlgebra, Random, Serialization, TOML, JSON, Printf
using TensorKit: ⊗

module AuditPrimitives
include("../single-step/audit.jl")
end

function filter_basis()
    plus=fill(ComplexF64(1/16),256)
    orbit=zeros(ComplexF64,256,8)
    sites=((1,1),(1,2),(2,1))
    for bits in 0:7
        x=copy(plus)
        for (i,(r,c)) in enumerate(sites)
            isodd(bits>>(i-1)) && (x=AC.action(x,:B,r,c))
        end
        orbit[:,bits+1]=x
    end
    h=[(-1.0)^count_ones(q&s)/sqrt(8) for q in 0:7,s in 0:7]
    signs=[(-1.0)^(isodd(s>>(i-1))) for s in 0:7,i in 1:3]
    signs=hcat(signs,prod(signs;dims=2))
    return orbit*h,signs
end

"Fit amplitudes in a fixed B-eigenbasis; do not assume SU preserves this family."
function fit_filter_state(v)
    Q,signs=filter_basis()
    c=Q'*normalize(v);support=sum(abs2,c)
    a=abs.(c)
    if minimum(a)<1e-13
        return (;valid=false,K=zeros(2,2),support,residual=1.0)
    end
    design=hcat(ones(8),signs)
    coeff=design\log.(a)
    phase=c[argmax(a)]/maximum(a)
    predicted=phase*exp.(design*coeff)
    residual=norm(c-predicted)/norm(c)
    K=[coeff[2] coeff[3];coeff[4] coeff[5]]
    return (;valid=true,K,support,residual)
end

function make_alg(psi)
    AC.SimpleUpdate(;trunc=AC.PEPSKit._get_fixedspacetrunc(psi),imaginary_time=true)
end

function identity_gauge(psi;maxsteps=100)
    psi=deepcopy(psi);w=AC.init_suweight(psi,AC.uspace(2));alg=make_alg(psi)
    circuit=AC.build_su_circuit(0.0;P=AC.UPSPACE)
    before=AC.wave(psi);last=deepcopy(w);delta=Inf;steps=0
    for i in 1:maxsteps
        psi,w,_=AC.PEPSKit.su_iter(psi,circuit,alg,w)
        steps=i
        delta=AC.PEPSKit.compare_weights(last,w);last=deepcopy(w)
        delta<1e-12 && break
    end
    fidelity=abs2(dot(before,AC.wave(psi)))
    fidelity>1-1e-11 || error("identity gauging changed the finite physical state")
    return psi,w,(;steps,delta,infidelity=max(0.,1-fidelity))
end

"Closed star map on the filtered family when both cuts retain the dominant branch."
function star_filter_map(K,dt)
    K>=0 && dt>=0 || throw(ArgumentError("this branch assumes K,dt >= 0"))
    K==0 && return 0.0
    t=tanh(K)
    a=exp(2dt)*(1-t^2)/t
    r=2/(sqrt(a^2+4)+a)
    return atanh(r^2/t)
end

"Stationary K profile for the original row-major A/B sweep, not a ground state."
function stationary_profile(dt)
    dt>=0 || throw(ArgumentError("dt must be nonnegative"))
    low=dt==0 ? log(3)/4 : atanh(sinh(2dt)/(exp(4dt)/cosh(dt)^2-cosh(2dt)))/2
    high=low+dt
    K=[low low;high high]
    return (;K,b_low=tanh(2low),b_high=tanh(2high),energy=-0.5-(tanh(2low)+tanh(2high))/4)
end

"Use an existing legacy checkpoint, or replay it into this research directory."
function legacy_state(;force_replay=false)
    old=joinpath(@__DIR__,"..","single-step","results","stalled.jls")
    localpath=joinpath(@__DIR__,"results","legacy-reference","stalled.jls")
    for path in (old,localpath)
        !force_replay && isfile(path) && return deserialize(path)
    end
    AC.Random.seed!(1)
    a=AC.random_merged_tensor_VP(randn(2,2,2,2),randn(2,2,2),randn(2,2,2),AC.UPSPACE,AC.uspace(2))
    psi=AC.InfinitePEPS(fill(a,2,2));w=AC.init_suweight(psi,AC.uspace(2));alg=make_alg(psi)
    circuit=AC.build_su_circuit(0.05;P=AC.UPSPACE)
    for i in 1:175
        psi,w,_=AC.PEPSKit.su_iter(psi,circuit,alg,w)
        i%25==0 && (println("legacy replay $i/175");flush(stdout))
    end
    mkpath(dirname(localpath));serialize(localpath,(;psi,w))
    return (;psi,w)
end
const AC=AuditPrimitives

"Exact ∏ exp(K_p B_p)|+⟩ state, represented with one bit per dual plaquette."
function filtered_plaquette_state(K::AbstractMatrix)
    size(K)==(2,2) || throw(DimensionMismatch("K must have four plaquette positions"))
    p,v=AC.UPSPACE,AC.uspace(2)
    ts=map(CartesianIndices(K)) do site
        k=K[site]
        t=zeros(ComplexF64,2,2,2,2,2,2)
        for pe in 0:1,pn in 0:1,n in 0:1,e in 0:1,s in 0:1,w in 0:1
            n==e || continue
            a=n==0 ? cosh(k) : sinh(k)
            t[pe+1,pn+1,n+1,e+1,s+1,w+1]=a*(-1.0)^(pe*(n⊻s)+pn*(n⊻w))/2
        end
        AC.TensorMap(reshape(t,4,2,2,2,2),p,v⊗v⊗v'⊗v')
    end
    return AC.InfinitePEPS(ts)
end

"Closed-form gauge corresponding to the identity-SU fixed point, for positive K."
function canonical_filtered_state(K::AbstractMatrix)
    size(K)==(2,2) || throw(DimensionMismatch("K must be 2x2"))
    all(>(0),K) || throw(ArgumentError("the invertible gauge requires positive K"))
    a(r,c,q)=q==0 ? cosh(K[mod1(r,2),mod1(c,2)]) : sinh(K[mod1(r,2),mod1(c,2)])
    p,v=AC.UPSPACE,AC.uspace(2)
    ts=map(CartesianIndices(K)) do site
        r,c=Tuple(site)
        t=zeros(ComplexF64,2,2,2,2,2,2)
        for pe in 0:1,pn in 0:1,n in 0:1,e in 0:1,s in 0:1,w in 0:1
            n==e || continue
            t[pe+1,pn+1,n+1,e+1,s+1,w+1]=sqrt(a(r+1,c,s)*a(r,c-1,w))*(-1.0)^(pe*(n⊻s)+pn*(n⊻w))/2
        end
        AC.TensorMap(reshape(t,4,2,2,2,2),p,v⊗v⊗v'⊗v')
    end
    weights=[AC.DiagonalTensorMap([1.,tanh(K[r,c])],v) for r in 1:2,c in 1:2]
    return AC.InfinitePEPS(ts),AC.PEPSKit.SUWeight(deepcopy(weights),deepcopy(weights))
end

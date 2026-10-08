using Test, LinearAlgebra, Random, TensorKit, Zygote

isfile(joinpath(@__DIR__, "network.jl")) && include("network.jl")

# Catches double-counted/missing weights and the north/east physical-index swap.
function scalar_block(st, r, c; balanced)
    D = size(st.V[r,c], 1)
    w = st.weights
    at(r,c,d) = w[mod1(r,2),mod1(c,2),d]
    out = zeros(2,2,D,D,D,D)
    for p in 1:2, q in 1:2, n in 1:D, e in 1:D, s in 1:D, t in 1:D
        for a in 1:D, b in 1:D
            weight = at(r,c,2)[a]*at(r,c,1)[b]
            weight *= balanced ? sqrt(at(r,c+1,4)[e]*at(r-1,c,3)[n]*at(r,c,3)[s]*at(r,c,4)[t]) : at(r,c+1,4)[e]*at(r-1,c,3)[n]
            out[p,q,n,e,s,t] += st.V[r,c][b,a,s,t]*st.E[r,c][p,a,e]*st.N[r,c][q,b,n]*weight
        end
    end
    return reshape(out,4,D,D,D,D)
end

@testset "independent Pauli action and field normalization" begin
    _,ops = operators(0.1,0.1)
    mask = sum(1 << (i-1) for i in (1,3,4,6))
    expected_x = [i==xor(j,mask) ? 1.0 : 0.0 for i in 0:63,j in 0:63]
    expected_z = Diagonal([(-1)^sum((i >> (s-1)) & 1 for s in (1,3,4,6)) for i in 0:63])
    for r in 1:2,c in 1:2
        a = reshape(convert(Array,only(values(ops["A$r$c"].terms))),64,64)
        b = reshape(convert(Array,only(values(ops["B$r$c"].terms))),64,64)
        @test a ≈ expected_x
        @test b ≈ expected_z
    end
    zedge = reshape(convert(Array,only(values(ops["ZE11"].terms))),4,4)
    znorth = reshape(convert(Array,only(values(ops["ZN11"].terms))),4,4)
    @test diag(zedge)==[1,-1,1,-1]
    @test diag(znorth)==[1,1,-1,-1]
end

isfile(joinpath(@__DIR__,"optimize.jl")) && include("optimize.jl")
@testset "safeguarded L-BFGS on an independent quadratic" begin
    @test isdefined(@__MODULE__, :lbfgs_direction)
    if isdefined(@__MODULE__, :lbfgs_direction)
        Q = Diagonal([1.0,3.0,8.0])
        x = [1.0,-2.0,0.5]
        pairs = Tuple{Vector{Float64},Vector{Float64}}[]
        for _ in 1:30
            g = Q*x
            d = lbfgs_direction(g,pairs)
            norm(g)<1e-10 && break
            @test dot(g,d)<0
            alpha = -dot(g,d)/dot(d,Q*d)
            s = alpha*d
            y = Q*s
            push!(pairs,(s,y));length(pairs)>8 && popfirst!(pairs)
            x += s
        end
        @test norm(Q*x)<1e-8
        @test lbfgs_direction([1.0,2.0],[(zeros(2),zeros(2))]) ≈ [-1.0,-2.0]
    end
end

isfile(joinpath(@__DIR__,"objective.jl")) && include("objective.jl")

@testset "a passed derivative probe cannot authorize a different setup" begin
    @test isdefined(@__MODULE__, :probe_matches)
    if isdefined(@__MODULE__, :probe_matches)
        p = Dict("passed"=>true,"D"=>2,"chi"=>8,"hx"=>0.0,"hz"=>0.1,"parameter_count"=>40)
        @test probe_matches(p,2,8,0.0,0.1,40)
        @test !probe_matches(p,2,8,0.1,0.0,40)
        @test !probe_matches(p,2,16,0.0,0.1,40)
        @test !probe_matches(p,2,8,0.0,0.1,160)
        @test !probe_matches(Dict("passed"=>true),2,8,0.0,0.1,40)
    end
end

@testset "exact-state contraction verdict checks all sites" begin
    @test isdefined(@__MODULE__, :calibration_ok)
    if isdefined(@__MODULE__, :calibration_ok)
        obs = (;energy=-1.0,imagmax=0.0,observables=Dict("$k$r$c"=>1.0 for k in ("A","B"),r in 1:2,c in 1:2))
        rows = [(;status="converged",measurement=obs) for _ in 1:4]
        @test calibration_ok(rows)
        bad = merge(obs,(;observables=merge(obs.observables,Dict("A22"=>0.99))))
        @test !calibration_ok([merge(row,(;measurement=bad)) for row in rows])
        @test !calibration_ok(vcat(rows[1:3],[(;status="failed")]))
    end
end

@testset "final acceptance rejects false physical success" begin
    @test isdefined(@__MODULE__, :assess_audit)
    if isdefined(@__MODULE__, :assess_audit)
        vals = Dict("A$r$c"=>0.9948533065 for r in 1:2,c in 1:2)
        merge!(vals,Dict("B$r$c"=>1.0 for r in 1:2,c in 1:2))
        obs = (;energy=-1.002524034729,mx=0.0,mz=0.050973814624,imagmax=0.0,observables=vals)
        rows = [(;chi,seed,status="converged",residual=1e-10,measurement=obs) for (chi,seed) in ((8,2026),(8,1),(8,2),(16,2026))]
        @test assess_audit(rows,0.0,0.1;gradnorm=1e-6,status="stationary").accepted
        @test !assess_audit(rows,0.0,0.1;gradnorm=1e-6,status="budget").accepted
        @test !assess_audit(rows,0.0,0.1;gradnorm=1e-3,status="stationary").accepted
        @test !assess_audit(rows[2:end],0.0,0.1;gradnorm=1e-6,status="stationary").accepted
        bad = merge(obs,(;observables=merge(vals,Dict("B11"=>1.01))))
        badrows = [merge(r,(;measurement=bad)) for r in rows]
        @test !assess_audit(badrows,0.0,0.1;gradnorm=1e-6,status="stationary").accepted
        badrows = copy(rows)
        badrows[end] = merge(rows[end],(;measurement=merge(obs,(;energy=-1.0025))))
        @test !assess_audit(badrows,0.0,0.1;gradnorm=1e-6,status="stationary").accepted
        badrows = [merge(r,(;measurement=merge(obs,(;mz=0.04)))) for r in rows]
        @test !assess_audit(badrows,0.0,0.1;gradnorm=1e-6,status="stationary").accepted
        zobs = (;energy=-1.0,mx=0.0,mz=0.0,imagmax=0.0,observables=Dict("$k$r$c"=>1.0 for k in ("A","B"),r in 1:2,c in 1:2))
        zrows = [merge(r,(;measurement=zobs)) for r in rows]
        @test assess_audit(zrows,0.0,0.0;gradnorm=1e-6,status="stationary").accepted
        zbad = merge(zobs,(;energy=-0.99999875,observables=merge(zobs.observables,Dict("A11"=>0.99999))))
        @test !assess_audit([merge(r,(;measurement=zbad)) for r in rows],0.0,0.0;gradnorm=1e-6,status="stationary").accepted
    end
end

@testset "uniform sharing differentiates the same split network" begin
    x = initial_parameters(2,19;uniform=true)
    aa = blocked_arrays(x,2)
    @test all(a->a≈aa[1,1],aa)
    coeffs = [randn(MersenneTwister(i),size(a)) for (i,a) in enumerate(aa)]
    f(y) = sum(sum(a.*b) for (a,b) in zip(blocked_arrays(y,2),coeffs))
    g = only(Zygote.gradient(f,x))
    d = normalize(randn(MersenneTwister(21),length(x)))
    @test dot(g,d) ≈ (f(x+1e-5*d)-f(x-1e-5*d))/2e-5 rtol=1e-7 atol=1e-9
end

function torus_amplitudes(aa)
    a,b,c,d = aa[1,1],aa[1,2],aa[2,1],aa[2,2]
    @tensor state[p,q,r,s] := a[p,i,j,k,l]*b[q,m,l,n,j]*c[r,k,o,i,t]*d[s,n,t,m,o]
    return vec(state)/norm(state)
end

@testset "split-network blocking preserves the weighted wavefunction" begin
    @test isdefined(@__MODULE__, :initial_parameters)
    if isdefined(@__MODULE__, :initial_parameters)
        x = initial_parameters(2, 19)
        x[end-31:end] .= range(-0.3,0.7;length=32)
        st = split_parameters(x,2)
        aa = blocked_arrays(x,2)
        for r in 1:2,c in 1:2
            @test aa[r,c] ≈ scalar_block(st,r,c;balanced=true) atol=1e-13
        end
        unbalanced = [scalar_block(st,r,c;balanced=false) for r in 1:2,c in 1:2]
        @test torus_amplitudes(aa) ≈ torus_amplitudes(unbalanced) atol=1e-12
        coefficients = [randn(MersenneTwister(i),size(a)) for (i,a) in enumerate(aa)]
        f(y) = sum(sum(a.*b) for (a,b) in zip(blocked_arrays(y,2),coefficients))
        g = only(Zygote.gradient(f,x))
        direction = normalize(randn(MersenneTwister(20),length(x)))
        @test dot(g,direction) ≈ (f(x+1e-5*direction)-f(x-1e-5*direction))/2e-5 rtol=1e-7 atol=1e-9
        psi = make_peps(x,2)
        maps = [TensorMap(ComplexF64.(reshape(b,4,16)),codomain(t),domain(t)) for (t,b) in zip(psi.A,coefficients)]
        fp(y) = real(sum(dot(t,b) for (t,b) in zip(make_peps(y,2).A,maps)))
        gp = only(Zygote.gradient(fp,x))
        @test dot(gp,direction) ≈ (fp(x+1e-5*direction)-fp(x-1e-5*direction))/2e-5 rtol=1e-7 atol=1e-9
        exact = torus_amplitudes(blocked_arrays(initial_parameters(2,1;exact=true),2))
        # Independently act on physical bits of the 8-spin finite wavefunction.
        for sites in ((1,3,2,6),(3,1,4,8),(5,7,6,2),(7,5,8,4))
            mask = sum(1 << (i-1) for i in sites)
            @test exact ≈ exact[[xor(i,mask)+1 for i in 0:255]] atol=1e-12
        end
        for sites in ((1,5,2,4),(3,7,4,2),(5,1,6,8),(7,3,8,6))
            signs = [(-1)^sum((i >> (s-1)) & 1 for s in sites) for i in 0:255]
            @test exact ≈ exact.*signs atol=1e-12
        end
    end
end

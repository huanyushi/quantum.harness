# Independent real split-network parameterization. No SU or old AD imports.
using LinearAlgebra, Random, Statistics, TensorKit, PEPSKit, JSON, Zygote

parameter_count(D;uniform=false) = (uniform ? 1 : 4)*(D^4+4D^2+4D)

function initial_parameters(D, seed; exact=false,uniform=false)
    D >= 2 || throw(ArgumentError("D must be at least two"))
    rng = MersenneTwister(seed)
    data = map(1:(uniform ? 1 : 4)) do _
        V = exact ? [n<=2 && e<=2 && s<=2 && w<=2 && iseven(n+e+s+w) ? 1.0 : 0.0 for n in 1:D,e in 1:D,s in 1:D,w in 1:D] : randn(rng,D,D,D,D)
        P = exact ? [a==b && a<=2 ? (-1.0)^((p-1)*(a-1))/sqrt(2) : 0.0 for p in 1:2,a in 1:D,b in 1:D] : randn(rng,2,D,D)
        N = exact ? copy(P) : randn(rng,2,D,D)
        vcat(vec(V/norm(V)),vec(P/norm(P)),vec(N/norm(N)))
    end
    return vcat(data...,zeros((uniform ? 4 : 16)*D))
end

function expanded_parameters(x,D)
    if length(x)==parameter_count(D;uniform=true)
        width = D^4+4D^2
        # Repeat uses a supported array pullback, unlike a flattened generator
        # whose Julia collect implementation internally mutates with push!.
        logs = repeat(reshape(x[width+1:end],D,4);inner=(1,4))
        return vcat(repeat(x[1:width],4),vec(logs))
    end
    return x
end

function split_parameters(parameters,D)
    # Do not rebind a captured parameter across a branch: Zygote must see one
    # immutable array binding when differentiating the slicing comprehensions.
    x = expanded_parameters(parameters,D)
    length(x)==parameter_count(D) || throw(DimensionMismatch("wrong split parameter count"))
    width = D^4+4D^2
    start(r,c) = (r-1+2*(c-1))*width
    unit(a) = a/norm(a)
    V = [unit(reshape(x[start(r,c)+1:start(r,c)+D^4],D,D,D,D)) for r in 1:2,c in 1:2]
    E = [unit(reshape(x[start(r,c)+D^4+1:start(r,c)+D^4+2D^2],2,D,D)) for r in 1:2,c in 1:2]
    N = [unit(reshape(x[start(r,c)+D^4+2D^2+1:start(r,c)+width],2,D,D)) for r in 1:2,c in 1:2]
    weights = map(CartesianIndices((2,2,4))) do i
        r,c,d = Tuple(i)
        offset = 4width + (r-1+2*(c-1)+4*(d-1))*D
        u = x[offset+1:offset+D]
        exp.(u .- sum(u)/D)
    end
    return (;V,E,N,weights)
end

scale_axis(a,w,ax) = a .* reshape(w,ntuple(i->i==ax ? length(w) : 1,ndims(a)))

function block_site(st,r,c)
    at(r,c,d) = st.weights[mod1(r,2),mod1(c,2),d]
    v = st.V[r,c]
    for d in 1:4
        v = scale_axis(v,sqrt.(at(r,c,d)),d)
    end
    pe = scale_axis(scale_axis(st.E[r,c],sqrt.(at(r,c,2)),2),sqrt.(at(r,c+1,4)),3)
    pn = scale_axis(scale_axis(st.N[r,c],sqrt.(at(r,c,1)),2),sqrt.(at(r-1,c,3)),3)
    @tensor t[p,q,n,e,s,w] := v[b,a,s,w]*pe[p,a,e]*pn[q,b,n]
    D = size(v,1)
    return reshape(t,4,D,D,D,D)
end

function blocked_arrays(x,D)
    st = split_parameters(x,D)
    return [block_site(st,r,c) for r in 1:2,c in 1:2]
end

function make_peps(x,D)
    p,v = ComplexSpace(4),ComplexSpace(D)
    arrays = blocked_arrays(x,D)
    tensors = map(arrays) do a
        # TensorKit 0.16's constructor pullback returns the full leg-shaped
        # array: supply that shape rather than a (4,D^4) matrix.
        TensorMap(ComplexF64.(a/norm(a)),p,v⊗v⊗v'⊗v')
    end
    # Validate constant leg spaces outside AD; the typed constructor is only a
    # differentiable wrapper around A, avoiding TensorKit's metadata LRU cache.
    Zygote.ignore() do
        InfinitePEPS(tensors)
    end
    return InfinitePEPS{eltype(tensors)}(tensors)
end

function snapshot_parameters(path)
    data = JSON.parsefile(path)
    D = Int(data["D"])
    Int(data["L"])==2 || error("snapshot must have a 2x2 vertex cell")
    ts = Dict((String(t["key"][1]),Int(t["key"][2]),Int(t["key"][3]))=>Float64.(t["data"]) for t in data["tensors"])
    ws = Dict(Tuple(Int.(w["key"]))=>Float64.(w["data"]) for w in data["weights"])
    tensors = [vcat(ts[("V",r,c)],ts[("E",r,c)],ts[("N",r,c)]) for r in 1:2,c in 1:2]
    logs = [log.(ws[(r,c,d)]) for r in 1:2,c in 1:2,d in 1:4]
    all(isfinite,reduce(vcat,vec(logs))) || error("snapshot has nonpositive weights")
    return vcat(vec(tensors)...,vec(logs)...),D,data
end

function perturb_parameters(x,D,seed;amplitude=1e-3)
    amplitude>0 || throw(ArgumentError("perturbation amplitude must be positive"))
    width = D^4+4D^2
    count = length(x)==parameter_count(D;uniform=true) ? 1 : 4
    y = copy(x)
    rng = MersenneTwister(seed)
    for cell in 0:count-1
        for localinds in (1:D^4,D^4+1:D^4+2D^2,D^4+2D^2+1:width)
            inds = localinds .+ cell*width
            noise = normalize(randn(rng,length(inds)))
            y[inds] += amplitude*norm(x[inds])*noise
        end
    end
    # Log weights remain unconstrained; the physical map centers each bond scale.
    y[count*width+1:end] += amplitude*randn(rng,length(x)-count*width)
    return y
end

function virtual_symmetry_defect(x,D)
    D==2 || throw(ArgumentError("diagnostic assumes the D=2 virtual representation diag(1,-1)"))
    st = split_parameters(x,D)
    signs = [1.0,-1.0]
    rows = NamedTuple[]
    for r in 1:2,c in 1:2
        v = st.V[r,c]
        transformed = v
        for ax in 1:4;transformed=scale_axis(transformed,signs,ax);end
        push!(rows,(;kind="V",r,c,defect=norm(v-transformed)/norm(v)))
        for (kind,p) in (("E",st.E[r,c]),("N",st.N[r,c]))
            transformed = scale_axis(scale_axis(p,signs,2),signs,3)
            push!(rows,(;kind,r,c,defect=norm(p-transformed)/norm(p)))
        end
    end
    return (;maximum=maximum(row.defect for row in rows),rows)
end

function operators(hx,hz)
    p = ComplexSpace(4)
    lattice = fill(p,2,2)
    x,z,i2 = [0.0 1;1 0],[1.0 0;0 -1],Matrix{Float64}(I,2,2)
    xe,xn,ze,zn = kron(i2,x),kron(x,i2),kron(i2,z),kron(z,i2)
    sop = TensorMap(ComplexF64.(kron(xn,kron(xe*xn,xe))),p^3,p^3)
    bop = TensorMap(ComplexF64.(kron(zn,kron(ze*zn,ze))),p^3,p^3)
    H = LocalOperator{Any,eltype(lattice)}(lattice)
    ops = Dict{String,Any}()
    function add(name,sites,op,coefficient)
        localop = LocalOperator{Any,eltype(lattice)}(lattice)
        PEPSKit.add_term!(localop,sites,op)
        ops[name] = localop
        coefficient==0 || PEPSKit.add_term!(H,sites,coefficient*op)
    end
    for r in 1:2,c in 1:2
        add("A$r$c",[CartesianIndex(r,c-1),CartesianIndex(r,c),CartesianIndex(r+1,c)],sop,-1.0)
        add("B$r$c",[CartesianIndex(r-1,c),CartesianIndex(r,c),CartesianIndex(r,c+1)],bop,-1.0)
        for (name,op,coefficient) in (("XE",xe,-hx),("XN",xn,-hx),("ZE",ze,-hz),("ZN",zn,-hz))
            add("$name$r$c",[CartesianIndex(r,c)],TensorMap(ComplexF64.(op),p,p),coefficient)
        end
    end
    return H,ops
end

# Reference data only: the published polynomial and its analytic derivatives.
include(joinpath(@__DIR__,"..","..","model","reference.jl"))

module SplitSU

using LinearAlgebra, Random
export SplitState, initial_state, pauli_evolve, plaquette_path, path_cluster,
       restore_path!, split_chain, join_chain, star_update!, plaquette_update!,
       sweep!, blocked_tensor, state_weights
export ring_cluster, ring_tensor, restore_ring!, loop_update!
export local_stabilizer_residuals

const Node = Tuple{Symbol,Int,Int}
const Bond = NTuple{3,Int} # (vertex row, vertex col, N/E/S/W = 1/2/3/4)
const PINV_RTOL = 1e-14

mutable struct SplitState
    tensors::Dict{Node,Array{Float64}}
    weights::Dict{Bond,Vector{Float64}}
    L::Int
end

wrap(st, node::Node) = (node[1], mod1(node[2], st.L), mod1(node[3], st.L))
bond(st, r, c, d) = (mod1(r, st.L), mod1(c, st.L), d)
tensor(st, n) = st.tensors[wrap(st, n)]

"Ordered tensor legs: V[N,E,S,W], P[physical,tail,head]."
function bonds(st, node::Node)
    k, r, c = node
    k == :V && return [bond(st, r, c, d) for d in 1:4]
    k == :E && return [nothing, bond(st, r, c, 2), bond(st, r, c+1, 4)]
    k == :N && return [nothing, bond(st, r, c, 1), bond(st, r-1, c, 3)]
    error("Unknown tensor $node")
end

function initial_state(kind::Symbol, seed::Int; L=2)
    rng = MersenneTwister(seed)
    ts = Dict{Node,Array{Float64}}()
    ws = Dict{Bond,Vector{Float64}}()
    for r in 1:L, c in 1:L
        if kind == :exact
            v = [iseven(n+e+s+w) ? 1.0 : 0.0 for n in 0:1, e in 0:1, s in 0:1, w in 0:1]
            p = [a == b ? (-1.0)^(q*a)/sqrt(2) : 0.0 for q in 0:1, a in 0:1, b in 0:1]
            ts[(:V,r,c)] = v/norm(v)
            ts[(:E,r,c)] = p/norm(p)
            ts[(:N,r,c)] = copy(p)/norm(p)
        elseif kind == :random
            for (k, shape) in ((:V,(2,2,2,2)), (:E,(2,2,2)), (:N,(2,2,2)))
                a = randn(rng, shape...)
                ts[(k,r,c)] = a/norm(a)
            end
        else
            error("Initialization must be random or exact")
        end
        for d in 1:4
            ws[(r,c,d)] = ones(2)
        end
    end
    return SplitState(ts, ws, L)
end

function axis_scale(a, w, ax)
    shape = ntuple(i -> i == ax ? length(w) : 1, ndims(a))
    return a .* reshape(w, shape)
end

pinverse(w) = [v > PINV_RTOL*maximum(w) ? inv(v) : 0.0 for v in w]

"Multiply axis ax by matrix m[new,old], retaining axis order."
function mode_product(a, m, ax)
    perm = [ax; [i for i in 1:ndims(a) if i != ax]]
    dims = size(a)[perm]
    b = m * reshape(permutedims(a, perm), dims[1], :)
    return permutedims(reshape(b, size(m,1), dims[2:end]...), invperm(perm))
end

"Exact exp(dt P), with the physical bit fastest within each indicated group."
function pauli_evolve(a, axes, kind::Symbol, dt)
    b = copy(a)
    for ax in axes
        q = size(a, ax)
        @assert iseven(q)
        op = zeros(q, q)
        for j in 1:q
            if kind == :X
                op[isodd(j) ? j+1 : j-1, j] = 1
            elseif kind == :Z
                op[j,j] = isodd(j) ? 1 : -1
            else
                error("Expected X or Z")
            end
        end
        b = mode_product(b, op, ax)
    end
    return cosh(dt)*a + sinh(dt)*b
end

"A seven-node path around a plaquette, excluding one vertex (corner=0:3)."
function plaquette_path(r, c, corner=0)
    ring = [(:V,r,c), (:E,r,c), (:V,r,c+1), (:N,r,c+1),
            (:V,r-1,c+1), (:E,r-1,c), (:V,r-1,c), (:N,r,c)]
    cut = 2mod(corner,4)+1
    return [ring[mod1(cut+i,8)] for i in 1:7]
end

"Contract a weighted path. All λ appear once internally and once per open leg."
function path_cluster(st, path)
    bs = [bonds(st,n) for n in path]
    links = [only(intersect(filter(!isnothing, bs[i]), filter(!isnothing, bs[i+1])))
             for i in 1:length(path)-1]
    gs = Array{Float64,3}[]
    meta = NamedTuple[]
    for i in eachindex(path)
        a = copy(tensor(st,path[i]))
        la = i > 1 ? findfirst(==(links[i-1]), bs[i]) : nothing
        ra = i < length(path) ? findfirst(==(links[i]), bs[i]) : nothing
        ex = [j for j in 1:ndims(a) if j != la && j != ra]
        for j in ex
            isnothing(bs[i][j]) || (a = axis_scale(a, st.weights[bs[i][j]], j))
        end
        # Internal weights are absorbed on the right endpoint only.
        isnothing(la) || (a = axis_scale(a, st.weights[links[i-1]], la))
        perm = [isnothing(la) ? Int[] : [la]; ex; isnothing(ra) ? Int[] : [ra]]
        dims = size(a)
        dl, dr = isnothing(la) ? 1 : dims[la], isnothing(ra) ? 1 : dims[ra]
        push!(gs, reshape(permutedims(a,perm), dl, prod(dims[ex]), dr))
        push!(meta, (; perm, ex, dims, bs=bs[i], links, la, ra))
    end
    qdims = [size(g,2) for g in gs]
    acc = reshape(gs[1], qdims[1], :)
    for g in gs[2:end]
        acc = reshape(acc * reshape(g, size(g,1), :), :, size(g,3))
    end
    return reshape(acc, qdims...), meta
end

"TT-SVD followed by a right canonical sweep, returning explicit Vidal Γ and λ."
function split_chain(x; maxdim=2)
    dims = collect(size(x)); n = length(dims)
    a = x / norm(x)
    gs = Array{Float64,3}[]
    dl = 1; discarded = 0.0
    for i in 1:n-1
        f = svd(reshape(a, dl*dims[i], :); full=false)
        k = min(maxdim, length(f.S))
        discarded = max(discarded, sum(abs2, f.S[k+1:end])/sum(abs2,f.S))
        push!(gs, reshape(f.U[:,1:k], dl, dims[i], k))
        a = Diagonal(f.S[1:k])*f.Vt[1:k,:]
        dl = k
    end
    push!(gs, reshape(a/norm(a), dl, dims[n], 1))
    ls = Vector{Float64}[Float64[] for _ in 1:n-1]
    for i in n:-1:2
        g = gs[i]
        f = svd(reshape(g, size(g,1), :); full=false)
        k = length(f.S)
        ls[i-1] = f.S / norm(f.S)
        gs[i] = reshape(f.Vt, k, size(g,2), size(g,3))
        prev = gs[i-1]
        gs[i-1] = reshape(reshape(prev, :, size(prev,3))*(f.U*Diagonal(f.S)),
                           size(prev,1), size(prev,2), k)
    end
    # Right-canonical chain B1 B2 ... = Γ1 λ1 Γ2 λ2 ... Γn.
    for i in 1:n-1
        gs[i] = axis_scale(gs[i], pinverse(ls[i]), 3)
    end
    return gs, ls, discarded
end

function join_chain(gs, ls)
    dims = [size(g,2) for g in gs]
    a = reshape(gs[1], dims[1], :)
    for i in 2:length(gs)
        a = reshape((a*Diagonal(ls[i-1]))*reshape(gs[i],size(gs[i],1),:), :, size(gs[i],3))
    end
    return reshape(a, dims...)
end

function restore_path!(st, path, theta, meta; maxdim=2)
    gs, ls, discarded = split_chain(theta; maxdim)
    for i in eachindex(path)
        m = meta[i]
        ds = collect(m.dims)
        isnothing(m.la) || (ds[m.la] = size(gs[i],1))
        isnothing(m.ra) || (ds[m.ra] = size(gs[i],3))
        a = permutedims(reshape(gs[i], ds[m.perm]...), invperm(m.perm))
        for j in m.ex
            isnothing(m.bs[j]) || (a = axis_scale(a, pinverse(st.weights[m.bs[j]]), j))
        end
        st.tensors[wrap(st,path[i])] = a/norm(a)
    end
    for (b, l) in zip(meta[1].links,ls)
        st.weights[b] = l
    end
    return discarded
end

function plaquette_update!(st, r, c, dt; corner=0)
    path = plaquette_path(r,c,corner)
    x, meta = path_cluster(st,path)
    phys = findall(n -> n[1] != :V, path)
    y = pauli_evolve(x, phys, :Z, dt)
    return restore_path!(st,path,y,meta)
end

"Four-arm simplex update using HOSVD in the diagonal external-weight metric."
function star_cluster(st, r, c; external_weights=true)
    vkey = (:V,r,c)
    leaves = [(:N,r,c), (:E,r,c), (:N,r+1,c), (:E,r,c-1)]
    vb = bonds(st,vkey)
    x = copy(tensor(st,vkey))
    leafinfo = NamedTuple[]
    for i in 1:4
        p = tensor(st,leaves[i]); pb = bonds(st,leaves[i])
        inner = findfirst(==(vb[i]),pb)
        outer = inner == 2 ? 3 : 2
        a = external_weights ? axis_scale(p,st.weights[pb[outer]],outer) : copy(p)
        a = axis_scale(a,st.weights[vb[i]],inner)
        l = reshape(permutedims(a,[1,outer,inner]),4,2)
        x = mode_product(x,l,i)
        push!(leafinfo,(; inner,outer,pb))
    end
    return x,leafinfo
end

function star_update!(st, r, c, dt)
    vkey = (:V,r,c)
    leaves = [(:N,r,c), (:E,r,c), (:N,r+1,c), (:E,r,c-1)]
    vb = bonds(st,vkey)
    x,leafinfo = star_cluster(st,r,c)
    y = pauli_evolve(x,1:4,:X,dt)
    us = Matrix{Float64}[]; ls = Vector{Float64}[]
    discarded = 0.0
    for i in 1:4
        perm = [i; [j for j in 1:4 if j != i]]
        f = svd(reshape(permutedims(y,perm),4,:);full=false)
        discarded = max(discarded,sum(abs2,f.S[3:end])/sum(abs2,f.S))
        push!(us,f.U[:,1:2]); push!(ls,f.S[1:2]/norm(f.S[1:2]))
    end
    v = copy(y)
    for i in 1:4
        v = mode_product(v,us[i]',i)
        v = axis_scale(v,pinverse(ls[i]),i)
        m = leafinfo[i]
        p = permutedims(reshape(us[i],2,2,2), invperm([1,m.outer,m.inner]))
        p = axis_scale(p,pinverse(st.weights[m.pb[m.outer]]),m.outer)
        st.tensors[wrap(st,leaves[i])] = p/norm(p)
        st.weights[vb[i]] = ls[i]
    end
    st.tensors[wrap(st,vkey)] = v/norm(v)
    return discarded
end

function sweep!(st, dt, step; rotate=true, plaquette=:path, als_sweeps=12)
    es = 0.0; ep = 0.0
    for r in 1:st.L, c in 1:st.L
        es = max(es,star_update!(st,r,c,dt))
        e = plaquette == :path ? plaquette_update!(st,r,c,dt;corner=rotate ? mod(step-1,4) : 0) :
                                loop_update!(st,r,c,dt;als_sweeps)
        ep = max(ep,e)
    end
    return es, ep
end

state_weights(st) = reduce(vcat,[st.weights[b] for b in sort(collect(keys(st.weights)))])

"Measurement-only blocking: outgoing E/N P, with every λ included exactly once."
function blocked_tensor(st,r,c)
    v = tensor(st,(:V,r,c)); pe = tensor(st,(:E,r,c)); pn = tensor(st,(:N,r,c))
    be = bonds(st,(:E,r,c)); bn = bonds(st,(:N,r,c))
    out = zeros(2,2,2,2,2,2) # physical E,N; virtual N,E,S,W
    for p in 1:2, q in 1:2, n in 1:2, e in 1:2, s in 1:2, w in 1:2, a in 1:2, b in 1:2
        out[p,q,n,e,s,w] += v[b,a,s,w] * pe[p,a,e] * pn[q,b,n] *
            st.weights[be[2]][a] * st.weights[be[3]][e] *
            st.weights[bn[2]][b] * st.weights[bn[3]][n]
    end
    return reshape(out,4,2,2,2,2)
end

include("loop.jl")

"Local stabilizer tensor identities, leaving every external virtual leg open."
function local_stabilizer_residuals(st)
    vals = Float64[]
    for r in 1:st.L,c in 1:st.L
        a,_ = star_cluster(st,r,c;external_weights=false)
        gs,_ = ring_cluster(st,r,c;external_weights=false)
        b = ring_tensor(gs)
        for (x,axes,kind) in ((a,1:4,:X),(b,[2,4,6,8],:Z))
            px = (pauli_evolve(x,axes,kind,1.0)-cosh(1.0)*x)/sinh(1.0)
            push!(vals,norm(px-x)/norm(x))
        end
    end
    return vals
end

end # module

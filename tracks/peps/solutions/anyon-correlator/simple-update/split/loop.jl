# Included inside SplitSU. A loop-aware cluster update, not ordinary path-SU.
# External environment = independent diagonal λ, never CTMRG/full-update.

ring_nodes(r,c) = [(:V,r,c),(:E,r,c),(:V,r,c+1),(:N,r,c+1),
                   (:V,r-1,c+1),(:E,r-1,c),(:V,r-1,c),(:N,r,c)]

function ring_cluster(st,r,c;external_weights=true)
    nodes = ring_nodes(r,c)
    bs = [bonds(st,n) for n in nodes]
    links = [only(intersect(filter(!isnothing,bs[i]),filter(!isnothing,bs[mod1(i+1,8)]))) for i in 1:8]
    gs = Array{Float64,3}[]; meta = NamedTuple[]
    for i in 1:8
        a = copy(tensor(st,nodes[i])); dims = size(a)
        la = findfirst(==(links[mod1(i-1,8)]),bs[i])
        ra = findfirst(==(links[i]),bs[i])
        ex = [j for j in 1:ndims(a) if j != la && j != ra]
        for j in ex
            (isnothing(bs[i][j]) || !external_weights) || (a=axis_scale(a,st.weights[bs[i][j]],j))
        end
        a = axis_scale(a,st.weights[links[i]],ra)
        perm = [la;ex;ra]
        push!(gs,reshape(permutedims(a,perm),2,prod(dims[ex]),2))
        push!(meta,(;dims,ex,la,ra,perm,bs=bs[i],links))
    end
    return gs,meta
end

function open_chain(gs)
    dl = size(gs[1],1)
    a = reshape(gs[1],:,size(gs[1],3))
    for g in gs[2:end]
        a = reshape(a*reshape(g,size(g,1),:),:,size(g,3))
    end
    return reshape(a,dl,:,size(gs[end],3))
end

function ring_tensor(gs)
    a = open_chain(gs)
    x = zeros(size(a,2))
    for i in axes(a,1)
        x .+= a[i,:,i]
    end
    return reshape(x,(size(g,2) for g in gs)...)
end

"ALS in the local weighted-cluster Frobenius metric. No global environment."
function fit_ring!(gs,target;maxiter=12,tol=1e-11)
    target = target/norm(target)
    gs[1] ./= norm(ring_tensor(gs))
    olderr = Inf; err = Inf
    for sweep in 1:maxiter
        for i in 1:8
            order = [mod1(i+j,8) for j in 1:7]
            a = open_chain(gs[order]) # right_i, external groups, left_i
            design = reshape(permutedims(a,(2,3,1)),size(a,2),4)
            rhs = reshape(permutedims(target,[order;i]),:,size(gs[i],2))
            f = svd(design;full=false)
            sinv = [s > 1e-12*f.S[1] ? inv(s) : 0.0 for s in f.S]
            sol = f.V*Diagonal(sinv)*(f.U'*rhs)
            gs[i] = permutedims(reshape(sol,2,2,size(gs[i],2)),(1,3,2))
        end
        err = norm(ring_tensor(gs)-target)^2
        (err < tol^2 || abs(err-olderr) < tol*max(err,1e-12)) && break
        olderr = err
    end
    return err
end

function psdroot(m;inverse=false)
    f = eigen(Symmetric((m+m')/2))
    maxv = maximum(f.values)
    vals = [v > 1e-14*maxv ? (inverse ? inv(sqrt(v)) : sqrt(v)) : 0.0 for v in f.values]
    return f.vectors*Diagonal(vals)*f.vectors'
end

"Cavity messages along a periodic MPS; used only to choose internal λ gauges."
function ring_gauge(gs)
    left = [Matrix{Float64}(I,2,2) for _ in 1:8]
    right = deepcopy(left)
    for iter in 1:200
        err = 0.0
        for i in 1:8
            prev = left[mod1(i-1,8)]
            l = zeros(2,2)
            for q in axes(gs[i],2)
                g = gs[i][:,q,:]
                l .+= g'*prev*g
            end
            l ./= tr(l)
            err = max(err,norm(l-left[i])); left[i]=l
        end
        for i in 8:-1:1
            j = mod1(i+1,8); nxt = right[j]
            rr = zeros(2,2)
            for q in axes(gs[j],2)
                g = gs[j][:,q,:]
                rr .+= g*nxt*g'
            end
            rr ./= tr(rr)
            err = max(err,norm(rr-right[i]));right[i]=rr
        end
        err < 1e-11 && break
    end
    hs = deepcopy(gs); ls = Vector{Float64}[]
    for i in 1:8
        j = mod1(i+1,8)
        f = svd(psdroot(left[i])*psdroot(right[i]);full=false)
        # H_i' S H_j = I: gauges + explicit λ preserve the ring state.
        hi = f.U'*psdroot(left[i];inverse=true)
        hj = f.Vt*psdroot(right[i];inverse=true)
        hs[i] = mode_product(hs[i],hi,3)
        hs[j] = mode_product(hs[j],hj,1)
        push!(ls,f.S)
    end
    return hs,ls
end

function restore_ring!(st,r,c,gs,meta)
    hs,ls = ring_gauge(gs)
    nodes = ring_nodes(r,c)
    for i in 1:8
        m = meta[i]
        a = permutedims(reshape(hs[i],m.dims[m.perm]...),invperm(m.perm))
        for j in m.ex
            isnothing(m.bs[j]) || (a=axis_scale(a,pinverse(st.weights[m.bs[j]]),j))
        end
        st.tensors[wrap(st,nodes[i])] = a/norm(a)
    end
    for (b,l) in zip(meta[1].links,ls)
        st.weights[b] = l/norm(l)
    end
    return st
end

function loop_update!(st,r,c,dt;als_sweeps=12)
    gs,meta = ring_cluster(st,r,c)
    x = ring_tensor(gs)
    target = pauli_evolve(x,[2,4,6,8],:Z,dt)
    err = fit_ring!(gs,target;maxiter=als_sweeps)
    restore_ring!(st,r,c,gs,meta)
    return err
end

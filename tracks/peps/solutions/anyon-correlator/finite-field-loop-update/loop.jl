# General-D extension of the independently tested zero-field loop update.
ring_nodes(r,c)=[(:V,r,c),(:E,r,c),(:V,r,c+1),(:N,r,c+1),(:V,r-1,c+1),(:E,r-1,c),(:V,r-1,c),(:N,r,c)]
function ring_cluster(st,r,c)
    nodes=ring_nodes(r,c);bs=[bonds(st,n) for n in nodes]
    links=[only(intersect(filter(!isnothing,bs[i]),filter(!isnothing,bs[mod1(i+1,8)]))) for i in 1:8]
    gs=Array{Float64,3}[];meta=NamedTuple[]
    for i in 1:8
        a=copy(tensor(st,nodes[i]));dims=size(a)
        la=findfirst(==(links[mod1(i-1,8)]),bs[i]);ra=findfirst(==(links[i]),bs[i])
        ex=[j for j in 1:ndims(a) if j!=la && j!=ra]
        for j in ex;isnothing(bs[i][j]) || (a=axis_scale(a,st.weights[bs[i][j]],j));end
        a=axis_scale(a,st.weights[links[i]],ra);perm=[la;ex;ra]
        push!(gs,reshape(permutedims(a,perm),st.D,prod(dims[ex]),st.D))
        push!(meta,(;dims,ex,perm,bs=bs[i],links))
    end
    return gs,meta
end
function open_chain(gs)
    dl=size(gs[1],1);a=reshape(gs[1],:,size(gs[1],3))
    for g in gs[2:end];a=reshape(a*reshape(g,size(g,1),:),:,size(g,3));end
    return reshape(a,dl,:,size(gs[end],3))
end
function ring_tensor(gs)
    # Balanced contraction avoids a full physical tensor with two extra D legs.
    mid=length(gs)÷2
    a=open_chain(gs[1:mid]);b=open_chain(gs[mid+1:end])
    x=reshape(permutedims(a,(2,1,3)),size(a,2),:)*reshape(permutedims(b,(3,1,2)),:,size(b,2))
    return reshape(x,(size(g,2) for g in gs)...)
end
"Exact QR factorization of the open-chain map: physical outputs × two boundary bonds."
function environment_qr(gs)
    D=size(gs[1],1);R=reshape(Matrix{Float64}(I,D,D),1,D,D)
    qs=Array{Float64,3}[]
    for g in gs
        rank=size(R,1);q=size(g,2);dr=size(g,3)
        a=reshape(reshape(R,rank*D,:)*reshape(g,size(g,1),:),rank,D,q,dr)
        mat=reshape(permutedims(a,(1,3,2,4)),rank*q,D*dr)
        f=qr(mat);k=min(size(mat)...)
        Q=Matrix(f.Q)[:,1:k]
        push!(qs,reshape(Q,rank,q,k))
        R=reshape(Matrix(f.R)[1:k,:],k,D,dr)
    end
    # Column convention of the ALS unknown: left_i fastest, right_i slowest.
    design=reshape(permutedims(R,(1,3,2)),size(R,1),:)
    return qs,design
end
"Apply Q† to another open-chain tensor without forming its physical output."
function project_chain(qs,gs)
    D=size(gs[1],1);R=reshape(Matrix{Float64}(I,D,D),1,D,D)
    for (Q,g) in zip(qs,gs)
        rank=size(R,1);q=size(g,2);dr=size(g,3)
        a=reshape(reshape(R,rank*D,:)*reshape(g,size(g,1),:),rank,D,q,dr)
        mat=reshape(permutedims(a,(1,3,2,4)),rank*q,D*dr)
        R=reshape(reshape(Q,rank*q,:)'*mat,size(Q,3),D,dr)
    end
    return R
end
function projected_rhs(qs,order,i,cores,coefficients)
    rhs=zeros(size(qs[end],3),size(cores[1][i],2))
    for (old,c) in zip(cores,coefficients)
        R=project_chain(qs,old[order])
        g=old[i]
        rhs .+= c*(reshape(permutedims(R,(1,3,2)),size(R,1),:)*reshape(permutedims(g,(1,3,2)),:,size(g,2)))
    end
    return rhs
end
function fit_ring!(gs,target;maxiter=12,tol=1e-11,target_cores=nothing,coefficients=nothing)
    targetnorm=norm(target)
    target=target/targetnorm;gs[1] ./= norm(ring_tensor(gs))
    before=norm(ring_tensor(gs)-target)^2
    olderr=before;err=before;iterations=0;converged=false
    # Precompute RHS permutations, which do not change within ALS.
    orders=[[mod1(i+j,8) for j in 1:7] for i in 1:8]
    rhs=isnothing(target_cores) ? [reshape(permutedims(target,[orders[i];i]),:,size(gs[i],2)) for i in 1:8] : nothing
    for sweep in 1:maxiter
        for i in 1:8
            dl,dr=size(gs[i],1),size(gs[i],3)
            if isnothing(target_cores)
                a=open_chain(gs[orders[i]])
                design=reshape(permutedims(a,(2,3,1)),size(a,2),dl*dr)
                b=rhs[i]
            else
                qs,design=environment_qr(gs[orders[i]])
                b=projected_rhs(qs,orders[i],i,target_cores,coefficients)/targetnorm
            end
            f=svd(design;full=false)
            sinv=[s>1e-12*f.S[1] ? inv(s) : 0.0 for s in f.S]
            sol=f.V*Diagonal(sinv)*(f.U'*b)
            gs[i]=permutedims(reshape(sol,dl,dr,size(gs[i],2)),(1,3,2))
        end
        iterations=sweep;err=norm(ring_tensor(gs)-target)^2
        if err<tol^2 || abs(err-olderr)<tol*max(err,1e-12)
            converged=true;break
        end
        olderr=err
    end
    return (;fit_before=before,fit_after=err,als_iterations=iterations,als_converged=converged)
end
function psdroot(m;inverse=false)
    f=eigen(Symmetric((m+m')/2));maxv=maximum(f.values)
    vals=[v>1e-14*maxv ? (inverse ? inv(sqrt(v)) : sqrt(v)) : 0.0 for v in f.values]
    return f.vectors*Diagonal(vals)*f.vectors'
end
function ring_gauge(gs)
    D=size(gs[1],1)
    left=[Matrix{Float64}(I,D,D) for _ in 1:8];right=deepcopy(left)
    err=Inf;iterations=0
    for iter in 1:200
        err=0.0
        for i in 1:8
            l=zeros(D,D)
            for q in axes(gs[i],2)
                g=gs[i][:,q,:];l .+= g'*left[mod1(i-1,8)]*g
            end
            l ./= tr(l);err=max(err,norm(l-left[i]));left[i]=l
        end
        for i in 8:-1:1
            j=mod1(i+1,8);rr=zeros(D,D)
            for q in axes(gs[j],2)
                g=gs[j][:,q,:];rr .+= g*right[j]*g'
            end
            rr ./= tr(rr);err=max(err,norm(rr-right[i]));right[i]=rr
        end
        iterations=iter;err<1e-11 && break
    end
    hs=deepcopy(gs);ls=Vector{Float64}[]
    for i in 1:8
        j=mod1(i+1,8);f=svd(psdroot(left[i])*psdroot(right[i]);full=false)
        hs[i]=mode_product(hs[i],f.U'*psdroot(left[i];inverse=true),3)
        hs[j]=mode_product(hs[j],f.Vt*psdroot(right[i];inverse=true),1)
        push!(ls,f.S)
    end
    # Evaluate the actual message fixed-point residual after the whole sweep.
    residual=0.0
    for i in 1:8
        l=zeros(D,D);rr=zeros(D,D);j=mod1(i+1,8)
        for q in axes(gs[i],2);g=gs[i][:,q,:];l .+= g'*left[mod1(i-1,8)]*g;end
        for q in axes(gs[j],2);g=gs[j][:,q,:];rr .+= g*right[j]*g';end
        residual=max(residual,norm(l/tr(l)-left[i]),norm(rr/tr(rr)-right[i]))
    end
    return hs,ls,(;bp_residual=residual,bp_iterations=iterations,bp_converged=residual<1e-11)
end
function restore_ring!(st,r,c,gs,meta)
    before=ring_tensor(gs)
    hs,ls,bp=ring_gauge(gs)
    weighted=[axis_scale(hs[i],ls[i],3) for i in 1:8]
    after=ring_tensor(weighted)
    gauge_error=norm(before/norm(before)-after/norm(after))
    nodes=ring_nodes(r,c)
    for i in 1:8
        m=meta[i];a=permutedims(reshape(hs[i],m.dims[m.perm]...),invperm(m.perm))
        for j in m.ex;isnothing(m.bs[j]) || (a=axis_scale(a,pinverse(st.weights[m.bs[j]]),j));end
        st.tensors[wrap(st,nodes[i])]=a/norm(a)
    end
    for (b,l) in zip(meta[1].links,ls);st.weights[b]=l/norm(l);end
    return (;bp...,gauge_error)
end
function loop_update!(st,r,c,dt;als_sweeps=12)
    gs,meta=ring_cluster(st,r,c)
    target=pauli_evolve(ring_tensor(gs),[2,4,6,8],:Z,dt)
    old=deepcopy(gs);flipped=deepcopy(gs)
    for i in (2,4,6,8);flipped[i]=mode_product(flipped[i],[1.0 0;0 -1],2);end
    fit=fit_ring!(gs,target;maxiter=als_sweeps,target_cores=(old,flipped),coefficients=(cosh(dt),sinh(dt)))
    gauge=restore_ring!(st,r,c,gs,meta)
    return (;fit...,gauge...)
end

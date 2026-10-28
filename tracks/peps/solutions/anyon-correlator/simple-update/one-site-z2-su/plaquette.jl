# Ring order V, E, V, N, V, E, V, N. V external axes retain geometric order.
const RING_AXES=((1,2),(2,3),(4,1),(3,2),(3,4),(3,2),(2,3),(2,3))
function ring_factors(st;external_weights=true)
    gs=AbstractTensorMap[]
    for i in 1:8
        t=outgoing(isodd(i) ? st.V : st.P); la,ra=RING_AXES[i]
        ex=Tuple(j for j in 1:numind(t) if j!=la && j!=ra)
        for j in ex
            external_weights && isodd(i) && (t=mode(t,st.lambda,j))
        end
        t=mode(t,st.lambda,ra)
        push!(gs,permute(t,((la,ex...,ra),())))
    end
    gs
end
function ring_labels(gs)
    labels=Vector{Int}[]; k=0
    for i in 1:8
        nex=numind(gs[i])-2
        push!(labels,[mod1(i-1,8);[-(k+j) for j in 1:nex];i]); k+=nex
    end
    labels
end
ring_join(gs)=contract(gs,ring_labels(gs))
plaquette_tensor(st;external_weights=true)=ring_join(ring_factors(st;external_weights))
function plaquette_step!(st,dt;als_sweeps=12)
    als_sweeps>0 || throw(ArgumentError("positive ALS sweeps required"))
    gs=ring_factors(st); labels=ring_labels(gs)
    y=evolve(ring_join(gs),[3,6,9,12],:Z,dt); y/=norm(y)
    gs[1]/=norm(ring_join(gs))
    for sweep in 1:als_sweeps
        for i in 1:8
            others=[j for j in 1:8 if j!=i]
            # Exposed bonds become the last two output axes of the design tensor.
            own=-labels[i][2:end-1]; ext=sort([-l for j in others for l in labels[j] if l<0])
            labs=deepcopy(labels[others]); linkleft=mod1(i-1,8); linkright=i
            for row in labs, j in eachindex(row)
                l=row[j]
                row[j]=l==linkleft ? -length(ext)-1 : l==linkright ? -length(ext)-2 :
                    l<0 ? -findfirst(==(-l),ext) : l
            end
            d=contract(gs[others],labs)
            m=permute(d,(Tuple(1:length(ext)),(length(ext)+1,length(ext)+2)))
            rhs=permute(y,(Tuple(ext),Tuple(own)))
            sol=pinv(m;rtol=1e-12)*rhs
            # sol order left,right; own external. Restore left,external,right.
            gs[i]=outgoing(permute(sol,((1,Tuple(3:numind(sol))...,2),())))
        end
    end
    fit_residual=norm(ring_join(gs)-y)^2
    before=ring_join(gs)
    hs,ls=ring_gauge(gs)
    hs=align_ring(st,hs)
    weighted=[mode(hs[i],ls[i],numind(hs[i])) for i in 1:8]
    after=ring_join(weighted)
    lnew=DiagonalTensorMap(sum(l/norm(l) for l in ls)/8); lnew/=norm(lnew)
    # Absorb the exact change of every internal weight into its right-axis factor.
    for i in 1:8
        hs[i]=mode(hs[i],support_inverse(lnew)*ls[i],numind(hs[i]))
    end
    vs=AbstractTensorMap[]; ps=AbstractTensorMap[]
    for i in 1:8
        la,ra=RING_AXES[i]; n=isodd(i) ? 4 : 3
        ex=Tuple(j for j in 1:n if j!=la && j!=ra)
        order=(la,ex...,ra); t=permute(hs[i],(Tuple(invperm(collect(order))),()))
        if isodd(i)
            for j in ex; t=mode(t,support_inverse(lnew),j); end
            push!(vs,t)
        else
            push!(ps,t)
        end
    end
    d=commit_shared!(st,vs,ps,fill(lnew,8),before,after)
    return (;fit_residual,discarded_weight=nothing,d...)
end
function matrix_map(t)
    q=virtual_space(); ref=TensorMap(zeros(2,2),q←q)
    orient_like(permute(t,((1,),(2,))),ref)
end
function cavity(g,l;right=false)
    n=numind(g); ex=collect(1:n-2)
    if right
        labs1=[-1;ex;n]; labs2=[-2;ex;n+1]; ll=[n,n+1]
    else
        labs1=[n;ex;-1]; labs2=[n+1;ex;-2]; ll=[n,n+1]
    end
    a=matrix_map(contract([g,g,l],[labs1,labs2,ll]))
    tr(a)>0 || error("zero cavity support")
    return a/tr(a)
end
function ring_gauge(gs)
    q=virtual_space(); eye=id(Float64,q)
    left=[copy(eye) for _ in 1:8]; right=deepcopy(left)
    for iter in 1:200
        err=0.0
        for i in 1:8
            l=cavity(gs[i],left[mod1(i-1,8)])
            err=max(err,norm(l-left[i])); left[i]=l
        end
        for i in 8:-1:1
            j=mod1(i+1,8); r=cavity(gs[j],right[j];right=true)
            err=max(err,norm(r-right[i])); right[i]=r
        end
        err<1e-11 && break
    end
    hs=copy(gs); ls=AbstractTensorMap[]
    for i in 1:8
        j=mod1(i+1,8); l=psd_root(left[i]); r=psd_root(right[i])
        u,s,v,_=svd_trunc(l*r;trunc=truncspace(q))
        hs[i]=mode(hs[i],u'*support_inverse(l),numind(hs[i]))
        hs[j]=mode(hs[j],v*support_inverse(r),1)
        push!(ls,s)
    end
    return hs,ls
end
"Positive root in the supported charge blocks, without complex roundoff roots."
function psd_root(t)
    result=zero(t)
    maxvalue=maximum(real(b[1,1]) for (_,b) in blocks(t))
    for (c,b) in blocks(t)
        size(b)==(1,1) || error("D=2 requires one-dimensional charge blocks")
        x=real(b[1,1]); x >= -1e-14*maxvalue || error("non-positive cavity block")
        block(result,c)[1,1]=x>1e-14*maxvalue ? sqrt(x) : 0.0
    end
    result
end
function local_residuals(st)
    s=star_tensor(st;external_weights=false); p=plaquette_tensor(st;external_weights=false)
    sx=s; pz=p
    for i in [1,3,5,7]; sx=pauli_mode(sx,i,:X); end
    for i in [3,6,9,12]; pz=pauli_mode(pz,i,:Z); end
    return (star=norm(sx-s)/norm(s),plaquette=norm(pz-p)/norm(p))
end

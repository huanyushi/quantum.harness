"All outgoing legs, with the canonical real Z₂ identification on dual spaces."
function outgoing(t)
    a=permute(t,(Tuple(1:numind(t)),()))
    inds=Tuple(i for i in 1:numind(a) if isdual(space(a,i)))
    return isempty(inds) ? a : flip(a,inds)
end
"Contract native charge blocks; labels follow ncon's negative output convention."
function contract(ts,labels)
    maps=AbstractTensorMap[outgoing(t) for t in ts]
    seen=Set{Int}()
    for i in eachindex(maps)
        flips=Int[]
        for (j,l) in enumerate(labels[i])
            l>0 || continue
            l in seen ? push!(flips,j) : push!(seen,l)
        end
        isempty(flips) || (maps[i]=flip(maps[i],Tuple(flips)))
    end
    return outgoing(TensorKit.ncon(maps,labels))
end
function mode(t,m,axis)
    labs=[-i for i in 1:numind(t)]; labs[axis]=1
    contract([t,m],[labs,[-axis,1]])
end
support_inverse(l)=pinv(l;rtol=1e-14)
function validate_charge_gauge(m)
    norm(m*Diagonal([1.,-1.])-Diagonal([1.,-1.])*m)<=1e-12 || throw(ArgumentError("gauge mixes charges"))
    return m
end
function gauge_pair_residual(st,g)
    a=contract([st.P,st.lambda,st.P],[[-1,-2,1],[1,2],[-3,2,-4]])
    b=contract([mode(st.P,g,3),st.lambda,mode(st.P,g,2)],[[-1,-2,1],[1,2],[-3,2,-4]])
    return norm(a-b)/norm(a)
end
function canonical_v(t)
    q=virtual_space(); ref=TensorMap(zeros(2,2,2,2),q⊗q←q⊗q)
    orient_like(permute(t,((1,2),(3,4))),ref)
end
function canonical_p(t)
    q=virtual_space(); h=physical_space(); ref=TensorMap(zeros(2,2,2),h⊗q←q)
    orient_like(permute(t,((1,2),(3,))),ref)
end
"Independent real signs within each one-dimensional virtual charge space."
function charge_frame(p,reference)
    a=canonical_p(p); b=canonical_p(reference); q=virtual_space(); g=id(Float64,q)
    for c in sectors(q)
        block(g,c)[1,1]=real(dot(block(a,c),block(b,c)))<0 ? -1.0 : 1.0
    end
    return g
end
function align_star(st,v,ps)
    ps=copy(ps)
    for i in 1:4
        g=charge_frame(ps[i],st.P)
        ps[i]=mode(ps[i],g,i<=2 ? 2 : 3)
        v=mode(v,g,i)
    end
    return v,ps
end
function align_ring(st,hs)
    hs=copy(hs)
    for i in 2:2:8
        la,ra=RING_AXES[i]; ex=Tuple(j for j in 1:3 if j!=la && j!=ra)
        order=(la,ex...,ra)
        p=permute(hs[i],(Tuple(invperm(collect(order))),()))
        g=charge_frame(p,st.P); j=mod1(i+1,8)
        hs[i]=mode(hs[i],g,numind(hs[i])); hs[j]=mode(hs[j],g,1)
    end
    # P is diagonal in the virtual charges. Applying Z to both endpoints
    # leaves P unchanged but changes the adjacent V frames. Resolve those
    # four remaining binary choices using the preceding shared V.
    q=virtual_space(); z=TensorMap([1.0 0;0 -1],q←q)
    best=hs; bestcost=Inf
    for mask in 0:15
        candidate=copy(hs)
        for (bit,i) in enumerate(2:2:8)
            iszero(mask & (1<<(bit-1))) && continue
            prev=mod1(i-1,8); nxt=mod1(i+1,8)
            candidate[prev]=mode(candidate[prev],z,numind(candidate[prev]))
            candidate[nxt]=mode(candidate[nxt],z,1)
        end
        cost=0.0
        for i in 1:2:8
            la,ra=RING_AXES[i]; ex=Tuple(j for j in 1:4 if j!=la && j!=ra)
            order=(la,ex...,ra); v=canonical_v(permute(candidate[i],(Tuple(invperm(collect(order))),())))
            cost+=min(norm(v/norm(v)-st.V),norm(v/norm(v)+st.V))^2
        end
        if cost<bestcost; bestcost=cost; best=candidate; end
    end
    return best
end
function commit_shared!(st,vs,ps,ls,before,after)
    relative(a,b)=min(norm(a/norm(a)-b/norm(b)),norm(a/norm(a)+b/norm(b)))
    gauge_residual=relative(before,after)
    gauge_residual<=1e-8 || error("gauge changed cluster: $gauge_residual")
    phase(a,b)=real(dot(a,b))<0 ? -a/norm(a) : a/norm(a)
    v=sum(phase(canonical_v(x),st.V) for x in vs)/length(vs)
    p=sum(phase(canonical_p(x),st.P) for x in ps)/length(ps)
    vp,pp=spatial_project(v,p)
    sharing_change=max(norm(v-vp)/norm(v),norm(p-pp)/norm(p),
        maximum(relative(canonical_v(x),vp) for x in vs),maximum(relative(canonical_p(x),pp) for x in ps))
    norm(vp)>0 && norm(pp)>0 || error("shared projection has zero norm")
    st.V=vp/norm(vp); st.P=pp/norm(pp)
    st.lambda=DiagonalTensorMap(sum(ls)/length(ls)); st.lambda/=norm(st.lambda)
    return (;gauge_residual,sharing_change,symmetry=symmetry_residuals(st))
end
function sweep!(st,dt;als_sweeps=12)
    als_sweeps>0 || throw(ArgumentError("positive ALS sweeps required"))
    candidate=deepcopy(st)
    a=star_step!(candidate,dt); b=plaquette_step!(candidate,dt;als_sweeps)
    st.V=candidate.V; st.P=candidate.P; st.lambda=candidate.lambda
    return (;star=a,plaquette=b)
end
function pauli_mode(t,axis,kind)
    h=physical_space(); m=kind==:X ? [0. 1;1 0] : [1. 0;0 -1]
    mode(t,TensorMap(m,h←h),axis)
end
function evolve(t,axes,kind,dt)
    b=t
    for i in axes; b=pauli_mode(b,i,kind); end
    cosh(dt)*t+sinh(dt)*b
end

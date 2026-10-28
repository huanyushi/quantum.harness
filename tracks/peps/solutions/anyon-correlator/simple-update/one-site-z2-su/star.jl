function star_arms(st;external_weights=true)
    [mode(external_weights ? mode(st.P,st.lambda,outer) : outgoing(st.P),st.lambda,inner)
     for (inner,outer) in ((2,3),(2,3),(3,2),(3,2))]
end
function star_join(v,ps)
    contract([v;ps],[[1,2,3,4],[-1,1,-2],[-3,2,-4],[-5,-6,3],[-7,-8,4]])
end
star_tensor(st;external_weights=true)=star_join(st.V,star_arms(st;external_weights))
function star_step!(st,dt)
    y=evolve(star_tensor(st),[1,3,5,7],:X,dt)
    us=AbstractTensorMap[]; ls=AbstractTensorMap[]; discarded=0.0
    for i in 1:4
        arm=(2i-1,2i); rest=Tuple(j for j in 1:8 if !(j in arm))
        m=permute(y,(arm,rest))
        u,s,_,err=svd_trunc(m;trunc=truncspace(virtual_space()))
        push!(us,u); push!(ls,s/norm(s)); discarded=max(discarded,err^2/norm(m)^2)
    end
    # Contract the four retained isometries back into the central vertex.
    # Production tensors and SVD factors are real in this milestone.
    v=contract([y;us],[[1,2,3,4,5,6,7,8], [1,2,-1],[3,4,-2],[5,6,-3],[7,8,-4]])
    for i in 1:4; v=mode(v,support_inverse(ls[i]),i); end
    ps=AbstractTensorMap[]
    for i in 1:4
        p=outgoing(us[i]); p=permute(p,((1,3,2),()))
        i>2 && (p=permute(p,((1,3,2),())))
        outer=i<=2 ? 3 : 2
        push!(ps,mode(p,support_inverse(st.lambda),outer))
    end
    v,ps=align_star(st,v,ps)
    # Each internal bond weight is rewritten with exact compensation on P.
    lnew=DiagonalTensorMap(sum(ls)/4); lnew/=norm(lnew)
    before=star_join(v,[mode(mode(ps[i],ls[i],i<=2 ? 2 : 3),st.lambda,i<=2 ? 3 : 2) for i in 1:4])
    for i in 1:4; ps[i]=mode(ps[i],support_inverse(lnew)*ls[i],i<=2 ? 2 : 3); end
    after=star_join(v,[mode(mode(ps[i],lnew,i<=2 ? 2 : 3),st.lambda,i<=2 ? 3 : 2) for i in 1:4])
    # Reweight external legs before sharing P with unchanged sites.
    for i in 1:4; ps[i]=mode(ps[i],support_inverse(lnew)*st.lambda,i<=2 ? 3 : 2); end
    d=commit_shared!(st,[v],ps,fill(lnew,4),before,after)
    return (;fit_residual=nothing,discarded_weight=discarded,d...)
end

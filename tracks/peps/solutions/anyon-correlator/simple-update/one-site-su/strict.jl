module StrictOneSiteSU
using LinearAlgebra, Statistics
include("core.jl")
const K=OneSiteSU.SplitSU

mutable struct State
    V::Array{Float64,4}
    P::Array{Float64,3}
    lambda::Vector{Float64}
end
function rotation_project(v)
    result=similar(v);done=Set{NTuple{4,Int}}()
    for i in CartesianIndices(v)
        t=Tuple(i);t in done && continue
        orbit=unique([t,(t[2],t[3],t[4],t[1]),(t[3],t[4],t[1],t[2]),(t[4],t[1],t[2],t[3])])
        value=mean(v[j...] for j in orbit)
        for j in orbit;result[j...]=value;push!(done,j);end
    end
    return result
end
function commit!(st,v,p,l)
    a=rotation_project(v)
    b=(p+permutedims(p,(1,3,2)))/2
    norm(a)>1e-14 && norm(b)>1e-14 || error("spatial projection produced a zero tensor")
    all(isfinite,a) && all(isfinite,b) || error("nonfinite tensor")
    st.V=a/norm(a);st.P=b/norm(b)
    st.lambda=normalize(max.(l,1e-12))
    return st
end
function random_state(seed)
    old=OneSiteSU.random_state(seed)
    st=State(copy(old.tensors[(:V,1,1)]),copy(old.tensors[(:E,1,1)]),ones(2))
    return commit!(st,st.V,st.P,st.lambda)
end
function workspace(st)
    ts=Dict{K.Node,Array{Float64}}();ws=Dict{K.Bond,Vector{Float64}}()
    for r in 1:2,c in 1:2
        ts[(:V,r,c)]=copy(st.V)
        ts[(:E,r,c)]=copy(st.P);ts[(:N,r,c)]=copy(st.P)
        for d in 1:4;ws[(r,c,d)]=copy(st.lambda);end
    end
    return K.SplitState(ts,ws,2)
end
defect(st)=max(norm(st.V-permutedims(st.V,(2,3,4,1))),norm(st.P-permutedims(st.P,(1,3,2))))
relative(a,b)=min(norm(a/norm(a)-b/norm(b)),norm(a/norm(a)+b/norm(b)))

function endpoints(work,b)
    r,c,d=b;v=(:V,r,c)
    p,ax=d==1 ? ((:N,r,c),2) : d==2 ? ((:E,r,c),2) :
         d==3 ? ((:N,r+1,c),3) : ((:E,r,c-1),3)
    return v,d,K.wrap(work,p),ax
end
"Exact bond gauge: V'=Q V, P'=lambda_new^-1 Q lambda_old P."
function gauge_bond!(work,b,q,lnew)
    vn,va,pn,pa=endpoints(work,b)
    w=work.weights[b]
    work.tensors[vn]=K.mode_product(work.tensors[vn],q,va)
    work.tensors[pn]=K.mode_product(work.tensors[pn],Diagonal(1 ./lnew)*q*Diagonal(w),pa)
    work.weights[b]=copy(lnew)
    return work
end
function reweight_external!(work,b,lnew,node)
    axis=findfirst(==(b),K.bonds(work,node))
    work.tensors[node]=K.axis_scale(work.tensors[node],work.weights[b]./lnew,axis)
    work.weights[b]=copy(lnew)
end
weighted_p(p,l,r)=K.axis_scale(K.axis_scale(p,l,2),r,3)
function orthofit(a,b,axis)
    perm=axis==2 ? (2,1,3) : (3,1,2)
    am=reshape(permutedims(a,perm),2,:);bm=reshape(permutedims(b,perm),2,:)
    f=svd(bm*am')
    return f.U*f.Vt
end
function frames(a,b,left,right)
    eye=Matrix{Float64}(I,2,2);a=a/norm(a);b=b/norm(b)
    if left && !right;return orthofit(a,b,2),eye;end
    if right && !left;return eye,orthofit(a,b,3);end
    if norm(a-b)<1e-12;return eye,eye;end
    best=Inf;qlbest=eye;qrbest=eye
    # Deterministic initial frames cover reflections and index swaps; only
    # internal virtual bases are fitted, never physical states or energies.
    starts=Matrix{Float64}[Matrix(Diagonal([s1,s2]))*perm for s1 in (1.0,-1.0),s2 in (1.0,-1.0),perm in (eye,[0.0 1;1 0])][:]
    for qr0 in starts
        qr=qr0;ql=eye
        for _ in 1:12
            ql=orthofit(K.mode_product(a,qr,3),b,2)
            qr=orthofit(K.mode_product(a,ql,2),b,3)
        end
        err=norm(K.mode_product(K.mode_product(a,ql,2),qr,3)-b)
        if err<best;best=err;qlbest=ql;qrbest=qr;end
    end
    return qlbest,qrbest
end
function cluster(work,kind)
    kind==:star && return first(K.star_cluster(work,1,1))
    return K.ring_tensor(first(K.ring_cluster(work,1,1)))
end

"Resolve a two-sided P frame ambiguity using the adjacent V tensors."
function resolve_frame_ambiguities!(work,st,nodes,internal)
    refs=Dict{K.Node,Array{Float64,4}}()
    function weighted_vertex(n)
        a=work.tensors[n]
        for (axis,b) in enumerate(K.bonds(work,n))
            b in internal || (a=K.axis_scale(a,work.weights[b],axis))
        end
        return a
    end
    for n in nodes
        n[1]==:V || continue
        a=copy(st.V)
        for (axis,b) in enumerate(K.bonds(work,n))
            b in internal || (a=K.axis_scale(a,st.lambda,axis))
        end
        refs[n]=a
    end
    fixes=0
    for _ in 1:8
        changed=false
        for pn in nodes
            pn[1]==:V && continue
            bs=K.bonds(work,pn)
            bs[2] in internal && bs[3] in internal || continue
            p=weighted_p(work.tensors[pn],work.weights[bs[2]],work.weights[bs[3]])
            # A symmetric 2x2 slice fixes its eigenaxes. A reflection in those
            # axes is an ambiguity only when it preserves BOTH physical slices.
            hs=[Symmetric((p[q,:,:]+p[q,:,:]')/2-tr(p[q,:,:])*Matrix{Float64}(I,2,2)/2) for q in 1:2]
            h=hs[argmax(norm.(hs))]
            norm(h)>1e-12*norm(p) || continue
            f=eigen(h);g=f.vectors*Diagonal([1.0,-1.0])*f.vectors'
            reflected=K.mode_product(K.mode_product(p,g,2),g,3)
            norm(reflected-p)<=1e-10*norm(p) || continue
            ln,la,_,_=endpoints(work,bs[2]);rn,ra,_,_=endpoints(work,bs[3])
            lv=weighted_vertex(ln);rv=weighted_vertex(rn)
            before=relative(lv,refs[ln])^2+relative(rv,refs[rn])^2
            after=relative(K.mode_product(lv,g,la),refs[ln])^2+
                  relative(K.mode_product(rv,g,ra),refs[rn])^2
            if after<before-1e-12
                gauge_bond!(work,bs[2],g,work.weights[bs[2]])
                gauge_bond!(work,bs[3],g,work.weights[bs[3]])
                fixes+=1;changed=true
            end
        end
        changed || break
    end
    return fixes
end

"Align old local SU factors before the explicit spatial projection."
function restore_shared!(st,work,kind)
    nodes=kind==:star ? [(:V,1,1),(:N,1,1),(:E,1,1),(:N,2,1),(:E,1,0)] : K.ring_nodes(1,1)
    nodes=K.wrap.(Ref(work),nodes)
    changed=Set(nodes)
    internal=K.Bond[];external=Tuple{K.Bond,K.Node}[]
    for b in sort(collect(keys(work.weights)))
        vn,_,pn,_=endpoints(work,b)
        if vn in changed && pn in changed
            push!(internal,b)
        elseif vn in changed
            push!(external,(b,vn))
        elseif pn in changed
            push!(external,(b,pn))
        end
    end
    lnew=normalize(mean(work.weights[b] for b in internal))
    lnew=normalize(max.(lnew,1e-12))
    before=cluster(work,kind)
    reference=weighted_p(st.P,st.lambda,st.lambda)
    for pn in nodes
        pn[1]==:V && continue
        bs=K.bonds(work,pn);left=bs[2] in internal;right=bs[3] in internal
        a=weighted_p(work.tensors[pn],work.weights[bs[2]],work.weights[bs[3]])
        ql,qr=frames(a,reference,left,right)
        left && gauge_bond!(work,bs[2],ql,lnew)
        right && gauge_bond!(work,bs[3],qr,lnew)
    end
    for (b,n) in external;reweight_external!(work,b,lnew,n);end
    frame_ambiguities_resolved=resolve_frame_ambiguities!(work,st,nodes,internal)
    aligned=cluster(work,kind)
    gauge_error=relative(before,aligned)
    gauge_error<1e-8 || error("virtual alignment changed the cluster: $gauge_error")
    unit_phase(a,ref)=dot(a,ref)>=0 ? a/norm(a) : -a/norm(a)
    vs=[unit_phase(work.tensors[n],st.V) for n in nodes if n[1]==:V]
    ps=[unit_phase(work.tensors[n],st.P) for n in nodes if n[1]!=:V]
    v=mean(vs);p=mean(ps)
    v_projection=norm(rotation_project(v)-v)/max(norm(v),eps())
    p_projection=norm(p-permutedims(p,(1,3,2)))/(2max(norm(p),eps()))
    commit!(st,v,p,lnew)
    return (;gauge_error,v_projection,p_projection,frame_ambiguities_resolved,symmetry_defect=defect(st))
end
function star_step!(st,dt)
    work=workspace(st)
    discarded=K.star_update!(work,1,1,dt)
    return (;discarded,restore_shared!(st,work,:star)...)
end
function plaquette_step!(st,dt;als_sweeps=12)
    als_sweeps>0 || throw(ArgumentError("ALS sweep count must be positive"))
    work=workspace(st)
    fit=K.loop_update!(work,1,1,dt;als_sweeps)
    return (;fit,restore_shared!(st,work,:plaquette)...)
end
function sweep!(st,dt;als_sweeps=12)
    # Only commit a complete sweep; a failing second substep cannot leave an
    # advanced star tensor labelled with the preceding checkpoint step.
    candidate=State(copy(st.V),copy(st.P),copy(st.lambda))
    a=star_step!(candidate,dt)
    b=plaquette_step!(candidate,dt;als_sweeps)
    st.V=candidate.V;st.P=candidate.P;st.lambda=candidate.lambda
    return (;star=a,plaquette=b)
end
function local_residuals(st)
    r=K.local_stabilizer_residuals(workspace(st))
    return (;star=maximum(r[1:2:end]),plaquette=maximum(r[2:2:end]))
end
end

module BoundaryAudit
using LinearAlgebra, Random, TensorKit, MPSKit, KrylovKit, Printf
using ..FieldLoop
export prepare_layers, solve_boundaries, measure_boundary, column_apply,
       column_adjoint, layer, mpo_rows, strip_environment

"Double layer Q[n,e,s,w], with fused ket/bra virtual indices."
function layer(a,op=Matrix{Float64}(I,4,4))
    @tensor t[n,N,e,E,s,S,w,W] := a[p,n,e,s,w]*op[p,q]*conj(a[q,N,E,S,W])
    D=size(a,2)
    return ComplexF64.(reshape(t,D^2,D^2,D^2,D^2))
end
function prepare_layers(st;balanced=false)
    aa=[blocked_tensor(st,r,c) for r in 1:st.L,c in 1:st.L]
    logs=Float64[]
    for r in 1:st.L,c in 1:st.L
        a=aa[r,c]
        if balanced
            # Pure bond gauge: distribute the outgoing boundary λ equally.
            # It changes no physical state; zero-support entries remain zero.
            bn=FieldLoop.bonds(st,(:N,r,c));be=FieldLoop.bonds(st,(:E,r,c))
            bv=FieldLoop.bonds(st,(:V,r,c))
            a=FieldLoop.axis_scale(a,FieldLoop.pinverse(sqrt.(st.weights[bn[3]])),2)
            a=FieldLoop.axis_scale(a,FieldLoop.pinverse(sqrt.(st.weights[be[3]])),3)
            a=FieldLoop.axis_scale(a,sqrt.(st.weights[bv[3]]),4)
            a=FieldLoop.axis_scale(a,sqrt.(st.weights[bv[4]]),5)
        end
        push!(logs,2log(norm(a)))
        aa[r,c]=a/norm(a)
    end
    return aa,map(layer,aa),logs
end
function mpo_rows(qs;upward=false)
    q=size(qs[1],1);v=ComplexSpace(q)
    rows=map(1:2) do r
        rc=upward ? 3-r : r
        tensors=map(1:2) do c
            # MPO[left,physical_out;physical_in,right]
            data=upward ? permutedims(conj(qs[rc,c]),(4,1,3,2)) : permutedims(qs[rc,c],(4,3,1,2))
            TensorMap(data,v⊗v,v⊗v)
        end
        InfiniteMPO(tensors)
    end
    return MPSKit.Multiline(rows)
end
function expand_boundary(mps,chi)
    return MPSKit.MultilineMPS(map(1:2) do r
        tensors=map(1:2) do c
            old=convert(Array,mps[r].AL[c]);q=size(old,2)
            size(old,1)<=chi && size(old,3)<=chi || error("Cannot shrink boundary using padding")
            a=1e-5*randn(ComplexF64,chi,q,chi)/sqrt(chi*q)
            a[axes(old)...] .= old
            TensorMap(a,ComplexSpace(chi)⊗ComplexSpace(q),ComplexSpace(chi))
        end
        InfiniteMPS(tensors)
    end)
end
function solve_boundaries(qs;chi=8,seed=1,tol=1e-8,maxiter=100,maxseconds=240,initial=nothing)
    q=size(qs[1],1);p=ComplexSpace(q);v=ComplexSpace(chi)
    states=[];infos=[];start=time()
    for (label,upward) in (("down",false),("up",true))
        Random.seed!(seed)
        mps=isnothing(initial) ? MPSKit.MultilineMPS([InfiniteMPS(randn,ComplexF64,fill(p,2),fill(v,2)) for _ in 1:2]) : deepcopy(initial[upward ? 2 : 1])
        dim(domain(mps[1].AL[1],1))==chi || (mps=expand_boundary(mps,chi))
        @assert dim(domain(mps[1].AL[1],1))==chi && dim(codomain(mps[1].AL[1],2))==q "Boundary dimensions do not match requested χ and D"
        op=mpo_rows(qs;upward)
        history=NamedTuple[];lastiter=Ref(0)
        function progress(iter,psi,h,env)
            lastiter[]=iter+1
            if iter%10==0
                err=Float64(MPSKit.calc_galerkin(psi,h,psi,env))
                normcell=abs(prod(MPSKit.expectation_value(psi,h,env)))
                push!(history,(;iteration=iter+1,error=err,normcell,seconds=time()-start))
                @printf("[VUMPS %s χ=%d seed=%d] iter=%d projected_error=%.3e norm/cell=%.5e wall=%.1fs\n",label,chi,seed,iter+1,err,normcell,time()-start);flush(stdout)
            end
            time()-start>maxseconds && error("VUMPS local wall budget reached")
            return psi,env
        end
        alg=VUMPS(;tol,maxiter,verbosity=1,alg_eigsolve=MPSKit.Defaults.alg_eigsolve(;ishermitian=false),finalize=progress)
        mps,env,err=leading_boundary(mps,op,alg)
        normcell=abs(prod(MPSKit.expectation_value(mps,op,env)))
        push!(states,mps)
        push!(infos,(;direction=label,projected_error=Float64(err),converged=err<tol,iterations=lastiter[],normcell,history))
        println("[VUMPS $label] final_error=$err normcell=$normcell");flush(stdout)
    end
    return states[1],states[2],infos
end

"Transfer one horizontal column of a two-row strip from left to right."
function column_apply(x,a,b,u,v)
    @tensor t[n,B,w,z,c] := x[A,w,z,c]*a[A,n,B]
    @tensor t2[B,e,s,z,c] := t[n,B,w,z,c]*u[n,e,s,w]
    @tensor t3[B,e,f,p,c] := t2[B,e,s,z,c]*v[s,f,p,z]
    @tensor y[B,e,f,d] := t3[B,e,f,p,c]*conj(b[c,p,d])
    return y
end
function column_adjoint(y,a,b,u,v)
    return column_apply(y,permutedims(conj(a),(3,2,1)),permutedims(conj(b),(3,2,1)),
                        permutedims(conj(u),(1,4,3,2)),permutedims(conj(v),(1,4,3,2)))
end

function strip_environment(down,up,qs,r;seed=11)
    # Boundary r is north of physical row r; two rows return to boundary r.
    aa=[convert(Array,down[r].AL[c]) for c in 1:2]
    bb=[convert(Array,up[r].AL[c]) for c in 1:2]
    f(c,x)=column_apply(x,aa[c],bb[c],qs[r,c],qs[3-r,c])
    ft(c,x)=column_adjoint(x,aa[c],bb[c],qs[r,c],qs[3-r,c])
    Random.seed!(seed)
    shape=(size(aa[1],1),size(qs[1],1),size(qs[1],1),size(bb[1],1))
    x=randn(ComplexF64,shape...);normalize!(x)
    scale=norm(f(2,f(1,x)))
    vals,vecs,info=eigsolve(z->f(2,f(1,z))/scale,x,1,:LM;tol=1e-11,krylovdim=24,maxiter=100)
    valsl,vecsl,infol=eigsolve(z->ft(1,ft(2,z))/scale,x,1,:LM;tol=1e-11,krylovdim=24,maxiter=100)
    rr=vecs[1];ll=vecsl[1]
    resr=norm(f(2,f(1,rr))/scale-vals[1]*rr)/abs(vals[1])
    resl=norm(ft(1,ft(2,ll))/scale-valsl[1]*ll)/abs(valsl[1])
    rights=[rr,normalize(f(1,rr))]
    lefts=[ll,normalize(ft(2,ll))]
    denominators=[dot(lefts[c],f(3-c,f(c,rights[c]))) for c in 1:2]
    return (;aa,bb,rights,lefts,denominators,right_residual=resr,left_residual=resl,
             eigenvalue_abs=abs(vals[1])*scale,eigenvalue_mismatch=abs(vals[1]-conj(valsl[1]))/abs(vals[1]),
             converged=info.converged>=1 && infol.converged>=1)
end

function measure_boundary(st,aa,qs,down,up,hx,hz)
    strips=[strip_environment(down,up,qs,r) for r in 1:2]
    x,z,i2=[0.0 1;1 0],[1.0 0;0 -1],Matrix{Float64}(I,2,2)
    xe,xn,ze,zn=kron(i2,x),kron(x,i2),kron(i2,z),kron(z,i2)
    values=Dict{String,Float64}();imagmax=0.0
    function expectation(r,start,topops,botops)
        env=strips[r];v=env.rights[mod1(start,2)]
        for col in start:start+1
            c=mod1(col,2)
            t=haskey(topops,col) ? layer(aa[r,c],topops[col]) : qs[r,c]
            b=haskey(botops,col) ? layer(aa[3-r,c],botops[col]) : qs[3-r,c]
            v=column_apply(v,env.aa[c],env.bb[c],t,b)
        end
        result=dot(env.lefts[mod1(start,2)],v)/env.denominators[mod1(start,2)]
        imagmax=max(imagmax,abs(imag(result)))
        return real(result)
    end
    for r in 1:2,c in 1:2
        values["A$r$c"]=expectation(r,c-1,Dict(c-1=>xe,c=>xe*xn),Dict(c=>xn))
        values["B$r$c"]=expectation(3-r,c,Dict(c=>ze),Dict(c=>ze*zn,c+1=>zn))
        for (name,op) in (("XE",xe),("XN",xn),("ZE",ze),("ZN",zn))
            values["$name$r$c"]=expectation(r,c,Dict(c=>op),Dict())
        end
    end
    av=sum(values["A$r$c"] for r in 1:2,c in 1:2)/4
    bv=sum(values["B$r$c"] for r in 1:2,c in 1:2)/4
    mx=sum(values["$k$r$c"] for k in ("XE","XN"),r in 1:2,c in 1:2)/8
    mz=sum(values["$k$r$c"] for k in ("ZE","ZN"),r in 1:2,c in 1:2)/8
    info=[(;s.right_residual,s.left_residual,s.eigenvalue_abs,s.eigenvalue_mismatch,s.converged) for s in strips]
    return (;energy=-(av+bv)/2-hx*mx-hz*mz,mx,mz,astar=av,bplaq=bv,imagmax,
             physical_excess=max(0.0,maximum(abs,Base.values(values))-1),observables=values,strip_info=info)
end
end

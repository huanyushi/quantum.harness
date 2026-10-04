# Independent exact transverse contraction on a circumference-4 cylinder.
# Infinite length, finite circumference; a frozen-state diagnostic, not an
# independent infinite-plane ground-state solution or a new update algorithm.
using LinearAlgebra, Serialization, JSON, TensorKit, Printf, KrylovKit
BLAS.set_num_threads(1)
include("core.jl")
using .FieldLoop

function double_layer(a,op)
    @tensor t[n,N,e,E,s,S,w,W] := a[p,n,e,s,w]*op[p,q]*a[q,N,E,S,W]
    D=size(a,2)
    return reshape(t,D^2,D^2,D^2,D^2)
end
function row_apply(ts,v)
    q=size(ts[1],1);v=reshape(v,q,q,q,q)
    a,b,c,d=ts
    @tensor u[n1,e1,e4,s2,s3,s4] := a[n1,e1,s1,e4]*v[s1,s2,s3,s4]
    @tensor w[n1,n2,e2,e4,s3,s4] := u[n1,e1,e4,s2,s3,s4]*b[n2,e2,s2,e1]
    @tensor z[n1,n2,n3,e3,e4,s4] := w[n1,n2,e2,e4,s3,s4]*c[n3,e3,s3,e2]
    @tensor y[n1,n2,n3,n4] := z[n1,n2,n3,e3,e4,s4]*d[n4,e4,s4,e3]
    return vec(y)
end
function dominant(f,v;maxiter=12,tol=1e-11)
    v=v/norm(v);err=Inf;lam=0.0;iters=0
    for i in 1:maxiter
        w=f(v);lam=dot(v,w);w/=norm(w)
        err=norm(w-v);v=w;iters=i
        err<tol && break
    end
    if err>=tol
        # Near-topological sector eigenvalues can differ by only O(h^W);
        # power iteration is then slow. Scale out arbitrary tensor norms.
        scale=norm(f(v))
        evals,vecs,info=eigsolve(x->f(x)/scale,v,1,:LR;tol=1e-12,krylovdim=24,maxiter=100)
        @assert norm(imag.(vecs[1]))<1e-8
        v=normalize(real.(vecs[1]));lam=real(evals[1])*scale
        err=norm(f(v)/scale-real(evals[1])*v)
        iters+=info.numops
    end
    return v,(;residual=err,iterations=iters,converged=err<tol,eigenvalue=lam)
end
function cylinder_measure(st,hx,hz)
    t=time();D=st.D;q=D^2
    a=[blocked_tensor(st,r,c) for r in 1:2,c in 1:2];foreach(normalize!,a)
    x,z,i2=[0.0 1;1 0],[1.0 0;0 -1],Matrix{Float64}(I,2,2)
    i4=Matrix{Float64}(I,4,4);xe,xn,ze,zn=kron(i2,x),kron(x,i2),kron(i2,z),kron(z,i2)
    base=[[double_layer(a[r,mod1(c,2)],i4) for c in 1:4] for r in 1:2]
    trans=[[permutedims(t,(3,2,1,4)) for t in row] for row in base]
    boundary=vec(Matrix{Float64}(I,D,D));v=kron(boundary,boundary,boundary,boundary)
    right,ri=dominant(v->row_apply(base[1],row_apply(base[2],v)),v)
    left,li=dominant(v->row_apply(trans[2],row_apply(trans[1],v)),v)
    rights=[right,normalize(row_apply(base[2],right))]
    lefts=[left,normalize(row_apply(trans[1],left))]
    denoms=[dot(lefts[r],row_apply(base[r],row_apply(base[3-r],rights[r]))) for r in 1:2]
    function expectation(r,ops1,ops2)
        row1=copy(base[r]);row2=copy(base[3-r])
        for (c,op) in ops1;row1[mod1(c,4)]=double_layer(a[r,mod1(c,2)],op);end
        for (c,op) in ops2;row2[mod1(c,4)]=double_layer(a[3-r,mod1(c,2)],op);end
        return dot(lefts[r],row_apply(row1,row_apply(row2,rights[r])))/denoms[r]
    end
    vals=Dict{String,Float64}()
    for r in 1:2,c in 1:2
        vals["A$r$c"]=expectation(r,[(c,xe*xn),(c-1,xe)],[(c,xn)])
        vals["B$r$c"]=expectation(3-r,[(c,ze)],[(c,ze*zn),(c+1,zn)])
        for (name,op) in (("XE",xe),("XN",xn),("ZE",ze),("ZN",zn))
            vals["$name$r$c"]=expectation(r,[(c,op)],[])
        end
    end
    av=sum(vals["A$r$c"] for r in 1:2,c in 1:2)/4
    bv=sum(vals["B$r$c"] for r in 1:2,c in 1:2)/4
    mx=sum(vals["$(k)$r$c"] for k in ("XE","XN"),r in 1:2,c in 1:2)/8
    mz=sum(vals["$(k)$r$c"] for k in ("ZE","ZN"),r in 1:2,c in 1:2)/8
    return (;circumference=4,energy=-(av+bv)/2-hx*mx-hz*mz,mx,mz,astar=av,bplaq=bv,observables=vals,right=ri,left=li,seconds=time()-t)
end
function main()
    root=joinpath(@__DIR__,"results")
    # Exact control calibrates the independent contraction and operator incidence.
    exact=cylinder_measure(initial_state(:exact,1),0.0,0.0)
    @assert abs(exact.energy+1)<1e-10 && abs(exact.mx)<1e-10 && abs(exact.mz)<1e-10
    println("Cylinder exact control: ",(;exact.energy,exact.mx,exact.mz,exact.seconds));flush(stdout)
    for (name,hx,hz) in (("d2-xz",0.1,0.1),("d3-xz",0.1,0.1),("d3-x",0.1,0.0))
        st=deserialize(joinpath(root,name,"state.jls")).state
        m=cylinder_measure(st,hx,hz)
        open(joinpath(root,name,"cylinder4.json"),"w") do io;JSON.print(io,m);end
        println("Cylinder $name ",(;m.energy,m.mx,m.mz,m.right,m.left,m.seconds));flush(stdout)
    end
end
if abspath(PROGRAM_FILE)==@__FILE__;main();end

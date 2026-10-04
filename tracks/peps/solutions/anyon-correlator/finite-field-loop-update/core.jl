module FieldLoop
using LinearAlgebra, Random
export initial_state, field_gate, field_step!, ring_cluster, ring_tensor,
       restore_ring!, loop_update!, sweep!, blocked_tensor, reference_values,
       grow_state, state_weights

const Node=Tuple{Symbol,Int,Int}
const Bond=NTuple{3,Int}
const PINV_RTOL=1e-14
mutable struct SplitState
    tensors::Dict{Node,Array{Float64}}
    weights::Dict{Bond,Vector{Float64}}
    L::Int
    D::Int
end
wrap(st,n::Node)=(n[1],mod1(n[2],st.L),mod1(n[3],st.L))
bond(st,r,c,d)=(mod1(r,st.L),mod1(c,st.L),d)
tensor(st,n)=st.tensors[wrap(st,n)]
function bonds(st,n::Node)
    k,r,c=n
    k==:V && return [bond(st,r,c,d) for d in 1:4]
    k==:E && return [nothing,bond(st,r,c,2),bond(st,r,c+1,4)]
    k==:N && return [nothing,bond(st,r,c,1),bond(st,r-1,c,3)]
    error("Unknown node $n")
end
function initial_state(kind,seed;D=2,L=2)
    rng=MersenneTwister(seed)
    ts=Dict{Node,Array{Float64}}(); ws=Dict{Bond,Vector{Float64}}()
    for r in 1:L,c in 1:L
        for (k,shape) in ((:V,(D,D,D,D)),(:E,(2,D,D)),(:N,(2,D,D)))
            a=kind==:random ? randn(rng,shape...) : zeros(shape...)
            if kind==:exact
                @assert D>=2
                if k==:V
                    for n in 0:1,e in 0:1,s in 0:1,w in 0:1
                        a[n+1,e+1,s+1,w+1]=iseven(n+e+s+w)
                    end
                else
                    for p in 0:1,b in 0:1
                        a[p+1,b+1,b+1]=(-1.0)^(p*b)/sqrt(2)
                    end
                end
            elseif kind != :random
                error("initialization: random or exact")
            end
            ts[(k,r,c)]=a/norm(a)
        end
        for d in 1:4; ws[(r,c,d)]=ones(D); end
    end
    return SplitState(ts,ws,L,D)
end
function grow_state(st,D;seed=71,noise=1e-3)
    @assert D>=st.D
    D==st.D && return deepcopy(st)
    ns=initial_state(:random,seed;D,L=st.L)
    for (k,a) in st.tensors
        ns.tensors[k] .*= noise
        ns.tensors[k][axes(a)...] .= a
        normalize!(ns.tensors[k])
    end
    for (k,w) in st.weights
        ns.weights[k] .= noise
        ns.weights[k][1:st.D] .= w
        normalize!(ns.weights[k])
    end
    return ns
end
axis_scale(a,w,ax)=a.*reshape(w,ntuple(i->i==ax ? length(w) : 1,ndims(a)))
pinverse(w)=[v>PINV_RTOL*maximum(w) ? inv(v) : 0.0 for v in w]
function mode_product(a,m,ax)
    perm=[ax;[i for i in 1:ndims(a) if i!=ax]]
    dims=size(a)[perm]
    b=m*reshape(permutedims(a,perm),dims[1],:)
    return permutedims(reshape(b,size(m,1),dims[2:end]...),invperm(perm))
end
function pauli_evolve(a,axes,kind,dt)
    b=copy(a)
    for ax in axes
        n=size(a,ax); @assert iseven(n)
        op=zeros(n,n)
        for j in 1:n
            if kind==:X; op[isodd(j) ? j+1 : j-1,j]=1
            else; op[j,j]=isodd(j) ? 1 : -1; end
        end
        b=mode_product(b,op,ax)
    end
    return cosh(dt)*a+sinh(dt)*b
end
function field_gate(hx,hz,dt)
    h=hypot(hx,hz)
    h==0 && return Matrix{Float64}(I,2,2)
    return cosh(dt*h)*Matrix{Float64}(I,2,2)+(sinh(dt*h)/h)*[hz hx;hx -hz]
end
function field_step!(st,hx,hz,dt)
    g=field_gate(hx,hz,dt)
    for (n,p) in st.tensors
        n[1]==:V && continue
        a=mode_product(p,g,1)
        st.tensors[n]=a/norm(a)
    end
    return st
end
function star_update!(st,r,c,dt)
    D=st.D
    leaves=[(:N,r,c),(:E,r,c),(:N,r+1,c),(:E,r,c-1)]
    vb=bonds(st,(:V,r,c)); x=copy(tensor(st,(:V,r,c)))
    meta=NamedTuple[]
    for i in 1:4
        p=tensor(st,leaves[i]); pb=bonds(st,leaves[i])
        inner=findfirst(==(vb[i]),pb); outer=inner==2 ? 3 : 2
        a=axis_scale(p,st.weights[pb[outer]],outer)
        a=axis_scale(a,st.weights[vb[i]],inner)
        x=mode_product(x,reshape(permutedims(a,[1,outer,inner]),2D,D),i)
        push!(meta,(;inner,outer,pb))
    end
    y=pauli_evolve(x,1:4,:X,dt)
    us=Matrix{Float64}[]; ls=Vector{Float64}[]; discarded=0.0
    for i in 1:4
        perm=[i;[j for j in 1:4 if j!=i]]
        f=svd(reshape(permutedims(y,perm),2D,:);full=false)
        discarded=max(discarded,sum(abs2,f.S[D+1:end])/sum(abs2,f.S))
        push!(us,f.U[:,1:D]);push!(ls,f.S[1:D]/norm(f.S[1:D]))
    end
    v=copy(y)
    for i in 1:4
        v=axis_scale(mode_product(v,us[i]',i),pinverse(ls[i]),i)
        m=meta[i]
        p=permutedims(reshape(us[i],2,D,D),invperm([1,m.outer,m.inner]))
        p=axis_scale(p,pinverse(st.weights[m.pb[m.outer]]),m.outer)
        st.tensors[wrap(st,leaves[i])]=p/norm(p)
        st.weights[vb[i]]=ls[i]
    end
    st.tensors[wrap(st,(:V,r,c))]=v/norm(v)
    return discarded
end
state_weights(st)=reduce(vcat,[st.weights[b] for b in sort(collect(keys(st.weights)))])
function blocked_tensor(st,r,c)
    D=st.D
    v=tensor(st,(:V,r,c));pe=tensor(st,(:E,r,c));pn=tensor(st,(:N,r,c))
    be=bonds(st,(:E,r,c));bn=bonds(st,(:N,r,c));out=zeros(2,2,D,D,D,D)
    for p in 1:2,q in 1:2,n in 1:D,e in 1:D,s in 1:D,w in 1:D,a in 1:D,b in 1:D
        out[p,q,n,e,s,w]+=v[b,a,s,w]*pe[p,a,e]*pn[q,b,n]*st.weights[be[2]][a]*st.weights[be[3]][e]*st.weights[bn[2]][b]*st.weights[bn[3]][n]
    end
    return reshape(out,4,D,D,D,D)
end
include("loop.jl")
include("reference.jl")

function sweep!(st,hx,hz,dt;als_sweeps=12)
    field_step!(st,hx,hz,dt/2)
    star_error=0.0;fit_error=0.0;gauge_error=0.0;bp_residual=0.0
    als_iterations=0;bp_iterations=0;als_capped=0;bp_capped=0;fit_increase=0.0
    for r in 1:st.L,c in 1:st.L
        star_error=max(star_error,star_update!(st,r,c,dt))
        m=loop_update!(st,r,c,dt;als_sweeps)
        fit_error=max(fit_error,m.fit_after);gauge_error=max(gauge_error,m.gauge_error)
        bp_residual=max(bp_residual,m.bp_residual)
        als_iterations=max(als_iterations,m.als_iterations);bp_iterations=max(bp_iterations,m.bp_iterations)
        als_capped+=!m.als_converged;bp_capped+=!m.bp_converged
        fit_increase=max(fit_increase,m.fit_after-m.fit_before)
    end
    field_step!(st,hx,hz,dt/2)
    return (;star_error,fit_error,gauge_error,bp_residual,als_iterations,bp_iterations,als_capped,bp_capped,fit_increase)
end
end

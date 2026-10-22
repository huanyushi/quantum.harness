using TensorKit, PEPSKit, Random, LinearAlgebra, Statistics

function measurement_peps(st)
    p,v=ComplexSpace(4),ComplexSpace(st.D)
    ts=[TensorMap(ComplexF64.(reshape(blocked_tensor(st,r,c),4,st.D^4)),p,v⊗v⊗v'⊗v') for r in 1:st.L,c in 1:st.L]
    foreach(normalize!,ts)
    return InfinitePEPS(ts)
end

function observable_operators(L)
    p=ComplexSpace(4);lat=fill(p,L,L)
    x,z,eye=[0.0 1;1 0],[1.0 0;0 -1],Matrix{Float64}(I,2,2)
    xe,xn,ze,zn=kron(eye,x),kron(x,eye),kron(eye,z),kron(z,eye)
    a=TensorMap(ComplexF64.(kron(xn,xe*xn,xe)),p^3,p^3)
    b=TensorMap(ComplexF64.(kron(zn,ze*zn,ze)),p^3,p^3)
    ops=[];names=String[]
    function add(sites,op,name)
        h=LocalOperator{Any,eltype(lat)}(lat)
        PEPSKit.add_term!(h,sites,op)
        push!(ops,h);push!(names,name)
    end
    for r in 1:L,c in 1:L
        add([CartesianIndex(r,c-1),CartesianIndex(r,c),CartesianIndex(r+1,c)],a,"A$r$c")
        add([CartesianIndex(r-1,c),CartesianIndex(r,c),CartesianIndex(r,c+1)],b,"B$r$c")
        for (name,mat) in (("XE",xe),("XN",xn),("ZE",ze),("ZN",zn))
            add([CartesianIndex(r,c)],TensorMap(ComplexF64.(mat),p,p),"$name$r$c")
        end
    end
    return ops,names
end

function measure_state(st,hx,hz;chi=8,tol=1e-9,maxiter=250,envseed=2026,alg=:SimultaneousCTMRG,prepared=nothing)
    psi=isnothing(prepared) ? measurement_peps(st) : prepared
    Random.seed!(max(envseed,0));t=time()
    env0=CTMRGEnv(randn,ComplexF64,psi,ComplexSpace(chi))
    if envseed == -1
        # A positive factorized norm boundary, invariant under real orthogonal
        # virtual-basis changes. This only changes contraction initialization.
        for i in eachindex(env0.corners)
            a=env0.corners[i];n=dim(codomain(a));data=Matrix{ComplexF64}(I,n,n)
            env0.corners[i]=TensorMap(data,codomain(a),domain(a))
        end
        for i in eachindex(env0.edges)
            a=env0.edges[i]
            data=zeros(ComplexF64,dim(codomain(a,1)),dim(codomain(a,2)),dim(codomain(a,3)),dim(domain(a,1)))
            for c in axes(data,1),v in axes(data,2);data[c,v,v,c]=1;end
            env0.edges[i]=TensorMap(data,codomain(a),domain(a))
        end
    end
    env,info=leading_boundary(env0,psi;tol,maxiter,alg,verbosity=0)
    ops,names=observable_operators(st.L)
    vals=[real(expectation_value(psi,o,env)) for o in ops]
    As=vals[1:6:end];Bs=vals[2:6:end]
    xs=reduce(vcat,[vals[i:i+1] for i in 3:6:length(vals)])
    zs=reduce(vcat,[vals[i:i+1] for i in 5:6:length(vals)])
    mx,mz=mean(xs),mean(zs);astar,bplaq=mean(As),mean(Bs)
    energy=-(astar+bplaq)/2-hx*mx-hz*mz
    sector_error=hx==0 ? maximum(abs.(Bs.-1)) : hz==0 ? maximum(abs.(As.-1)) : nothing
    return (;energy,mx,mz,astar,bplaq,sector_error,site_spread=maximum((maximum(v)-minimum(v) for v in (As,Bs,xs,zs))),
            physical_excess=max(0.0,maximum(abs.(vals))-1),ctm_converged=info.converged,
            ctm_residual=Float64(info.convergence_error),ctm_seconds=time()-t,chi,envseed,ctm_algorithm=String(alg),
            observables=Dict(zip(names,vals)))
end

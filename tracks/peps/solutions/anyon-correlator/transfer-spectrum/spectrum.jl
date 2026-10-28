# Compatibility boundary: MPSKit 0.13.12, quasiparticleexcitation.jl lines
# 193-265 (statmech :LM and the public real-momentum Multiline wrapper),
# 282-294 and 374-389 (the SAME normalized effective action for residuals).
# Never call the InfiniteMPO/InfiniteQP excitations overload: it contains a
# self-reference at line 217 in this installed version.
struct SectorPoint
    g::Int
    q::Int
    k::Float64
    branch::Int
    ratio_real::Union{Nothing,Float64}
    ratio_imag::Union{Nothing,Float64}
    rho::Union{Nothing,Float64}
    phase::Union{Nothing,Float64}
    quasi_energy::Union{Nothing,Float64}
    projected_residual::Union{Nothing,Float64}
    variational_dim::Int
    status::String
    state
end
function mode_status(r,res,dim;zero_proven=false,budget=false,tol=1e-10,resolution=1e-12)
    budget && return "budget"
    dim==0 && return "empty_sector"
    (!isfinite(r) || !isfinite(res) || res>tol) && return "not_converged"
    zero_proven && return "zero_mode"
    abs(r)<=resolution && return "below_resolution"
    return "converged"
end
function qp_basis(proto)
    basis=typeof(proto)[]
    for (i,x) in enumerate(proto.Xs),(c,b) in blocks(x),j in eachindex(b)
        v=zero(proto); block(v.Xs[i],c)[j]=1
        push!(basis,v)
    end
    basis
end
function effective_action(op,l,r,proto;tol=1e-10)
    le=environments(l,op); re=l===r ? le : environments(r,op)
    E=MPSKit.effective_excitation_renormalization_energy(op,proto,le,re)
    eff=MPSKit.EffectiveExcitationHamiltonian(op,le,re,E)
    action=v->Base.invokelatest(eff,v;tol,maxiter=100,verbosity=0)
    return action,le,re
end
# Runtime boundaries keep Julia 1.12 from recursively inferring the whole
# eigensolver/environment graph into callers (a 600s top-level inference
# timeout was reproduced). Numerical algorithms and tolerance are unchanged.
solve_modes(op,l,r,q,k;num=1,options=(;))=Base.invokelatest(_solve_modes,op,l,r,q,k;num,options)
function _solve_modes(op,l,r,q,k;num=1,options=(;))
    pkgversion(MPSKit)==v"0.13.12" || error("excitation adapter requires revalidation for this MPSKit version")
    tol=get(options,:tol,1e-10); resolution=get(options,:resolution,1e-12)
    reference=get(options,:reference,false)
    proto=Base.invokelatest(MPSKit.LeftGaugedQP,randn,l,r;sector=Z2Irrep(q),momentum=Float64(k))
    dimvar=sum(dim,proto.Xs)
    dimvar==0 && return (;ratios=ComplexF64[],residuals=Float64[],states=Any[],variational_dim=0,zero_proven=false)
    num>0 || throw(ArgumentError("num must be positive")); count=min(num,dimvar)
    action,le,re=Base.invokelatest(effective_action,op,l,r,proto;tol)
    basis=dimvar<=8 ? qp_basis(proto) : nothing
    projected=basis===nothing ? nothing : ComplexF64[dot(a,action(b)) for a in basis,b in basis]
    small_action=projected!==nothing && norm(projected)<=resolution
    if small_action
        # Complete basis of the allowed QP parameters, NOT a finite cylinder.
        # This avoids Arnoldi breakdown for a vanishing effective operator.
        f=eigen(projected); order=sortperm(abs.(f.values);rev=true)[1:count]
        vals=ComplexF64.(f.values[order]); states=typeof(proto)[]
        for j in order
            v=zero(proto)
            for i in eachindex(basis); axpy!(f.vectors[i,j],basis[i],v); end
            normalize!(v);push!(states,v)
        end
    else
        env_alg=MPSKit.Defaults.alg_environments(;tol,dynamic_tols=false)
        alg=QuasiparticleAnsatz(;tol,ishermitian=false,krylovdim=max(8,min(dimvar+2,32)),maxiter=100,alg_environments=env_alg)
        # Explicit one-row Multiline is the other public statmech route.
        # Construct its QP from the already calibrated sector/momentum rather
        # than asking the generic real-momentum wrapper to infer that graph.
        mlop=MPSKit.Multiline([op]); mlphi=MPSKit.Multiline([proto])
        mlenv=MPSKit.Multiline([le]); mrenv=l===r ? mlenv : MPSKit.Multiline([re])
        vals,multistates=Base.invokelatest(excitations,mlop,alg,mlphi,mlenv,mrenv;num=count)
        vals=ComplexF64.(vals);states=[first(parent(v)) for v in multistates]
        # KrylovKit may return ALL converged Ritz pairs, more than requested.
        # Keep the requested largest-modulus pairs and their matching states.
        length(vals)>=count || error("excitation solver returned fewer modes than requested")
        order=sortperm(abs.(vals);rev=true)[1:count]
        vals=vals[order];states=states[order]
        foreach(normalize!,states)
    end
    residuals=[Float64(norm(action(v)-value*v)) for (value,v) in zip(vals,states)]
    return (;ratios=vals,residuals,states,variational_dim=dimvar,zero_proven=reference && small_action)
end
function solve_sector(net,pair,label;momenta,num=1,options=(;))
    string_dictionary().validated || error("string dictionary calibration failed")
    pair.diagnostics.converged || error("unconverged boundary pair")
    bg=sector_backgrounds(pair,label);points=SectorPoint[]
    tol=get(options,:tol,1e-10);resolution=get(options,:resolution,1e-12)
    deadline=get(options,:deadline,Inf)
    for k in momenta
        if time()>=deadline
            push!(points,SectorPoint(label.g,label.q,k,1,nothing,nothing,nothing,nothing,nothing,nothing,0,"budget",nothing));break
        end
        result=solve_modes(net.mpo,bg.left,bg.right,label.q,k;num,options)
        if result.variational_dim==0
            push!(points,SectorPoint(label.g,label.q,k,1,nothing,nothing,nothing,nothing,nothing,nothing,0,"empty_sector",nothing))
        end
        for (branch,(r,res,state)) in enumerate(zip(result.ratios,result.residuals,result.states))
            status=mode_status(r,res,result.variational_dim;zero_proven=result.zero_proven,tol,resolution)
            resolved=status=="converged" && abs(r)>resolution
            push!(points,SectorPoint(label.g,label.q,k,branch,real(r),imag(r),abs(r),resolved ? angle(r) : nothing,
                resolved ? -log(abs(r)) : nothing,res,result.variational_dim,status,state))
        end
    end
    return points
end

include("audit.jl")

function finite_arrays(ts)
    ds=(size(ts[1],2),size(ts[1],3),size(ts[1],4),size(ts[1],5),size(ts[2],2),size(ts[2],4),size(ts[3],3),size(ts[3],5))
    v=zeros(ComplexF64,256)
    for (a,b,c,d,e,f,g,h) in Iterators.product((1:n for n in ds)...)
        for p in 0:255
            p1,p2,p3,p4=[1+((p>>(2k))&3) for k in 0:3]
            v[p+1]+=ts[1][p1,a,b,c,d]*ts[2][p2,e,d,f,b]*ts[3][p3,c,g,a,h]*ts[4][p4,f,h,e,g]
        end
    end
    normalize(v)
end
function join_open(ms)
    a=convert(Array,ms[1]); dims=collect(size(a))[1:end-1]
    acc=reshape(a,:,size(a,ndims(a)))
    for m in ms[2:end]
        b=convert(Array,m)
        acc=reshape(acc*reshape(b,size(b,1),:),:,size(b,ndims(b)))
        append!(dims,collect(size(b))[2:end-1])
    end
    push!(dims,size(acc,2))
    reshape(acc,dims...)
end
function dense_mode(x,op,axis)
    order=[axis;[i for i in 1:ndims(x) if i!=axis]]
    dims=size(x)[order]
    z=op*reshape(permutedims(x,order),dims[1],:)
    permutedims(reshape(z,dims...),invperm(order))
end
function stage_wave(ms,psi,w,sites,invperms,axes,flips)
    ms=deepcopy(ms);PEPSKit._flip_virtuals!(ms,flips)
    ts=[convert(Array,psi[r,c]) for r in 1:2 for c in 1:2]
    for (m,s,p,axs) in zip(ms,sites,invperms,axes)
        restored=PEPSKit.absorb_weight(permute(m,p),w,s,axs;inv=true)
        ts[2*(mod1(s[1],2)-1)+mod1(s[2],2)]=convert(Array,restored)
    end
    finite_arrays(ts)
end
function stages()
    chk=deserialize(joinpath(OUT,"stalled.jls"));psi,w=chk.psi,chk.w
    alg=SimpleUpdate(;trunc=PEPSKit._get_fixedspacetrunc(psi),imaginary_time=true)
    data=Dict{String,Any}()
    for kind in (:A,:B)
        sites=kind==:A ? [CartesianIndex(1,0),CartesianIndex(1,1),CartesianIndex(2,1)] :
                         [CartesianIndex(0,1),CartesianIndex(1,1),CartesianIndex(1,2)]
        ms,axs,ip=PEPSKit._get_cluster_with_weights(psi,sites,w)
        flips=[isdual(space(m,1)) for m in ms[2:end]]
        PEPSKit._flip_virtuals!(ms,flips)
        initial=deepcopy(ms);x=join_open(ms)
        gate=PEPSKit.gate_to_mpo(pauli_gate3(phys_tmap(kind==:A ? STAR_A_mat() : PLAQ_B_mat(),3,UPSPACE),0.05,1.0))
        PEPSKit._apply_gatempo!(ms,gate)
        y=join_open(ms)
        # Independent physical action on the dense open cluster, not gate_to_mpo.
        X=[0.0 1;1 0];Z=[1.0 0;0 -1];I2=Matrix{Float64}(I,2,2)
        O=kind==:A ? X : Z
        factors=(kron(I2,O),kron(O,O),kron(O,I2))
        operated=x
        for (axis,op) in zip((2,5,8),factors)
            operated=dense_mode(operated,op,axis)
        end
        dense_target=cosh(0.05)*x+sinh(0.05)*operated
        gate_error=norm(y-dense_target)/norm(dense_target)
        @test gate_error < 1e-12
        before=wave(psi);target=normalize(cosh(0.05)*before+sinh(0.05)*action(before,kind,1,1))
        raw=stage_wave(ms,psi,w,sites,ip,axs,flips)
        @test abs2(dot(raw,target)) ≈ 1 atol=1e-12
        @test abs2(dot(stage_wave(initial,psi,w,sites,ip,axs,flips),before)) ≈ 1 atol=1e-12
        truncs=PEPSKit._get_cluster_trunc(alg.trunc,sites)
        pa,pb,ws,errs=PEPSKit._get_allprojs(ms,truncs)
        rs,ls=PEPSKit._get_allRLs(ms)
        spectra=[svdvals(reshape(convert(Array,r*l),dim(codomain(r*l)),:)) for (r,l) in zip(rs,ls)]
        # Normalize spectra to squared norm one; expose discarded rather than library-specific error conventions.
        spectra=[s/norm(s) for s in spectra]
        outputs=Any[]
        for chosen in ((1,),(2,),(1,2))
            fitted=deepcopy(ms)
            for i in chosen
                fitted[i]=fitted[i]*PEPSKit.twistdual(pa[i],1)
                # No graded spaces here: dense multiplication implements the left projector independently.
                m=fitted[i+1];p=pb[i]
                arr=convert(Array,m);mat=reshape(convert(Array,p),dim(codomain(p)),:)
                res=reshape(mat*reshape(arr,size(arr,1),:),size(mat,1),size(arr)[2:end]...)
                fitted[i+1]=TensorMap(res,codomain(p)⊗codomain(m,2)⊗codomain(m,3)⊗codomain(m,4),domain(m))
            end
            v=stage_wave(fitted,psi,w,sites,ip,axs,flips)
            localfit=join_open(fitted)
            openerror=max(0.,1-abs2(dot(vec(y),vec(localfit)))/(sum(abs2,y)*sum(abs2,localfit)))
            push!(outputs,Dict("bonds"=>collect(chosen),"stabilizers"=>vals(v),"infidelity"=>max(0.,1-abs2(dot(target,v))),
                "open_optimal_scale_squared_error"=>openerror))
            if chosen==(1,2)
                actual,_,_=step!(deepcopy(psi),deepcopy(w),alg,kind,1,1)
                @test abs2(dot(v,wave(actual))) ≈ 1 atol=1e-12
            end
        end
        data[String(kind)]=Dict("dense_gate_relative_error"=>gate_error,
            "unchanged_open_optimal_scale_squared_error"=>max(0.,1-abs2(dot(vec(y),vec(x)))/(sum(abs2,y)*sum(abs2,x))),
            "open_stabilizer_residual"=>norm(operated-x)/norm(x),
            "open_gate_non_scalar_residual"=>norm(y-exp(0.05)*x)/norm(y),
            "no_trunc_infidelity"=>max(0.,1-abs2(dot(raw,target))),"spectra"=>spectra,
            "discarded_squared_weights"=>[sum(abs2,s[3:end]) for s in spectra],"stages"=>outputs)
        println("$kind ",data[String(kind)]);flush(stdout)
    end
    open(joinpath(OUT,"stages.toml"),"w") do io;TOML.print(io,data);end
end
stages()

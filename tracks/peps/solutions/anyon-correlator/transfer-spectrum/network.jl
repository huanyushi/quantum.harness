struct TransferNetwork
    A::Array{Float64,5}
    Q_raw::Array{Float64,4}
    Q::Array{Float64,4}
    mpo::InfiniteMPO
    fusion::TensorMap
    ket_action::TensorMap
    bra_action::TensorMap
    step::Int
    normalization::Float64
    input_hash::String
end
function mode_array(a,m,axis)
    order=[axis;[i for i in 1:ndims(a) if i!=axis]]; dims=size(a)[order]
    permutedims(reshape(m*reshape(permutedims(a,order),dims[1],:),size(m,1),dims[2:end]...),invperm(order))
end
function build_transfer(st;direction=:down)
    direction==:down || throw(ArgumentError("only the confirmed downward transfer is defined"))
    a=dense_arrays(st); A=zeros(4,2,2,2,2)
    for p in 0:1,q in 0:1,n in 1:2,e in 1:2,s in 1:2,w in 1:2,b in 1:2,c in 1:2
        A[p+2q+1,n,e,s,w]+=a.V[c,b,s,w]*a.P[p+1,b,e]*a.P[q+1,c,n]*a.lambda[b]*a.lambda[e]*a.lambda[c]*a.lambda[n]
    end
    @tensor raw[n,N,e,E,s,S,w,W] := A[p,n,e,s,w]*conj(A[p,N,E,S,W])
    Q_raw=reshape(raw,4,4,4,4)
    v=virtual_space(); W=fuse(v⊗conj(v)); F=unitary(Float64,W,v⊗conj(v))
    f=reshape(convert(Array,F),4,4); Q=copy(Q_raw)
    for i in 1:4; Q=mode_array(Q,f,i); end
    z=TensorMap([1.0 0;0 -1],v←v)
    ket=F*(z⊗id(Float64,conj(v)))*F'
    zb=TensorMap([1.0 0;0 -1],conj(v)←conj(v))
    bra=F*(id(Float64,v)⊗zb)*F'
    scale=norm(Q); scale>0 || error("zero double layer")
    # Ordinary MPO: left,down <- up,right. No seam or twist is inserted.
    local_mpo=TensorMap(permutedims(Q/scale,(4,3,1,2)),W⊗W←W⊗W;tol=1e-12)
    h=bytes2hex(sha256(reinterpret(UInt8,[vec(a.V);vec(a.P);a.lambda])))
    TransferNetwork(A,Q_raw,Q,InfiniteMPO([local_mpo]),F,ket,bra,1,scale,h)
end
"Tiny periodic-row algebra check, never the infinite-plane production spectrum."
function closed_row(net,width)
    width in (2,3) || throw(ArgumentError("algebra checks only support widths 2,3"))
    q=net.Q/net.normalization; d=4^width; t=zeros(d,d)
    tuples=collect(Iterators.product(ntuple(_->1:4,width)...))[:]
    for (i,s) in enumerate(tuples),(j,n) in enumerate(tuples),bonds in tuples
        value=1.0
        for x in 1:width; value*=q[n[x],bonds[x],s[x],bonds[mod1(x-1,width)]]; end
        t[i,j]+=value
    end
    return t
end

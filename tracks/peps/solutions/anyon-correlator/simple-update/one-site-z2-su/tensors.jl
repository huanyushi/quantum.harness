# Geometric arrays: V[N,E,S,W], P[physical,head,tail].
virtual_space() = Z2Space(0=>1, 1=>1)
physical_space() = Z2Space(0=>2)
mutable struct Z2State
    V::TensorMap
    P::TensorMap
    lambda::DiagonalTensorMap
end

"Identify dual Z₂ representations after a geometric reordering (real irreps)."
function orient_like(t, reference)
    inds = Tuple(i for i in 1:numind(t) if space(t,i) != space(reference,i))
    return isempty(inds) ? t : flip(t, inds)
end
rotate(v) = orient_like(permute(v, ((2,3),(4,1))), v)
exchange(p) = orient_like(permute(p, ((1,3),(2,))), p)
function spatial_project(v,p)
    r=rotate(v); r2=rotate(r); r3=rotate(r2)
    return (v+r+r2+r3)/4, (p+exchange(p))/2
end
function state_from_arrays(v,p,l)
    q=virtual_space(); h=physical_space()
    V=TensorMap(v,q⊗q←q⊗q;tol=1e-13)
    P=TensorMap(p,h⊗q←q;tol=1e-13)
    V,P=spatial_project(V,P)
    norm(V)>0 && norm(P)>0 && norm(l)>0 || throw(ArgumentError("zero state"))
    all(isfinite,l) && all(>=(0),l) || throw(ArgumentError("invalid weights"))
    L=DiagonalTensorMap(TensorMap(Matrix(Diagonal(l/norm(l))),q←q))
    return Z2State(V/norm(V),P/norm(P),L)
end
function random_state(seed::Int)
    rng=MersenneTwister(seed)
    v=[iseven(n+e+s+w) ? randn(rng) : 0.0 for n in 0:1,e in 0:1,s in 0:1,w in 0:1]
    p=[a==b ? randn(rng) : 0.0 for p in 0:1,a in 0:1,b in 0:1]
    return state_from_arrays(v,p,ones(2))
end
function exact_reference_state()
    v=[iseven(n+e+s+w) ? 1.0 : 0.0 for n in 0:1,e in 0:1,s in 0:1,w in 0:1]
    p=[a==b ? (-1.0)^(p*a) : 0.0 for p in 0:1,a in 0:1,b in 0:1]
    return state_from_arrays(v,p,ones(2))
end
"Public TensorKit conversion, never reshape raw block storage."
dense_arrays(st)=(V=convert(Array,st.V),P=convert(Array,st.P),lambda=diag(convert(Array,st.lambda)))
function symmetry_residuals(st)
    a=dense_arrays(st); z=[1.0,-1.0]
    vr=sqrt(sum(abs2(a.V[n,e,s,w]*(1-z[n]*z[e]*z[s]*z[w])) for n in 1:2,e in 1:2,s in 1:2,w in 1:2))/norm(st.V)
    pr=sqrt(sum(abs2(a.P[p,a0,b]*(1-z[a0]*z[b])) for p in 1:2,a0 in 1:2,b in 1:2))/norm(st.P)
    Z=TensorMap(Matrix(Diagonal(z)),virtual_space()←virtual_space())
    return (virtual_V=vr,virtual_P=pr,rotation_V=norm(st.V-rotate(st.V))/norm(st.V),
            exchange_P=norm(st.P-exchange(st.P))/norm(st.P),
            weight_commutator=norm(st.lambda*Z-Z*st.lambda)/norm(st.lambda))
end

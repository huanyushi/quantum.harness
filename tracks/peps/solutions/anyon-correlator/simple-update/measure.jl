# Measurement only: no calls to the legacy tc_peps.jl or PEPSKit SU.
using TensorKit, PEPSKit, Random, LinearAlgebra

function measurement_peps(st)
    p, v = ComplexSpace(4), ComplexSpace(2)
    ts = [TensorMap(ComplexF64.(reshape(blocked_tensor(st,r,c),4,16)),
                    p, v⊗v⊗v'⊗v') for r in 1:st.L, c in 1:st.L]
    for a in ts
        normalize!(a)
    end
    return InfinitePEPS(ts)
end

function stabilizer_operators(L)
    p = ComplexSpace(4)
    lat = fill(p,L,L)
    x, z, eye = [0.0 1; 1 0], [1.0 0; 0 -1], Matrix{Float64}(I,2,2)
    # Julia's first physical index is fastest: E then N.
    xe, xn = kron(eye,x), kron(x,eye)
    ze, zn = kron(eye,z), kron(z,eye)
    a = TensorMap(ComplexF64.(kron(xn,xe*xn,xe)),p^3,p^3)
    b = TensorMap(ComplexF64.(kron(zn,ze*zn,ze)),p^3,p^3)
    ops = []
    for r in 1:L, c in 1:L
        ha = LocalOperator{Any,eltype(lat)}(lat)
        hb = LocalOperator{Any,eltype(lat)}(lat)
        PEPSKit.add_term!(ha,[CartesianIndex(r,c-1),CartesianIndex(r,c),CartesianIndex(r+1,c)],a)
        PEPSKit.add_term!(hb,[CartesianIndex(r-1,c),CartesianIndex(r,c),CartesianIndex(r,c+1)],b)
        push!(ops,ha,hb)
    end
    return ops
end

function measure_state(st; chi=8, tol=1e-9, maxiter=250, envseed=2026)
    psi = measurement_peps(st)
    return measure_peps(psi,st.L;chi,tol,maxiter,envseed)
end

function measure_peps(psi,L; chi=8, tol=1e-9, maxiter=250, envseed=2026)
    Random.seed!(envseed)
    t = time()
    env, info = leading_boundary(CTMRGEnv(randn,ComplexF64,psi,ComplexSpace(chi)),psi;
                                tol,maxiter,verbosity=0)
    vals = [real(expectation_value(psi,o,env)) for o in stabilizer_operators(L)]
    # PEPSKit returns the unit-cell SUM; there are 2 L² physical spins.
    energy = -sum(vals)/(2L^2)
    residual = Float64(info.convergence_error)
    return (; energy, vals, maxerr=maximum(abs.(vals .- 1)),
             converged=info.converged, residual, iterations=-1, # 0.8.0 does not return count
             seconds=time()-t, chi)
end

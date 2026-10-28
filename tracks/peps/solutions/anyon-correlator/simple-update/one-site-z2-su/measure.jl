module Measurement
include("../split/core.jl")
using .SplitSU: blocked_tensor
include("../split/measure.jl")
end
function measurement_workspace(st)
    a=dense_arrays(st); w=Measurement.SplitSU.initial_state(:exact,0)
    for n in keys(w.tensors); w.tensors[n]=copy(n[1]==:V ? a.V : a.P); end
    for b in keys(w.weights); w.weights[b]=copy(a.lambda); end
    return w
end
function measure_state(st;chi=8,tol=1e-10,maxiter=400,envseed=2026)
    m=Measurement.measure_state(measurement_workspace(st);chi,tol,maxiter,envseed)
    return merge(m,(iterations=nothing,))
end

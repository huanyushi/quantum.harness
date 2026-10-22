module OneSiteSU
using LinearAlgebra

# Shared numerical kernels and spatially symmetric random initialization.
include("../split/core.jl")

"A spatially symmetric random start, not the analytic toric-code tensor."
function random_state(seed)
    st=SplitSU.initial_state(:random,seed;L=2)
    v=st.tensors[(:V,1,1)]
    v=(v+permutedims(v,(2,3,4,1))+permutedims(v,(3,4,1,2))+
       permutedims(v,(4,1,2,3)))/4
    p=st.tensors[(:E,1,1)]
    p=(p+permutedims(p,(1,3,2)))/2
    normalize!(v);normalize!(p)
    for n in keys(st.tensors)
        st.tensors[n]=copy(n[1]==:V ? v : p)
    end
    return st
end

end

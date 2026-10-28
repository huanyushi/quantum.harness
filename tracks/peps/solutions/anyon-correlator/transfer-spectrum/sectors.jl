struct SectorLabel
    g::Int
    q::Int
    name::String
    function SectorLabel(g::Int,q::Int)
        g in (0,1) && q in (0,1) || throw(ArgumentError("g and q must be 0 or 1"))
        new(g,q,(("1","e"),("m","epsilon"))[g+1][q+1])
    end
end
function sector_backgrounds(pair,label)
    return (left=pair.r0,right=label.g==0 ? pair.r0 : pair.r1,
        leftenv=pair.env0,rightenv=label.g==0 ? pair.env0 : pair.env1,sector=Z2Irrep(label.q))
end
function string_dictionary()
    p=dense_arrays(exact_reference_state()).P
    X=[0. 1;1 0];Z=[1. 0;0 -1];I2=Matrix{Float64}(I,2,2)
    virtual_error=norm(mode_array(p,X,1)-mode_array(p,Z,2))
    # Two consecutive Z links anticommute with the star at each endpoint,
    # while the intermediate star touches both links and commutes.
    Ze=kron(I2,Z); Zw=kron(Z,I2); Xe=kron(I2,X); Xw=kron(X,I2)
    electric=Ze*Zw; magnetic=Xe*Xw
    e_end=max(norm(electric*Xe+Xe*electric),norm(electric*Xw+Xw*electric))
    e_bulk=norm(electric*(Xe*Xw)-(Xe*Xw)*electric)
    m_end=max(norm(magnetic*Ze+Ze*magnetic),norm(magnetic*Zw+Zw*magnetic))
    m_bulk=norm(magnetic*(Ze*Zw)-(Ze*Zw)*magnetic)
    validated=maximum((virtual_error,e_end,e_bulk,m_end,m_bulk))<=1e-12
    return (;validated,virtual_X_equals_Z=virtual_error,e_endpoint_anticommutator=e_end,
        e_interior_commutator=e_bulk,m_endpoint_anticommutator=m_end,m_interior_commutator=m_bulk)
end

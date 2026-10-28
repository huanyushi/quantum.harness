function control_fixture(scale=1.0)
    W=Z2Space(0=>2,1=>2); b=Z2Space(0=>1)
    data=reshape(scale*Matrix(Diagonal([5.,3,2,1])),1,4,4,1)
    op=ToricTransferSpectrum.MPSKit.InfiniteMPO([TensorMap(data,b⊗W←W⊗b)])
    psi=ToricTransferSpectrum.MPSKit.InfiniteMPS([TensorMap(reshape(ComplexF64[1,0,0,0],1,4,1),b⊗W←b)])
    return op,psi
end
@testset "test_statmech_largest_modulus_and_normalization" begin
    for scale in (1.,7.), k in (0.,pi/2)
        op,psi=control_fixture(scale)
        a=ToricTransferSpectrum.solve_modes(op,psi,psi,0,k;num=1)
        b=ToricTransferSpectrum.solve_modes(op,psi,psi,1,k;num=2)
        c=ToricTransferSpectrum.solve_modes(op,psi,psi,1,k;num=1)
        @test only(a.ratios) ≈ 3/5 atol=1e-10
        @test sort(real.(b.ratios)) ≈ [1/5,2/5] atol=1e-10
        @test only(c.ratios) ≈ 2/5 atol=1e-10
        @test maximum([a.residuals;b.residuals;c.residuals]) < 1e-10
    end
end
@testset "test_sector_background_selection" begin
    net=build_transfer(S.exact_reference_state()); pair=solve_boundaries(net)
    for g in 0:1,q in 0:1
        label=SectorLabel(g,q); bg=sector_backgrounds(pair,label)
        @test (bg.left===bg.right)==(g==0)
        @test bg.sector==Z2Irrep(q)
        @test label.name==(("1","e"),("m","epsilon"))[g+1][q+1]
    end
    @test_throws ArgumentError SectorLabel(2,0)
    @test_throws ArgumentError SectorLabel(0,-1)
end
@testset "test_e_m_string_dictionary" begin
    @test ToricTransferSpectrum.string_dictionary().validated
end
@testset "test_zero_empty_and_unconverged_are_distinct" begin
    status=ToricTransferSpectrum.mode_status
    @test status(0.,0.,1;zero_proven=true)=="zero_mode"
    @test status(1e-14,1e-14,1)=="below_resolution"
    @test status(0.,0.,0)=="empty_sector"
    @test status(0.2,1e-2,1)=="not_converged"
    @test status(0.2,0.,1;budget=true)=="budget"
    @test status(0.2,1e-14,1)=="converged"
    net=build_transfer(S.exact_reference_state());pair=solve_boundaries(net)
    for g in 0:1,q in 0:1
        pts=solve_sector(net,pair,SectorLabel(g,q);momenta=[0.,pi/2],num=1,options=(;reference=true))
        @test length(pts)==2
        @test all(p->p.status=="zero_mode" && p.quasi_energy===nothing && p.phase===nothing,pts)
        @test maximum(p.projected_residual for p in pts)<1e-12
    end
end

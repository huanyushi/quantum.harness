@testset "test_exact_boundary_representatives" begin
    net=build_transfer(S.exact_reference_state())
    pair=solve_boundaries(net;options=(;tol=1e-10,maxiter=40))
    @test pair.diagnostics.converged
    @test pair.diagnostics.chi==1
    @test pair.diagnostics.density_mismatch < 1e-10
    @test pair.diagnostics.overlap01 < 1e-10
    @test pair.diagnostics.diagonal_overlap0 ≈ 1 atol=1e-10
    @test pair.diagnostics.diagonal_overlap1 ≈ 1 atol=1e-10
    @test pair.diagnostics.single_layer_overlap ≈ 1 atol=1e-10
    @test pair.diagnostics.adjoint_density_mismatch < 1e-10
    mktempdir() do dir
        file=joinpath(dir,"boundaries.json")
        ToricTransferSpectrum.save_boundaries(file,pair)
        data=S.JSON.parsefile(file)
        @test data["r0"][1]["left_charge_dims"]["0"]==1
        @test data["diagnostics"]["converged"]===true
    end
end
@testset "test_gauge_equivalent_boundary_is_not_distinct" begin
    net=build_transfer(S.exact_reference_state())
    a=ToricTransferSpectrum.product_boundary(net)
    b=ToricTransferSpectrum.product_boundary(net;phase=-1)
    @test ToricTransferSpectrum.boundary_overlap(a,b) ≈ 1 atol=1e-12
    @test !ToricTransferSpectrum.distinct_boundaries(a,b)
    c=ToricTransferSpectrum.apply_group(a,net.ket_action)
    @test ToricTransferSpectrum.distinct_boundaries(a,c)
end
@testset "test_boundary_zero_schmidt_support" begin
    net=build_transfer(S.exact_reference_state())
    a=ToricTransferSpectrum.product_boundary(net)
    reduced=ToricTransferSpectrum.remove_zero_support(a)
    @test ToricTransferSpectrum.boundary_overlap(a,reduced) ≈ 1 atol=1e-12
    @test dim(space(reduced.AL[1],2),Z2Irrep(0))==2
    @test dim(space(reduced.AL[1],2),Z2Irrep(1))==2
    # A canonical redundant product representation with Schmidt values (1,0).
    # The constructor with explicit AL,AR,C keeps this zero support available
    # for the truncation routine, rather than silently fixing it beforehand.
    W=space(a.AL[1],2); b=Z2Space(0=>2)
    v=vec(convert(Array,a.AL[1]));raw=zeros(ComplexF64,2,4,2)
    raw[1,:,1]=v;raw[2,:,2]=v
    A=TensorMap(raw,b⊗W←b); C=TensorMap(ComplexF64[1 0;0 0],b←b)
    padded=ToricTransferSpectrum.MPSKit.InfiniteMPS([A],[A],[C])
    reduced=ToricTransferSpectrum.remove_zero_support(padded)
    @test dim(space(padded.AL[1],1))==2
    @test dim(space(reduced.AL[1],1))==1
    @test ToricTransferSpectrum.boundary_overlap(a,reduced) ≈ 1 atol=1e-12
    @test dim(space(reduced.AL[1],2),Z2Irrep(1))==2
end

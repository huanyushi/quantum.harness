@testset "test_input_state_provenance" begin
    mktempdir() do dir
        p=joinpath(dir,"state.json")
        S.save_state(p,S.exact_reference_state();metadata=(;origin="reference",accepted=true))
        @test_throws ArgumentError ToricTransferSpectrum.load_production_state(p)
        @test ToricTransferSpectrum.load_production_state(p;reference=true)[1] isa S.Z2State
        numerical=joinpath(dir,"numerical.json")
        S.save_state(numerical,S.random_state(5);metadata=(;origin="numerical_random_optimization",accepted=false))
        @test_throws ArgumentError ToricTransferSpectrum.load_production_state(numerical)
    end
end
@testset "test_incremental_output_and_no_overwrite" begin
    mktempdir() do dir
        @test_throws ArgumentError S.prepare_output(dir)
        file=joinpath(dir,"spectrum.jsonl")
        p=SectorPoint(0,0,0.,1,0.,0.,0.,nothing,nothing,0.,1,"zero_mode",nothing)
        ToricTransferSpectrum.append_point(file,p)
        @test length(readlines(file))==1
        @test S.JSON.parse(only(readlines(file)))["quasi_energy"]===nothing
        b=ToricTransferSpectrum.budget_point(SectorLabel(1,1),pi)
        ToricTransferSpectrum.append_point(file,b)
        @test length(readlines(file))==2
        @test S.JSON.parse(last(readlines(file)))["status"]=="budget"
        @test !occursin("Infinity",read(file,String))
        @test !occursin("NaN",read(file,String))
    end
end
@testset "test_plot_zero_and_invalid_modes" begin
    @test success(`python3 $(joinpath(@__DIR__,"plot_tests.py"))`)
end

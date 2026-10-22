# Catch renamed scientific modules or broken paths to retained numerical states.
using Test, Serialization, JSON, JLD2

const PROJECT = normpath(joinpath(@__DIR__, ".."))

# Historical Serialization files name Main.SplitSU and Main.FieldLoop.
# Keep those definitions when checking the original binary objects.
Base.include(Main, joinpath(PROJECT, "simple-update", "split", "core.jl"))
Base.include(Main, joinpath(PROJECT, "simple-update", "loop", "core.jl"))

@testset "retained loop states preserve binary and portable numbers" begin
    for name in ("d2-xz", "d3-xz", "d4-xz")
        snapshot = JSON.parsefile(joinpath(PROJECT, "simple-update", "loop", "frozen_states", "$name.json"))
        @test snapshot["L"] == 2
        @test length(snapshot["tensors"]) == 12
        binary = joinpath(PROJECT, "simple-update", "loop", "results", name, "state.jls")
        if isfile(binary)
            checkpoint = deserialize(binary)
            st = checkpoint.state
            @test st.D == snapshot["D"]
            for t in snapshot["tensors"]
                key = (Symbol(t["key"][1]), Int(t["key"][2]), Int(t["key"][3]))
                @test vec(st.tensors[key]) == Float64.(t["data"])
            end
            for w in snapshot["weights"]
                key = Tuple(Int.(w["key"]))
                @test st.weights[key] == Float64.(w["data"])
            end
        else
            @test_skip isfile(binary) # Binary results are local; portable states are tracked.
        end
    end
end

@testset "retained split-SU zero-field state still loads" begin
    file = joinpath(PROJECT, "simple-update", "split", "results", "loop-seed1", "random-seed1.jls")
    if isfile(file)
        checkpoint = deserialize(file)
        @test checkpoint.step == 120
        @test checkpoint.state.L == 2
        @test length(checkpoint.state.tensors) == 12
    else
        @test_skip isfile(file)
    end
end

@testset "AD portable parameters refer to unchanged accepted states" begin
    selected = JSON.parsefile(joinpath(PROJECT, "ad", "split", "selected_fields.json"))
    for label in ("x", "z", "xz")
        portable = JSON.parsefile(joinpath(PROJECT, "ad", "split", "frozen_states", "$label.json"))
        @test portable["D"] == 2
        @test length(portable["parameters"]) == 40
        checkpoint = joinpath(PROJECT, "ad", "split", dirname(selected[label]), "final.jld2")
        if isfile(checkpoint)
            data = load(checkpoint)
            @test data["parameters"] == Float64.(portable["parameters"])
        else
            @test_skip isfile(checkpoint)
        end
    end
end

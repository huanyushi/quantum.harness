# Portable numerical states, with exact JSON round-trip verification.
using JLD2, JSON, SHA

function export_states()
    out=joinpath(@__DIR__,"frozen_states")
    mkpath(out)
    for (label,cohort) in (("x","unrolled-repair01"),("z","unrolled-repair01"),("xz","unrolled-repair02"))
        dir=joinpath(@__DIR__,"results",cohort,label)
        path=joinpath(dir,"final.jld2")
        saved=load(path)
        summary=JSON.parsefile(joinpath(dir,"summary.json"))
        summary["audit"]["verdict"]["accepted"] || error("cannot export unaccepted state $label")
        params=Vector{Float64}(saved["parameters"])
        data=Dict("format"=>"real uniform V-lambda-P flat parameters; network.jl layout",
                  "D"=>Int(saved["D"]),"hx"=>Float64(saved["hx"]),"hz"=>Float64(saved["hz"]),
                  "uniform"=>true,"virtual_z2_constraint"=>false,"parameters"=>params,
                  "initialization"=>"analytic zero-field anchor then AD, historical checkpoint continuation",
                  "source_checkpoint_sha256"=>bytes2hex(sha256(read(path))),
                  "source_summary_sha256"=>bytes2hex(sha256(read(joinpath(dir,"summary.json")))),
                  "energy"=>summary["measurement"]["energy"],"mx"=>summary["measurement"]["mx"],
                  "mz"=>summary["measurement"]["mz"],"gradient_method"=>summary["gradient_method"],
                  "unroll_steps"=>summary["unroll_steps"],"gradnorm"=>summary["gradnorm"],
                  "accuracy"=>Dict("energy_per_spin"=>1e-5,"relative_nonzero_magnetization"=>0.01))
        target=joinpath(out,label*".json")
        open(target,"w") do io;write(io,JSON.json(data));write(io,'\n');end
        Vector{Float64}(JSON.parsefile(target)["parameters"])==params || error("JSON changed numerical parameters")
        println("已导出并核对：$target");flush(stdout)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__;export_states();end

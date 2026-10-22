# Isolated namespaces share the installed stack and its compilation within one process.
module OracleChecks
include("tests.jl")
end
module PilotExperiment
include("pilot.jl")
end
module FollowupExperiment
include("followup.jl")
end
module BranchExperiment
include("branch_check.jl")
end
module LegacyExperiment
include("legacy_check.jl")
end
repo=normpath(joinpath(@__DIR__,"..","..","..","..","..","..",".."))
run(`$(joinpath(repo,".venv","bin","python")) $(joinpath(@__DIR__,"analyze.py"))`)

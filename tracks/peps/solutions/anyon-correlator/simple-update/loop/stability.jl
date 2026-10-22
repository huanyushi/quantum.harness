# Three branches from exactly the same frozen D=2 diagonal-field state.
include("run.jl")
root=joinpath(@__DIR__,"results")
source=joinpath(root,"d2-xz","state.jls")
for (name,schedule,als,every) in (("d2-xz-continue","0.002:600",12,200),
                                ("d2-xz-dt-half","0.001:1200",12,400),
                                ("d2-xz-als24","0.002:600",24,200))
    main(["--D=2","--hx=0.1","--hz=0.1","--resume=$source","--schedule=$schedule",
          "--als-sweeps=$als","--every=$every","--out=$(joinpath(root,name))","--final-chis=8,16"])
end

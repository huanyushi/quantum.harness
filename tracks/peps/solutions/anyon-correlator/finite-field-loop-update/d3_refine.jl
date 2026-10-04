include("run.jl")
root=joinpath(@__DIR__,"results")
for (name,source,hx,hz,schedule,als,seed) in (
    ("d3-xz-dt-half","d3-xz",0.1,0.1,"0.0025:200",12,2026),
    ("d3-x-continue","d3-x",0.1,0.0,"0.005:200",12,17),
    ("d3-x-als24","d3-x",0.1,0.0,"0.005:200",24,17))
    main(["--D=3","--hx=$hx","--hz=$hz","--resume=$(joinpath(root,source,"state.jls"))",
          "--schedule=$schedule","--als-sweeps=$als","--chi=8","--final-chis=8","--envseed=$seed",
          "--every=100","--out=$(joinpath(root,name))","--max-seconds=450"])
end

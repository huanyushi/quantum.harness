include("run.jl")
root=joinpath(@__DIR__,"results")
for (name,hx,hz) in (("z","0","0.1"),("x","0.1","0"))
    main(["--D=3","--hx=$hx","--hz=$hz","--resume=$(joinpath(root,"d2-$name","state.jls"))",
          "--schedule=0.02:100,0.005:200","--chi=12","--final-chis=8,12","--every=50",
          "--out=$(joinpath(root,"d3-$name"))","--max-seconds=450"])
end

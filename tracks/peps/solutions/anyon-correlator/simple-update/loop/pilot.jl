include("run.jl")
for (name,hx,hz) in (("z","0","0.1"),("x","0.1","0"),("xz","0.1","0.1"))
    main(["--D=2","--hx=$hx","--hz=$hz","--schedule=0.05:200,0.01:200,0.002:200",
          "--out=$(joinpath(@__DIR__,"results","d2-$name"))","--max-seconds=450","--every=40"])
end

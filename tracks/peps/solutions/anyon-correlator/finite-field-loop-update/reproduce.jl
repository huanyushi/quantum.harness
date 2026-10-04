# Run stages separately so each compute budget is explicit.
include("run.jl")
root=joinpath(@__DIR__,"results")
stage=isempty(ARGS) ? "d2" : ARGS[1]
if stage=="d2"
    for (name,hx,hz) in (("z",0.0,0.1),("x",0.1,0.0),("xz",0.1,0.1))
        main(["--D=2","--hx=$hx","--hz=$hz","--every=40","--out=$(joinpath(root,"d2-$name"))"])
    end
elseif stage=="stability"
    source=joinpath(root,"d2-xz","state.jls")
    for (name,schedule,als,every) in (("d2-xz-continue","0.002:600",12,200),
                                    ("d2-xz-dt-half","0.001:1200",12,400),
                                    ("d2-xz-als24","0.002:600",24,200))
        main(["--hx=0.1","--hz=0.1","--resume=$source","--schedule=$schedule","--als-sweeps=$als",
              "--every=$every","--out=$(joinpath(root,name))"])
    end
elseif stage in ("d3-z","d3-x","d3-xz")
    name=replace(stage,"d3-"=>"")
    hx=name=="z" ? 0.0 : 0.1;hz=name=="x" ? 0.0 : 0.1
    main(["--D=3","--hx=$hx","--hz=$hz","--resume=$(joinpath(root,"d2-$name","state.jls"))",
          "--schedule=0.02:100,0.005:200","--chi=12","--final-chis=8,12","--every=50",
          "--out=$(joinpath(root,stage))","--max-seconds=450"])
elseif stage=="qr-recheck"
    # Independent fresh trajectories with the optimized but algebraically
    # equivalent contraction, compared to retained dense-ALS D2 endpoints.
    for (name,hx,hz) in (("z",0.0,0.1),("x",0.1,0.0),("xz",0.1,0.1))
        main(["--D=2","--hx=$hx","--hz=$hz","--every=200","--final-chis=8",
              "--out=$(joinpath(root,"d2-$name-qr"))"])
    end
else
    error("Stage must be d2, stability, d3-z, d3-x, d3-xz, or qr-recheck. D4 needs a separately budgeted run.")
end

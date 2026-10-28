const JSON=OneSiteZ2SU.JSON
const POINT_FIELDS=Tuple(filter(!=(:state),collect(fieldnames(SectorPoint))))
function load_production_state(path;reference=false)
    st,md=load_state(path)
    valid=get(md,"origin",nothing)=="numerical_random_optimization" && get(md,"accepted",false)===true
    valid || reference || throw(ArgumentError("production spectrum requires an accepted numerical state"))
    return st,md
end
budget_point(label,k)=SectorPoint(label.g,label.q,Float64(k),1,nothing,nothing,nothing,nothing,nothing,nothing,0,"budget",nothing)
function point_data(p)
    data=Dict{String,Any}()
    for f in POINT_FIELDS
        v=getproperty(p,f)
        data[string(f)]=v isa Number && !isfinite(v) ? nothing : v
    end
    data
end
function append_point(path,p)
    open(path,"a") do io; JSON.print(io,point_data(p));println(io);flush(io);end
end
function write_csv(jsonl,path)
    open(path,"w") do io
        println(io,join(string.(POINT_FIELDS),','))
        for line in eachline(jsonl)
            d=JSON.parse(line)
            println(io,join([d[string(f)]===nothing ? "" : string(d[string(f)]) for f in POINT_FIELDS],','))
        end
        flush(io)
    end
end
function boundary_data(psi)
    [Dict("left_charge_dims"=>Dict(string(q)=>dim(space(psi.AL[i],1),Z2Irrep(q)) for q in 0:1),
        "shape"=>collect(size(convert(Array,psi.AL[i]))),
        "AL_real"=>vec(real.(convert(Array,psi.AL[i]))),"AL_imag"=>vec(imag.(convert(Array,psi.AL[i]))),
        "physical_space"=>string(space(psi.AL[i],2)),"partition"=>"left physical <- right") for i in 1:length(psi)]
end
function save_boundaries(path,pair)
    OneSiteZ2SU.write_json(path,(;format="z2-boundaries-v1",r0=boundary_data(pair.r0),r1=boundary_data(pair.r1),
        adjoint0=boundary_data(pair.adjoint0),adjoint1=boundary_data(pair.adjoint1),diagnostics=pair.diagnostics))
end

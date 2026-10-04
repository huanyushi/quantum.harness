"""Publish the frozen-state VUMPS comparison, including failed candidates."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"results"/"vumps"


def read(name):
    return json.loads((BASE/name/"result.json").read_text())


def save(fig,name):
    for ext in ("svg","png"):
        fig.savefig(ROOT/"figures"/f"{name}.{ext}",dpi=170)
    plt.close(fig)


def main():
    rows=[]
    for file in sorted(BASE.glob("*/result.json")):
        r=json.loads(file.read_text())
        if r["state"]!="exact":
            source=(ROOT/"frozen_states"/(r["state"]+".json") if r["source"].endswith(".json")
                    else ROOT/"results"/r["state"]/"state.jls")
            assert hashlib.sha256(source.read_bytes()).hexdigest()==r["checkpoint_sha256"]
        if "measurement" not in r:
            print("NO OBSERVABLES",file.parent.name,r["status"])
            continue
        b=r["boundaries"];m=r["measurement"]
        norms=[v["normcell"] for v in b]
        row=dict(run=file.parent.name,state=r["state"],D=r["D"],chi=r["chi"],seed=r["seed"],balanced=r["balanced"],
                 status=r["status"],error_down=b[0]["projected_error"],error_up=b[1]["projected_error"],
                 norm_down=norms[0],norm_up=norms[1],norm_direction_relative_difference=abs(norms[0]-norms[1])/max(norms),
                 energy=m["energy"],mx=m["mx"],mz=m["mz"],astar=m["astar"],bplaq=m["bplaq"],
                 imagmax=m["imagmax"],seconds=r["seconds"],checkpoint_sha256=r["checkpoint_sha256"])
        rows.append(row)
        print(json.dumps(row),flush=True)
    with (ROOT/"vumps_summary.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)

    exact=read("exact-chi4-seed1-balancedfalse")
    assert exact["exact_control_passed"]
    chosen=read("d3-xz-chi16-seed2-balancedtrue")
    ctm=json.loads((ROOT/"results"/"d3-xz"/"final.json").read_text())[0]
    agreement={k:abs(chosen["measurement"][k]-ctm[k]) for k in ("energy","mx","mz","astar","bplaq")}
    assert max(agreement.values())<1e-8
    print("D3 VUMPS/CTMRG difference",agreement,flush=True)
    (ROOT/"vumps_agreement.json").write_text(json.dumps(agreement,indent=2)+"\n")

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for label,key,color in (("D3 raw gauge", "d3-xz-chi8-seed1-balancedfalse", "#c66a17"),
                            ("D3 balanced gauge", "d3-xz-chi8-seed1-balancedtrue", "#2166ac"),
                            ("D4 balanced, χ=16", "d4-xz-chi16-seed1-balancedtrue", "#777777")):
        r=read(key)
        for ax,b in zip(axes,r["boundaries"]):
            h=b["history"]
            its=[x["iteration"] for x in h]+[b["iterations"]]
            errors=[x["error"] for x in h]+[b["projected_error"]]
            ax.semilogy(its,errors,"o-",label=label,color=color,ms=4)
    for ax,title in zip(axes,("Downward boundary","Upward boundary")):
        ax.axhline(1e-9,color="black",ls="--",lw=1)
        ax.set(xlabel="VUMPS iteration",ylabel="Projected residual",title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.supxlabel("Frozen PEPS: gauge balancing changes conditioning, not the physical state.",fontsize=9)
    save(fig,"vumps_residuals")

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    labels=["χ=8, seed 1","χ=16, seed 1","χ=16, seed 2","χ=16, seed 17"]
    names=["d3-xz-chi8-seed1-balancedtrue","d3-xz-chi16-seed1-balancedtrue",
           "d3-xz-chi16-seed2-balancedtrue","d3-xz-chi16-seed17-balancedtrue"]
    states=[read(name) for name in names]
    lam=max(s["boundaries"][0]["normcell"] for s in states)
    ratios=[1-s["boundaries"][0]["normcell"]/lam for s in states]
    axes[0].bar(range(4),np.array(ratios)*1e6,color=["#2166ac"]*3+["#c66a17"])
    axes[0].set(ylabel="Relative norm-density deficit (×10⁻⁶)",title="Smaller residual does not select the dominant branch")
    vals=[(s["measurement"]["energy"]-ctm["energy"])*1e6 for s in states]
    axes[1].bar(range(4),vals,color=["#2166ac"]*3+["#c66a17"])
    axes[1].set(ylabel="Energy difference from CTMRG branch (×10⁻⁶)",title="Lower measured energy is not the selection rule")
    for ax in axes:
        ax.set_xticks(range(4),labels,rotation=15,ha="right");ax.grid(axis="y",alpha=.2)
    save(fig,"vumps_d3_branches")

    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    names=["d4-xz-chi8-seed1-balancedtrue","d4-xz-chi16-seed1-balancedtrue-continued","d4-xz-chi24-seed1-balancedtrue"]
    states=[read(name) for name in names]
    chi=[8,16,24]
    ref=json.loads((ROOT/"results"/"d4-xz"/"reference.json").read_text())["order10"]
    for k,color in (("mx","#2166ac"),("mz","#c66a17")):
        vals=[100*(s["measurement"][k]/ref[k]-1) for s in states]
        axes[0].plot(chi,vals,"o--",label=k,color=color)
    axes[0].set(xlabel="Boundary χ",ylabel="Deviation from series (%)",title="D4 checkpoint: candidate observables")
    axes[0].legend()
    axes[1].semilogy(chi,[max(b["projected_error"] for b in s["boundaries"]) for s in states],"o-",color="0.3")
    axes[1].axhline(1e-9,color="black",ls="--",lw=1)
    axes[1].set(xlabel="Boundary χ",ylabel="Maximum boundary residual",title="χ=24 has not converged")
    for ax in axes:ax.set_xticks(chi);ax.grid(alpha=.2)
    fig.supxlabel("Dashed candidate curve is not a χ-convergence claim. Physical PEPS D=4 is fixed.",fontsize=9)
    save(fig,"vumps_d4_candidates")


if __name__=="__main__":main()

"""Summarize recorded observations, retaining rejected contraction branches."""
from pathlib import Path
import csv
import json
import tomllib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read_json(name, filename="final.json"):
    return json.loads((RESULTS / name / filename).read_text())


def save(fig, name):
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for extension in ("svg", "png"):
        fig.savefig(out / f"{name}.{extension}", dpi=170)
    plt.close(fig)


def main():
    selection = [
        ("d2-z", "final.json", 1, "stable; magnetization error >1%"),
        ("d2-x", "final.json", 1, "stable; within provisional targets"),
        ("d2-xz", "final.json", 1, "stable; z magnetization error >1%"),
        ("d3-z", "final.json", 1, "stable; within provisional targets"),
        ("d3-x", "sequential.json", 0, "provisional; larger-chi/ALS issues"),
        ("d3-xz", "final.json", 0, "selected non-product branch; within targets"),
        ("d4-xz", "sequential.json", 0, "not accepted; unfinished and branch-sensitive"),
    ]
    rows = []
    for name, filename, index, status in selection:
        m = read_json(name, filename)[index]
        ref = read_json(name, "reference.json")
        p = tomllib.loads((RESULTS / name / "settings.toml").read_text())
        row = dict(run=name, D=int(p["D"]), hx=ref["hx"], hz=ref["hz"], status=status,
                   selected_source=f"results/{name}/{filename}[{index}]",
                   energy=m["energy"], mx=m["mx"], mz=m["mz"], astar=m["astar"], bplaq=m["bplaq"],
                   chi=m["chi"], envseed=m["envseed"], ctm_converged=m["ctm_converged"],
                   ctm_residual=m["ctm_residual"], physical_excess=m["physical_excess"],
                   sector_error=m["sector_error"], site_spread=m["site_spread"],
                   energy_error=m["energy"]-ref["order10"]["energy"],
                   mx_error=m["mx"]-ref["order10"]["mx"], mz_error=m["mz"]-ref["order10"]["mz"])
        for k in ("mx", "mz"):
            row[k+"_relative_error"] = (row[k+"_error"]/ref["order10"][k]
                                          if ref["order10"][k] else None)
        rows.append(row)
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
    with (ROOT / "benchmark_summary.csv").open("w", newline="") as file:
        writer=csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)

    # Reference order sensitivity, explicitly not certified uncertainty.
    refs = []
    for name in ("d2-z", "d2-x", "d2-xz"):
        data=read_json(name,"reference.json")
        row={"hx":data["hx"],"hz":data["hz"],**data["order10"]}
        row.update({"order_difference_"+k:abs(data["order10"][k]-data["order8"][k]) for k in ("energy","mx","mz")})
        refs.append(row)
    with (ROOT / "reference_values.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=refs[0].keys());writer.writeheader();writer.writerows(refs)
    print("REFERENCES", refs, flush=True)

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    labels=[]
    for j,row in enumerate(rows):
        labels.append(row["run"])
        axes[0].bar(j-.16,100*(row["mx_relative_error"] or 0),.3,color="#2166ac",label="mx" if j==0 else None)
        axes[0].bar(j+.16,100*(row["mz_relative_error"] or 0),.3,color="#c66a17",label="mz" if j==0 else None)
        axes[1].bar(j,abs(row["energy_error"]),color="0.55" if j>=4 else "#2166ac")
    axes[0].axhline(1,color="0.3",ls="--",lw=1);axes[0].axhline(-1,color="0.3",ls="--",lw=1)
    axes[0].set(ylabel="Magnetization error vs series (%)",title="Nonzero reference components")
    axes[1].set(yscale="log",ylabel="Absolute energy error per edge spin",title="Energy alone is insufficient")
    axes[1].axhline(1e-5,color="0.3",ls="--",lw=1)
    for ax in axes:
        ax.set_xticks(range(len(rows)),labels,rotation=35,ha="right");ax.grid(axis="y",alpha=.2)
    axes[0].legend()
    fig.supxlabel("D3-x is provisional; D4-xz is NOT accepted. A displayed converged branch is not a success claim.",fontsize=9)
    save(fig,"benchmark_errors")

    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    ref=read_json("d2-xz","reference.json")["order10"]["mx"]
    for D,color in ((2,"#2166ac"),(3,"#c66a17"),(4,"#777777")):
        with (RESULTS/f"d{D}-xz"/"trajectory.csv").open() as file:
            data=list(csv.DictReader(file))
        t=np.array([float(r["tau"]) for r in data])
        ok=np.array([r["ctm_converged"]=="true" and float(r["physical_excess"])<1e-6 for r in data])
        for ax,k in zip(axes,("mx","mz")):
            val=np.array([float(r[k]) for r in data])
            ax.plot(t,np.where(ok,val,np.nan),"o-",color=color,label=f"D={D}",ms=4)
            ax.plot(t[~ok],val[~ok],"x",color=color,ms=7)
    for ax,k in zip(axes,("mx","mz")):
        ax.axhline(ref,color="black",ls="--",label="10th-order energy derivative")
        ax.set(xlabel="Accumulated imaginary time",ylabel=k,ylim=(.047,.055));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("hx=hz=0.1: random D2 start, then D3/D4 bond growth")
    fig.supxlabel("×: rejected contraction; initial transients outside the vertical window. D4 stopped before convergence.",fontsize=9)
    save(fig,"diagonal_trajectory")

    names=("d2-xz-continue","d2-xz-dt-half","d2-xz-als24")
    stable=[]
    for n in names:
        m=read_json(n)[1];stable.append((n,m["energy"],m["mx"],m["mz"]))
    print("D2 STABILITY",stable,flush=True)
    # Numerical cross-check against fresh trajectories run with the optimized
    # QR kernel; comparing only the final physical observables, not gauges.
    recheck = []
    for name in ("z", "x", "xz"):
        original = read_json(f"d2-{name}")[0]
        optimized = read_json(f"d2-{name}-qr")[0]
        diff = max(abs(original[k]-optimized[k]) for k in ("energy", "mx", "mz", "astar", "bplaq"))
        assert original["ctm_converged"] and optimized["ctm_converged"]
        assert diff < 1e-8, (name,diff)
        recheck.append({"point":name,"max_observable_difference":diff})
    (ROOT/"qr_recheck.json").write_text(json.dumps(recheck,indent=2)+"\n")
    print("QR RECHECK",recheck,flush=True)
    fig,ax=plt.subplots(figsize=(7,3.8),layout="constrained")
    vals=np.array([r[2:] for r in stable]);delta=(vals-vals[0])*1e6
    x=np.arange(3)
    ax.bar(x-.15,delta[:,0],.3,label="mx",color="#2166ac")
    ax.bar(x+.15,delta[:,1],.3,label="mz",color="#c66a17")
    ax.set(xticks=x,xticklabels=["dt=.002, ALS≤12","dt=.001, ALS≤12","dt=.002, ALS≤24"],
           ylabel="Change from continuation baseline (×10⁻⁶)",title="Equal additional imaginary time: 1.2, same D2 checkpoint")
    ax.legend();ax.grid(axis="y",alpha=.2)
    save(fig,"d2_stability")

    allchecks=[]
    for folder in sorted(RESULTS.iterdir()):
        if not folder.is_dir():continue
        for filename in ("final.json","sequential.json","positive-boundary.json","frozen-extra.json"):
            file=folder/filename
            if not file.exists():continue
            for i,m in enumerate(json.loads(file.read_text())):
                allchecks.append(dict(run=folder.name,source=filename,index=i,chi=m["chi"],envseed=m["envseed"],
                                      converged=m["ctm_converged"],residual=m["ctm_residual"],energy=m["energy"],mx=m["mx"],mz=m["mz"]))
    with (ROOT/"contraction_audit.csv").open("w",newline="") as file:
        w=csv.DictWriter(file,fieldnames=allchecks[0].keys());w.writeheader();w.writerows(allchecks)

    # Same frozen state, two initial environments: both may report convergence.
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    checks=[("Random, simultaneous",read_json("d3-xz")[0]),
            ("Random, sequential",read_json("d3-xz","sequential.json")[1]),
            ("Factorized identity",read_json("d3-xz","positive-boundary.json")[1])]
    for ax,key in zip(axes,("energy","mz")):
        values=[m[key] for _,m in checks]
        ax.bar(range(3),values,color=["#2166ac","#2166ac","#c66a17"])
        ax.set_xticks(range(3),[label for label,_ in checks],rotation=15,ha="right")
        ax.set_ylabel(key);ax.grid(axis="y",alpha=.2)
    axes[0].set_ylim(-1.009,-1.004)
    axes[1].axhline(ref,color="black",ls="--",label="Series reference")
    axes[1].legend(fontsize=8)
    fig.suptitle("Same frozen D3 state, χ=12: a small CTMRG residual is insufficient")
    fig.supxlabel("Identity-boundary branch is not accepted as physical evidence; finite-cylinder audit is reported separately.",fontsize=9)
    save(fig,"ctm_branches")


if __name__=="__main__":main()

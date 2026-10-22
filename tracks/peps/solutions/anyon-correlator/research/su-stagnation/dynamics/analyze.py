"""Recompute analytic predictions and check retained SU evidence; no physics run."""
from pathlib import Path
import csv
import io
import json
import math
import hashlib
import tomllib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
pilot=json.loads((ROOT/"results/pilot/pilot.json").read_text())
follow=json.loads((ROOT/"results/followup/followup.json").read_text())
branches=json.loads((ROOT/"results/branch-check/branches.json").read_text())

def star_map(k,dt):
    if k==0:
        return 0.0
    t=math.tanh(k)
    a=math.exp(2*dt)*(1-t*t)/t
    r=2/(math.sqrt(a*a+4)+a)
    return math.atanh(r*r/t)

def profile(dt):
    low=math.atanh(math.sinh(2*dt)/(math.exp(4*dt)/math.cosh(dt)**2-math.cosh(2*dt)))/2
    blo,bhi=math.tanh(2*low),math.tanh(2*(low+dt))
    return low,blo,bhi,-0.5-(blo+bhi)/4

scaling=[]
max_step_error=0.0
for label,case in pilot.items():
    rows=case if label=="legacy" else case.get("probes",[]) if isinstance(case,dict) else []
    for row in rows:
        assert row["fit_support"]>1-1e-10 and row["fit_residual"]<1e-10
        predicted=list(row["K_before"])
        if row["kind"]=="A":
            predicted[3]=star_map(predicted[3],row["dt"])
            k=row["K_before"][3]
            scaling.append(dict(case=label,dt=row["dt"],K=k,
                measured_rate=(row["K_after"][3]-k)/row["dt"],
                analytic_rate=(predicted[3]-k)/row["dt"],limit_rate=-2*math.tanh(2*k),
                velocity_norm=row["velocity_norm"],infidelity=row["infidelity"]))
        else:
            predicted[0]+=row["dt"]
        err=max(abs(a-b) for a,b in zip(predicted,row["K_after"]))
        max_step_error=max(max_step_error,err)
        assert err<1e-10

for spec in follow["spectra"]:
    assert max(spec["difference"])<1e-11
max_trajectory_error=max(s["map_error"] for r in follow["trajectories"] for s in r["samples"])
assert max_trajectory_error<1e-9
for row in follow["trajectories"]:
    for sample in row["samples"]:
        assert sample["support"]>1-1e-10 and sample["fit_residual"]<1e-10
for row in follow["fixed_points"]:
    low,blo,bhi,e=profile(row["dt"])
    assert abs(e-row["infinite_energy"])<1e-13
    assert row["sweep_infidelity"]<1e-12 and row["map_error"]<1e-11
    for chi in (8,16):
        if f"chi{chi}" in row:
            m=row[f"chi{chi}"]
            assert m["converged"] and abs(m["energy"]-e)<1e-9
for row in branches:
    assert row["piecewise_error"]<1e-11

legacy_path=ROOT.parent/"single-step/results/audit.toml"
# The committed prior report retains this value even when local raw data are absent.
legacy=tomllib.loads(legacy_path.read_text())["stalled_chi16"] if legacy_path.exists() else {"energy":-0.7533036457686739}
low,blo,bhi,e=profile(0.05)
legacy_error=abs(e-legacy["energy"])
assert legacy_error<1e-10
summary=dict(max_single_step_K_error=max_step_error,max_trajectory_K_error=max_trajectory_error,
    legacy_energy_reference="research/su-stagnation/single-step/REPORT.md, infinite-network table",
    legacy_prediction_energy=e,legacy_measured_energy=legacy["energy"],legacy_energy_error=legacy_error,
    zero_step_limit_K=math.log(3)/4,zero_step_limit_B=0.5,zero_step_limit_energy=-0.75,
    trajectory_count=len(follow["trajectories"]),branch_cases=len(branches),
    source_hashes_at_analysis={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.jl")})
(ROOT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
with (ROOT/"dt_scaling.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(scaling[0]),lineterminator="\n")
    writer.writeheader();writer.writerows(scaling)
with (ROOT/"fixed_points.csv").open("w",newline="") as f:
    writer=csv.writer(f,lineterminator="\n")
    writer.writerow(["dt","K_low","K_high","B_low","B_high","analytic_infinite_energy","sweep_infidelity","ctm_chi8_energy","ctm_chi16_energy"])
    for row in follow["fixed_points"]:
        writer.writerow([row["dt"],row["K"][0],row["K"][2],row["b_low"],row["b_high"],row["infinite_energy"],row["sweep_infidelity"],
            row.get("chi8",{}).get("energy",""),row.get("chi16",{}).get("energy","")])

fig,axes=plt.subplots(1,2,figsize=(11,4))
for row in follow["trajectories"]:
    if row["dt"] not in (0.05,0.005):
        continue
    samples=row["samples"]
    label=f'dt={row["dt"]}, K(0)={row["initial"]}'
    axes[0].plot([s["tau"] for s in samples],[sum(s["predicted_K"])/4 for s in samples],label=label)
    axes[0].scatter([s["tau"] for s in samples],[sum(s["fitted_K"])/4 for s in samples],s=14)
axes[0].axhline(math.log(3)/4,color="black",linestyle=":",label="continuous-time false fixed point")
axes[0].set(xlabel="Cumulative imaginary time",ylabel="Mean fitted plaquette filter K",
    title="Lines: derived map; points: exact torus fit")
axes[0].legend(fontsize=7)
fixed=sorted(follow["fixed_points"],key=lambda r:r["dt"])
axes[1].plot([0]+[r["dt"] for r in fixed],[-0.75]+[r["infinite_energy"] for r in fixed],"o-",label="Derived stationary energy")
measured=[r for r in fixed if "chi16" in r]
axes[1].scatter([r["dt"] for r in measured],[r["chi16"]["energy"] for r in measured],marker="x",s=70,label="Infinite CTMRG, chi=16")
axes[1].axhline(-1,color="black",linestyle="--",label="Exact ground-state energy")
axes[1].set(xlabel="Time step dt",ylabel="Energy per edge spin",ylim=(-1.02,-0.72),title="Reducing dt does not recover the ground state")
axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=0.2)
fig.tight_layout()
stream=io.StringIO();fig.savefig(stream,format="svg")
(ROOT/"fixed_point_dynamics.svg").write_text("\n".join(x.rstrip() for x in stream.getvalue().splitlines())+"\n")
fig.savefig(ROOT/"fixed_point_dynamics.png",dpi=180)
print(json.dumps(summary,indent=2),flush=True)

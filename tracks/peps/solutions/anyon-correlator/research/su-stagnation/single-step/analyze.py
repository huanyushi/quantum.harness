"""Validate saved evidence and produce a small SVG; no new physics computation."""
from pathlib import Path
import tomllib
import csv

ROOT = Path(__file__).resolve().parent
with (ROOT / "results/audit.toml").open("rb") as f:
    data = tomllib.load(f)
with (ROOT / "results/stages.toml").open("rb") as f:
    stages = tomllib.load(f)

for branch in ("full", "plaq_only", "identity"):
    for row in data[branch]:
        if branch == "identity" or row["kind"] == "B":
            assert row["target_infidelity"] < 1e-12
        else:
            assert row["target_infidelity"] > 1e-4
            assert -sum(row["after"])/8 > -sum(row["exact"])/8 + 0.01
for kind in ("A", "B"):
    assert stages[kind]["no_trunc_infidelity"] < 1e-12
    assert stages[kind]["dense_gate_relative_error"] < 1e-12
astar=stages["A"]
assert astar["open_stabilizer_residual"] > 0.7
assert astar["stages"][-1]["open_optimal_scale_squared_error"] < astar["unchanged_open_optimal_scale_squared_error"]
assert astar["stages"][-1]["infidelity"] > 1e-3
for stage in astar["stages"]:
    assert abs(stage["infidelity"]-0.00162426251648) < 1e-12
for name in ("stalled", "full", "plaq_only", "identity", "isolated_A", "isolated_B"):
    a, b = (data[f"{name}_chi{chi}"] for chi in (8, 16))
    assert a["converged"] and b["converged"]
    assert abs(a["energy"]-b["energy"]) < 1e-10
    assert max(abs(x-y) for x,y in zip(a["stabilizers"], b["stabilizers"])) < 1e-10
    assert max(abs(x) for x in b["stabilizers"]) < 1+1e-10
assert abs(data["stalled_chi16"]["energy"]+0.753303645768674) < 1e-10
assert abs(data["full_chi16"]["energy"]-data["stalled_chi16"]["energy"]) < 1e-10
assert abs(data["identity_chi16"]["energy"]-data["stalled_chi16"]["energy"]) < 1e-10
assert data["isolated_A_chi16"]["energy"] > data["stalled_chi16"]["energy"]+0.009
assert data["plaq_only_chi16"]["energy"] < data["stalled_chi16"]["energy"]-0.03

with (ROOT/"step_summary.csv").open("w",newline="") as f:
    w=csv.writer(f, lineterminator="\n")
    w.writerow(["branch","step","gate","e_before_torus","e_exact_torus","e_after_torus","target_infidelity"])
    for branch in ("full","plaq_only","identity"):
        for i,r in enumerate(data[branch],1):
            w.writerow([branch,i,f'{r["kind"]}{r["r"]}{r["c"]}',-sum(r["before"])/8,-sum(r["exact"])/8,-sum(r["after"])/8,r["target_infidelity"]])

ys=[-sum(data["stalled_torus"])/8]+[-sum(r["after"])/8 for r in data["full"]]
def point(i,y):
    return 75+75*i, 240-(y+0.802)/0.021*180
poly=" ".join(f"{x:.2f},{y:.2f}" for x,y in (point(i,y) for i,y in enumerate(ys)))
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="330" viewBox="0 0 760 330">',
     '<rect width="760" height="330" fill="white"/>',
     '<g font-family="sans-serif" font-size="14" fill="#222">',
     '<text x="35" y="28">M2 stalled state: exact 2x2 torus, one SU sweep</text>',
     '<text x="35" y="49">Energy per edge spin (finite diagnostic, not infinite-plane energy)</text>']
for tick in (-0.800,-0.795,-0.790,-0.785):
    _,y=point(0,tick)
    svg += [f'<path d="M75 {y} H685" stroke="#ddd"/>',f'<text x="8" y="{y+5}">{tick:.3f}</text>']
svg += [f'<polyline points="{poly}" fill="none" stroke="#215caa" stroke-width="2.5"/>']
labels=["start","A11","B11","A12","B12","A21","B21","A22","B22"]
for i,(value,label) in enumerate(zip(ys,labels)):
    x,y=point(i,value)
    svg += [f'<circle cx="{x}" cy="{y}" r="4" fill="#215caa"/>',f'<text x="{x-15}" y="268">{label}</text>']
svg += ['<text x="75" y="300">Star compression raises energy; plaquette gates restore it.</text>','</g></svg>']
(ROOT/"gate_cycle.svg").write_text("\n".join(svg)+"\n")
print("Saved-data assertions passed; step_summary.csv and gate_cycle.svg written.",flush=True)

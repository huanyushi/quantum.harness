"""Plot recorded data only; never launches a calculation."""
from pathlib import Path
import csv
import tomllib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parent


def save_figure(fig, name):
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=170)
    plt.close(fig)
    print(output / f"{name}.svg", flush=True)


def tensor_setup():
    """Schematic only: the omitted path corner still exists in the full state."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), layout="constrained")

    def node(ax, xy, kind, faded=False):
        x, y = xy
        color = "#b7b7b7" if faded else ("#c66a17" if kind in ("V", "T") else "#2166ac")
        patch = (Rectangle((x-.15, y-.15), .30, .30, facecolor=color, zorder=4)
                 if kind in ("V", "T") else Circle((x, y), .15, facecolor=color, zorder=4))
        ax.add_patch(patch)
        ax.text(x, y, kind, color="white", ha="center", va="center", fontsize=10, zorder=5)

    def line(ax, p, q, outside=False, weight=True):
        ax.plot([p[0], q[0]], [p[1], q[1]], "--" if outside else "-",
                color="0.7" if outside else "0.25", lw=1.5, zorder=1)
        if weight and not outside:
            mid = (np.array(p)+np.array(q))/2
            ax.plot(*mid, marker="D", color="0.25", ms=4, zorder=2)

    ax = axes[0]
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        p = np.array([dx, dy], dtype=float)
        line(ax, (0, 0), p)
        line(ax, p, 1.55*p)
        physical_end = p + .43*np.array([-dy, dx])
        ax.plot([p[0], physical_end[0]], [p[1], physical_end[1]], color="#2166ac", lw=1.5)
        ax.text(*physical_end, "p", color="#2166ac", fontsize=10)
        node(ax, p, "P")
    node(ax, (0, 0), "V")
    ax.set_title("(a) Split V–λ–P star")
    ax.text(0, -1.95, "V: rank 4; P: rank 3\n◆: λ, D=2; p: d=2", ha="center", fontsize=9)
    ax.set(xlim=(-1.85, 1.85), ylim=(-2.2, 1.8))

    ax = axes[1]
    points = [(0, 0), (0, 1.5), (1.5, 0)]
    line(ax, points[0], points[1]); line(ax, points[0], points[2])
    for p in points:
        node(ax, p, "T")
        ax.plot([p[0], p[0]-.36], [p[1], p[1]+.36], color="#2166ac", lw=2.5)
    ax.text(.7, -.9, "T = block(V, P_E, P_N)\n3 T tensors; 2 updated bonds\nfused physical dimension 4", ha="center", fontsize=9)
    ax.set_title("(b) Original M2 plaquette")
    ax.set(xlim=(-1.0, 2.6), ylim=(-1.45, 2.55))

    ring = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for ax, omitted, title in ((axes[2], 4, "(c) Split path-SU plaquette"),
                                (axes[3], None, "(d) Loop/cluster plaquette")):
        for i, p in enumerate(ring):
            j = (i+1) % 8
            line(ax, p, ring[j], outside=(i == omitted or j == omitted))
        for i, p in enumerate(ring):
            node(ax, p, "V" if i % 2 == 0 else "P", faded=(i == omitted))
        label = ("7 tensors; 6 updated bonds\ngray V: outside this update\ncorner rotates between sweeps"
                 if omitted is not None else
                 "8 tensors; 8 updated bonds\nALS fits the closed ring\nexternal environment: diagonal λ")
        ax.text(1, -.8, label, ha="center", fontsize=9)
        ax.set_title(title)
        ax.set(xlim=(-.8, 2.8), ylim=(-1.45, 2.55))
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    fig.supxlabel("Schematics, not a finite lattice. Other external legs are omitted in (b–d); the full state always retains its loops.", fontsize=9)
    save_figure(fig, "tensor_setup")


def final_stabilizers(audit):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    x = np.arange(4)
    for key, label, color, shift in (
        ("legacy_replay", "Original M2", "#777777", -.23),
        ("path_random", "Split path-SU", "#c66a17", 0),
        ("loop_random", "Loop/cluster update", "#2166ac", .23),
    ):
        values = np.array(audit[key]["chi16"]["stabilizers"])
        for ax, vals in zip(axes, (values[::2], values[1::2])):
            ax.bar(x+shift, vals, width=.22, label=label, color=color)
    for ax, title in zip(axes, ("Star stabilizers", "Plaquette stabilizers")):
        ax.axhline(1, color="0.2", ls="--", lw=1)
        ax.set(xticks=x, xticklabels=["(1,1)", "(1,2)", "(2,1)", "(2,2)"],
               xlabel="Position in the 2×2 vertex cell", ylabel="Expectation value",
               title=title, ylim=(-.04, 1.12))
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=9)
    fig.supxlabel("Infinite lattice, fresh CTMRG χ=16. Loop maximum error: 4.43×10⁻¹⁰ (not resolved by bar height).", fontsize=9)
    save_figure(fig, "final_stabilizers")


def load(relative):
    with (ROOT / relative).open() as file:
        return list(csv.DictReader(file))


def main():
    path = load("results/pilot/random-seed1.csv")
    path += load("results/path-seed1/random-seed1.csv")[1:]
    loop = load("results/loop-seed1/random-seed1.csv")
    with (ROOT / "results/final-audit/audit.toml").open("rb") as file:
        audit = tomllib.load(file)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    cols = [f"{kind}{r}{c}" for r in (1, 2) for c in (1, 2) for kind in ("A", "B")]
    for rows, label, color in ((path, "Split path-SU", "#c66a17"),
                               (loop, "Split loop/cluster update", "#2166ac")):
        tau = np.array([float(row["tau"]) for row in rows])
        energy = np.array([float(row["energy_per_spin"]) for row in rows])
        error = np.array([max(abs(float(row[col]) - 1) for col in cols) for row in rows])
        valid = np.array([row["ctm_converged"] == "true" for row in rows])
        ax.plot(tau, np.where(valid, energy, np.nan), "o-", color=color, label=label, ms=4)
        bx.semilogy(tau, np.where(valid, error, np.nan), "o-", color=color, label=label, ms=4)
        if not valid.all():
            ax.plot(tau[~valid], energy[~valid], "x", color=color, ms=8)
    legacy = audit["legacy_replay"]["chi16"]["energy_per_spin"]
    ax.axhline(legacy, color="0.45", ls=":", label="Legacy merged-SU endpoint")
    ax.axhline(-1, color="0.2", ls="--", label="Exact ground state")
    bx.axhline(1e-6, color="0.45", ls="--", label="Acceptance tolerance")
    ax.set(xlabel=r"Imaginary time $\tau$", ylabel="Energy per edge spin", ylim=(-1.04, 0.04))
    bx.set(xlabel=r"Imaginary time $\tau$", ylabel=r"max $|\langle A_s,B_p\rangle-1|$")
    ax.legend(fontsize=8, loc="upper right")
    bx.legend(fontsize=8, loc="lower right")
    for a in (ax, bx):
        a.grid(alpha=0.2)
    fig.suptitle("Zero-field toric code: D=2, d=2 split V–λ–P network; random seed 1")
    fig.supxlabel("χ=8 trajectory; final loop state checked at χ=16. ×: unconverged CTMRG.", fontsize=9)
    save_figure(fig, "convergence")
    tensor_setup()
    final_stabilizers(audit)


if __name__ == "__main__":
    main()

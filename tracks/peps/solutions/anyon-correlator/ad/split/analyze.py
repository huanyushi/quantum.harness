"""Plot recorded optimization/audit evidence; never launches a physics run."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def save_svg(fig, path: Path) -> None:
    """Keep matplotlib's generated SVG paths free of trailing whitespace."""
    stream = io.StringIO()
    fig.savefig(stream, format="svg")
    path.write_text("\n".join(line.rstrip() for line in stream.getvalue().splitlines()) + "\n")


def boundary_tables(results: Path) -> None:
    rows = []
    for path in sorted(results.glob("boundary-*/boundary_results.json")):
        for r in json.loads(path.read_text()):
            m = r.get("measurement", {})
            info = r.get("info", [{}, {}])
            rows.append(dict(run=path.parent.name, label=r["label"], chi=r["chi"], seed=r["seed"],
                             status=r["status"], energy=m.get("energy"), mx=m.get("mx"), mz=m.get("mz"),
                             norm_down=info[0].get("normcell"), norm_up=info[1].get("normcell"),
                             norm_difference=r.get("norm_difference")))
    if rows:
        with (ROOT / "boundary_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    rows = []
    for path in sorted(results.glob("dominance-*/rows.json")):
        for r in json.loads(path.read_text()):
            m = r.get("measurement", {})
            rows.append(dict(run=path.parent.name, case=r["case"], chi=r["chi"], seed=r["seed"],
                             status=r["status"], normcell=r.get("normcell"), residual=r.get("residual"),
                             energy=m.get("energy"), mx=m.get("mx"), mz=m.get("mz")))
    if rows:
        with (ROOT / "dominance_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    rows = []
    for path in sorted(results.glob("perturb-preflight*/*/forward_audit.json")):
        for r in json.loads(path.read_text())["rows"]:
            m = r.get("measurement", {})
            rows.append(dict(run=str(path.parent.relative_to(results)), chi=r["chi"], seed=r["seed"],
                             status=r["status"], residual=r.get("residual"), energy=m.get("energy"),
                             mx=m.get("mx"), mz=m.get("mz")))
    if rows:
        with (ROOT / "perturb_forward.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def finite_field_plot(results: Path, figures: Path) -> None:
    """Explicitly selected unrolled continuation cohort, including any failures."""
    cohorts = sorted(results.glob("unrolled-repair*"))
    if not cohorts:
        return
    data = {}
    sources = {}
    # Latest available continuation per field, never selected by lowest energy.
    for cohort in cohorts:
        for label in ("x", "z", "xz"):
            path = cohort / label / "summary.json"
            if path.exists():
                data[label] = json.loads(path.read_text())
                sources[label] = str(path.relative_to(ROOT))
    if not data:
        return
    fig, (ae, am) = plt.subplots(1, 2, figsize=(10, 4))
    fields = [k for k in ("x", "z", "xz") if k in data]
    names = {"x": "(0.1, 0)", "z": "(0, 0.1)", "xz": "(0.1, 0.1)"}
    ae.bar(range(len(fields)), [data[k]["measurement"]["energy"] - data[k]["reference"]["energy"] for k in fields],
           color=["#176d49" if data[k]["audit"]["verdict"]["accepted"] else "#a05a55" for k in fields])
    ae.axhline(1e-5, color="black", linestyle=":", label="energy target")
    ae.set(xticks=range(len(fields)), xticklabels=[names[k] for k in fields],
           xlabel="(hx, hz)", ylabel="Energy minus series / edge spin")
    ae.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ae.legend(fontsize=8)
    components = [(k, m) for k, m in (("x", "mx"), ("z", "mz"), ("xz", "mx"), ("xz", "mz")) if k in data]
    errors = [100 * (data[k]["measurement"][m] / data[k]["reference"][m] - 1) for k, m in components]
    am.bar(range(len(components)), errors,
           color=["#176d49" if data[k]["audit"]["verdict"]["accepted"] else "#a05a55" for k, _ in components])
    am.axhline(1, color="black", linestyle=":")
    am.axhline(-1, color="black", linestyle=":")
    am.set(xticks=range(len(components)), xticklabels=[names[k] + " " + m for k, m in components],
           ylabel="Magnetization minus series (%)")
    am.tick_params(axis="x", labelrotation=20, labelsize=8)
    for ax in (ae, am):
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Finite-field V-lambda-P AD: green accepted / red rejected", fontsize=11)
    fig.tight_layout()
    save_svg(fig, figures / "finite_field_benchmark.svg")
    fig.savefig(figures / "finite_field_benchmark.png", dpi=180)
    (ROOT / "selected_fields.json").write_text(json.dumps(sources, indent=2)+"\n")


def main() -> None:
    results = ROOT / "results"
    boundary_tables(results)
    summaries = sorted(results.glob("**/summary.json"))
    if not summaries:
        print("No completed optimization/audit summaries; no AD result plot created.", flush=True)
        return
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for path in summaries:
        summary = json.loads(path.read_text())
        settings = json.loads((path.parent / "settings.json").read_text())
        reference = summary["reference"]
        measurement = summary["measurement"]
        accepted = summary["audit"]["verdict"]["accepted"]
        table.append(
            dict(
                run=str(path.parent.relative_to(results)),
                initialization=settings["initialization"],
                D=settings["D"],
                chi=settings["chi"],
                hx=settings["hx"],
                hz=settings["hz"],
                status=summary["status"],
                accepted=accepted,
                energy=measurement["energy"],
                energy_error=measurement["energy"] - reference["energy"],
                mx=measurement["mx"],
                mz=measurement["mz"],
                gradnorm=summary["gradnorm"],
                reasons=";".join(summary["audit"]["verdict"]["reasons"]),
            )
        )
        rows = [json.loads(line) for line in (path.parent / "trajectory.jsonl").read_text().splitlines()]
        rows = [row for row in rows if "gradnorm" in row]
        label = str(path.parent.relative_to(results)) + (" (accepted)" if accepted else " (rejected)")
        axes[0].plot([r["step"] for r in rows], [r["energy"] for r in rows], label=label)
        axes[1].semilogy([r["step"] for r in rows], [r["gradnorm"] for r in rows], label=label)
    axes[0].set(xlabel="AD step", ylabel="Energy per edge spin")
    axes[1].set(xlabel="AD step", ylabel="Recorded parameter-gradient norm")
    axes[1].axhline(1e-5, color="black", linestyle=":", label="stationarity threshold")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6)
    fig.tight_layout()
    save_svg(fig, figures / "ad_convergence.svg")
    fig.savefig(figures / "ad_convergence.png", dpi=160)
    fig2, (ax_e, ax_m) = plt.subplots(1, 2, figsize=(10, 4))
    labels = [r["run"] for r in table]
    colors = ["#176d49" if r["accepted"] else "#a05a55" for r in table]
    ax_e.bar(range(len(table)), [r["energy_error"] for r in table], color=colors)
    ax_e.set(ylabel="Energy minus series, per spin", xticks=range(len(labels)), xticklabels=labels)
    finite = [(i, p) for i, p in enumerate(summaries) if table[i]["hx"] or table[i]["hz"]]
    for i, path in finite:
        summary = json.loads(path.read_text())
        for offset, name, color in ((-0.15, "mx", "#355e91"), (0.15, "mz", "#b58521")):
            target = summary["reference"][name]
            if abs(target) > 1e-12:
                delta = 100 * (summary["measurement"][name] / target - 1)
                ax_m.bar(i + offset, delta, width=0.25, color=color)
    ax_m.axhline(1, color="gray", linestyle=":")
    ax_m.axhline(-1, color="gray", linestyle=":")
    ax_m.set(ylabel="Nonzero magnetization error (%)", xticks=range(len(labels)), xticklabels=labels)
    for ax in (ax_e, ax_m):
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    fig2.suptitle("Recorded candidates: green accepted, red rejected; magnetization blue X / gold Z", fontsize=9)
    fig2.tight_layout()
    save_svg(fig2, figures / "benchmark_errors.svg")
    fig2.savefig(figures / "benchmark_errors.png", dpi=160)
    finite_field_plot(results, figures)
    with (ROOT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved {ROOT / 'summary.csv'} and {figures / 'ad_convergence.svg'}", flush=True)


if __name__ == "__main__":
    main()

"""Summarize retained calculations without optimizing or recontracting a state."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tomllib


COLUMNS = (
    "method", "implementation", "case", "hx", "hz", "D", "chi",
    "initialization", "energy_per_spin", "mx", "mz", "status", "source",
)


def collect_results(root: Path) -> list[dict]:
    rows = []
    zero = root / "simple-update/split/final_measurements.toml"
    if zero.exists():
        data = tomllib.loads(zero.read_text())
        for key, implementation in (
            ("legacy_replay", "merged-L"), ("path_random", "split-path"),
            ("loop_random", "split-loop"),
        ):
            if key not in data:
                continue
            measurement = data[key]["chi16"]
            accepted = (
                measurement["converged"]
                and abs(measurement["energy_per_spin"] + 1) <= 1e-6
                and measurement.get("maxerr", 1) <= 1e-6
            )
            rows.append(dict(
                method="simple-update", implementation=implementation,
                case="zero", hx=0, hz=0, D=2, chi=16, initialization="random",
                energy_per_spin=measurement["energy_per_spin"], mx="", mz="",
                status="accepted" if accepted else "rejected",
                source=f"{zero.relative_to(root)}#{key}.chi16",
            ))

    loop = root / "simple-update/loop/benchmark_summary.csv"
    if loop.exists():
        with loop.open(newline="") as stream:
            for value in csv.DictReader(stream):
                description = value["status"]
                if "not accepted" in description or "error >1%" in description:
                    status = "rejected"
                elif description.startswith("provisional"):
                    status = "provisional"
                elif "within" in description:
                    status = "accepted_approximation"
                else:
                    raise ValueError(f"Unknown loop status: {description}")
                rows.append(dict(
                    method="simple-update", implementation="split-loop",
                    case=value["run"], hx=value["hx"], hz=value["hz"],
                    D=value["D"], chi=value["chi"],
                    initialization="random" if int(value["D"]) == 2 else "bond-growth",
                    energy_per_spin=value["energy"], mx=value["mx"], mz=value["mz"],
                    status=status,
                    source=f"{loop.relative_to(root)}#{value['run']}",
                ))

    ad = root / "ad/split"
    table = ad / "summary.csv"
    if table.exists():
        with table.open(newline="") as stream:
            values = {row["run"]: row for row in csv.DictReader(stream)}
        selected = json.loads((ad / "selected_fields.json").read_text())
        cases = [
            (label, str(Path(source).parent.relative_to("results")))
            for label, source in selected.items()
        ]
        cases.extend(
            ("fresh-" + run.rsplit("/", 1)[-1], run)
            for run in sorted(values) if run.startswith("fresh-unrolled-suite01/")
        )
        for label, run in cases:
            if run not in values:
                raise ValueError(f"Selected AD result is missing from summary.csv: {run}")
            value = values[run]
            accepted = value["accepted"].lower() == "true" and value["status"] == "stationary"
            rows.append(dict(
                method="AD", implementation="split-unrolled", case=label,
                hx=value["hx"], hz=value["hz"], D=value["D"], chi=value["chi"],
                initialization=value["initialization"],
                energy_per_spin=value["energy"], mx=value["mx"], mz=value["mz"],
                status="accepted" if accepted else "rejected",
                source=f"ad/split/summary.csv#{run}",
            ))

    merged = root / "ad/merged/final_measurements.csv"
    if merged.exists():
        with merged.open(newline="") as stream:
            values = list(csv.DictReader(stream))
        final = next(row for row in reversed(values) if int(row["step"]) == 86)
        rows.append(dict(
            method="AD", implementation="merged-tied", case="zero-step86",
            hx=0, hz=0, D=2, chi=final["chi"], initialization="random-continuation",
            energy_per_spin=final["E_per_spin"], mx="", mz="",
            status="accepted_zero_field_observables",
            source="ad/merged/final_measurements.csv#step86",
        ))
    if not rows:
        raise ValueError("No retained result tables found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = collect_results(args.root)
    output = args.out or args.root / "RESULTS.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} results: {output}", flush=True)


if __name__ == "__main__":
    main()

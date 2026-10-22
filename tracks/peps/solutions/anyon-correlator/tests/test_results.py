"""Exercise the result-table CLI on small, independently specified datasets."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_results.py"


class ResultTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def table(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def run_cli(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root)],
            text=True, capture_output=True, check=False,
        )

    def output(self):
        with (self.root / "RESULTS.csv").open() as stream:
            return list(csv.DictReader(stream))

    def ad_input(self, *, accepted):
        self.table("ad/split/summary.csv", [dict(
            run="chosen/z", initialization="analytic-anchor-uniform", D=2,
            chi=8, hx=0, hz=0.1, status="stationary" if accepted else "armijo_failed",
            accepted=accepted, energy=-1.0025, energy_error=0.000024,
            mx=0, mz=0.05, gradnorm=1e-7, reasons="" if accepted else "optimizer_not_stationary",
        )])
        (self.root / "ad/split/selected_fields.json").write_text(
            json.dumps({"z": "results/chosen/z/summary.json"})
        )

    def test_ad_keeps_analytic_initialization_separate_from_random_search(self):
        self.ad_input(accepted=True)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        row, = self.output()
        self.assertEqual(row["method"], "AD")
        self.assertEqual(row["initialization"], "analytic-anchor-uniform")
        self.assertEqual(row["status"], "accepted")
        self.assertEqual(float(row["energy_per_spin"]), -1.0025)

    def test_selected_but_failed_ad_candidate_is_not_reported_as_accepted(self):
        self.ad_input(accepted=False)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        row, = self.output()
        self.assertEqual(row["status"], "rejected")

    def test_loop_d4_remains_rejected_despite_a_converged_environment(self):
        self.table("simple-update/loop/benchmark_summary.csv", [dict(
            run="d4-xz", D=4, hx=0.1, hz=0.1,
            status="not accepted; unfinished and branch-sensitive",
            selected_source="results/d4-xz/sequential.json[0]",
            energy=-1.004887, mx=0.0532, mz=0.04996, chi=8,
            ctm_converged=True,
        )])
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        row, = self.output()
        self.assertEqual(row["method"], "simple-update")
        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["initialization"], "bond-growth")

    def test_zero_field_path_and_loop_keep_different_statuses(self):
        path = self.root / "simple-update/split/final_measurements.toml"
        path.parent.mkdir(parents=True)
        path.write_text("""[path_random]
step = 600
[path_random.chi16]
energy_per_spin = -0.53
maxerr = 1.0
converged = true
[loop_random]
step = 120
[loop_random.chi16]
energy_per_spin = -0.9999999999
maxerr = 0.0000000004
converged = true
""")
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = {row["implementation"]: row for row in self.output()}
        self.assertEqual(rows["split-path"]["status"], "rejected")
        self.assertEqual(rows["split-loop"]["status"], "accepted")

    def test_missing_selected_run_fails_without_writing_a_misleading_table(self):
        self.ad_input(accepted=True)
        (self.root / "ad/split/selected_fields.json").write_text(
            json.dumps({"z": "results/missing/z/summary.json"})
        )
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing/z", result.stderr)
        self.assertFalse((self.root / "RESULTS.csv").exists())


if __name__ == "__main__":
    unittest.main()

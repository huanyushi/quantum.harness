import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from plot import render


class PlotChecks(unittest.TestCase):
    def test_zero_unresolved_and_failed_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "spectrum.csv"
            rows = [
                dict(g=0, q=0, k=-3.14, rho=0.5, quasi_energy=0.693, status="converged"),
                dict(g=1, q=1, k=0, rho=0, quasi_energy="", status="zero_mode"),
                dict(g=1, q=0, k=1.57, rho=1e-15, quasi_energy="", status="below_resolution"),
                dict(g=0, q=1, k=2, rho=0.9, quasi_energy=0.1, status="not_converged"),
                dict(g=0, q=1, k=2.5, rho="", quasi_energy="", status="empty_sector"),
            ]
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            paths = render(source, root / "figures")
            self.assertEqual(len(paths), 2)
            for path in paths:
                tree = ET.parse(path)
                modes = [el for el in tree.iter() if "data-status" in el.attrib]
                self.assertEqual(len(modes), 3)
                self.assertEqual({m.attrib["data-status"] for m in modes},
                                 {"converged", "zero_mode", "below_resolution"})
                self.assertEqual({m.attrib["data-g"] for m in modes}, {"0", "1"})
                self.assertEqual({m.attrib["data-q"] for m in modes}, {"0", "1"})
                unresolved = next(m for m in modes if m.attrib["data-status"] == "below_resolution")
                self.assertTrue(any(el.attrib.get("class") == "resolution" for el in unresolved))
            energy = ET.parse(paths[1])
            zero = next(el for el in energy.iter() if el.attrib.get("data-status") == "zero_mode")
            self.assertIn("∞", "".join(zero.itertext()))
            self.assertNotIn("36", "".join(zero.itertext()))


if __name__ == "__main__":
    unittest.main()

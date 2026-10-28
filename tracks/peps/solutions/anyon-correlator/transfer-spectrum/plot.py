"""Read saved transfer spectra and produce SVG; never launches numerical work."""
import argparse
import csv
import html
import math
from pathlib import Path

COLORS = {0: "#2563eb", 1: "#dc2626"}
VALID = {"converged", "zero_mode", "below_resolution"}


def _number(value):
    try:
        x = float(value)
    except (ValueError, TypeError):
        return None
    return x if math.isfinite(x) else None


def _marker(x, y, q, color):
    if q == 1:
        return f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="none" stroke="{color}"/>'
    return (f'<path d="M{x-3:.3f},{y:.3f}h6 M{x:.3f},{y-3:.3f}v6" '
            f'fill="none" stroke="{color}"/>')


def render(source, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with Path(source).open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["status"] in VALID]
    paths = []
    for quantity, title, filename in (
        ("rho", "Normalized transfer eigenvalue magnitude |r(k)|", "rho.svg"),
        ("quasi_energy", "Transfer spectrum -log|r(k)| (not Hamiltonian energy)", "epsilon.svg"),
    ):
        values = [_number(r.get(quantity)) for r in rows if r["status"] == "converged"]
        maximum = max([v for v in values if v is not None] + [1.0]) * 1.1
        def xpos(k):
            return 75 + (k + math.pi) / (2 * math.pi) * 620
        def ypos(v):
            return 355 - 250 * v / maximum
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="780" height="450" viewBox="0 0 780 450">',
               '<rect width="780" height="450" fill="white"/>',
               f'<text x="35" y="25" font-size="16">{html.escape(title)}</text>',
               '<path d="M75 100V355H695" fill="none" stroke="black"/>',
               '<text x="70" y="378">−π</text><text x="380" y="378">0</text><text x="685" y="378">π</text>',
               '<text x="375" y="400">momentum k</text>',
               '<text x="35" y="420" font-size="12">Blue/red: g=0/1; +/circle: q=0/1. Dotted ring: below resolution. Failed/empty modes omitted.</text>']
        for tick in range(5):
            v = maximum * tick / 4
            y = ypos(v)
            svg.append(f'<text x="8" y="{y+4:.3f}" font-size="11">{v:.3g}</text>')
        for row in rows:
            k, rho = _number(row.get("k")), _number(row.get("rho"))
            if k is None or rho is None:
                continue
            g, q = int(row["g"]), int(row["q"])
            status = row["status"]
            x = xpos(k)
            value = _number(row.get(quantity))
            if quantity == "quasi_energy" and status != "converged":
                y = 56 + 12 * (2 * g + q)
                label = "∞" if status == "zero_mode" else "?"
            elif value is not None:
                y, label = ypos(value), ""
            else:
                continue
            svg.append(f'<g data-g="{g}" data-q="{q}" data-status="{status}" stroke-width="1.3">')
            svg.append(_marker(x, y, q, COLORS[g]))
            if status == "below_resolution":
                svg.append(f'<circle class="resolution" cx="{x:.3f}" cy="{y:.3f}" r="5" fill="none" stroke="{COLORS[g]}" stroke-dasharray="1 2"/>')
            if label:
                svg.append(f'<text x="{x+4:.3f}" y="{y-2:.3f}" fill="{COLORS[g]}" font-size="10">{label}</text>')
            svg.append(f'<title>g={g}, q={q}, k={k:.8g}, status={status}, |r|={rho:.6g}</title></g>')
        svg.append('</svg>')
        path = out / filename
        path.write_text("\n".join(svg), encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render(args.input, args.out)

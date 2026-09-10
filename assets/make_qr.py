#!/usr/bin/env python3
# Build static QR-code SVGs (Tübingen-AI dark-blue) for the margin "side info" cards.
# Offline: uses the local `qrcode` library, no external QR service. Re-run: python3 make_qr.py
import os, qrcode

DARK = "#1a3a5b"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qr")
os.makedirs(OUT, exist_ok=True)

# (card label, URL encoded in the QR, output slug) — add more lines as needed
PACKAGES = [
    ("ProbNum 2026 tutorial",         "https://github.com/timweiland/probnum26-tutorial",            "tutorial"),
    ("FunctionalGPs.jl",               "https://github.com/timweiland/FunctionalGPs.jl",               "functionalgps"),
    ("GaussianMarkovRandomFields.jl",  "https://github.com/timweiland/GaussianMarkovRandomFields.jl",  "gmrfs"),
    ("Latte.jl",                       "https://github.com/timweiland/Latte.jl",                       "latte"),
    ("linpde-gp",                      "https://github.com/marvinpfoertner/linpde-gp",                 "linpdegp"),
    ("Pluto.jl",                       "https://github.com/fonsp/Pluto.jl",                            "pluto"),
]

def svg_for(url):
    qr = qrcode.QRCode(border=4, box_size=1)      # border=4 = spec quiet zone (reliable scanning)
    qr.add_data(url); qr.make(fit=True)
    m = qr.get_matrix(); n = len(m)
    rects = "".join(f'<rect x="{j}" y="{i}" width="1" height="1"/>'
                    for i, row in enumerate(m) for j, v in enumerate(row) if v)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" '
            f'width="100%" shape-rendering="crispEdges">'
            f'<rect width="100%" height="100%" fill="white"/>'
            f'<g fill="{DARK}">{rects}</g></svg>')

for name, url, slug in PACKAGES:
    with open(os.path.join(OUT, slug + ".svg"), "w") as f:
        f.write(svg_for(url))
    print(f"wrote qr/{slug}.svg   ({name} -> {url})")

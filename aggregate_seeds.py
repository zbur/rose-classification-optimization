"""Average results.json files from several seeds (mean +/- std per variant).

Example:
    python aggregate_seeds.py results/s0 results/s1 results/s2 --out results/summary.md
"""
import argparse
import json
import os

import numpy as np

from config import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = []
    for d in args.dirs:
        with open(os.path.join(d, "results.json")) as f:
            runs.append(json.load(f))

    fields = {"macro_f1": lambda r: r["metrics"]["macro_f1"],
              "delta_pp": lambda r: r.get("macro_f1_delta_pp", np.nan)}
    fields.update({f"recall_{c}": (lambda c: lambda r: r["metrics"]["per_class"][c]["recall"])(c)
                   for c in CLASSES})
    fields.update({"median_ms": lambda r: r["benchmark"]["inference"]["median_ms"] if r["benchmark"] else np.nan,
                   "p95_ms": lambda r: r["benchmark"]["inference"]["p95_ms"] if r["benchmark"] else np.nan})

    variants = sorted(runs[0]["variants"])
    lines = [f"Seeds aggregated: {len(runs)}\n",
             "| variant | " + " | ".join(fields) + " |", "|---|" + "---|" * len(fields)]
    for v in variants:
        cells = []
        for fn in fields.values():
            vals = np.array([fn(r["variants"][v]) for r in runs if v in r["variants"]], dtype=float)
            cells.append(f"{np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()

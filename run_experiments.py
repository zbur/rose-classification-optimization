"""Build, evaluate and benchmark variants A-E, then apply the quality gate (slide 5).

  A  PyTorch       FP32  224   fine-tuned baseline
  B  ONNX Runtime  FP32  224   B vs A: runtime / export effect
  C  ONNX Runtime  INT8  224   C vs B: quantization effect
  D  ONNX Runtime  FP32  160   D vs B: resolution effect
  E  ONNX Runtime  INT8  160   E vs C: resolution effect (INT8), E vs D: INT8 at 160

Resolution variants use --ckpt-160 (matched fine-tuning at 160) if given,
otherwise the shared 224 checkpoint evaluated at 160; the choice is recorded.
Among variants whose Macro-F1 drops <= --gate-pp points vs. A, the one with
the lowest median latency is selected.

Example:
    python run_experiments.py --ckpt-224 runs/fp32_224_s0/best.pt \
        --ckpt-160 runs/fp32_160_s0/best.pt --split splits/rose_split.json --out results/s0
"""
import argparse
import csv
import json
import os

from benchmark import run_benchmark
from config import CLASSES, GATE_MACRO_F1_PP, VARIANTS
from evaluate import evaluate
from export_onnx import export
from inference import make_runner
from preprocess import items_for, load_split
from quantize_int8 import quantize


def build_models(args, model_dir):
    """Return {variant: {"model": path, "size_file": path, "source": str}}."""
    ckpt_160 = args.ckpt_160 or args.ckpt_224
    weights_224 = os.path.join(os.path.dirname(args.ckpt_224), "weights_fp32.pt")
    paths = {
        "A": {"model": args.ckpt_224, "source": "fine-tuned checkpoint (224)",
              "size_file": weights_224 if os.path.exists(weights_224) else args.ckpt_224},
        "B": {"model": os.path.join(model_dir, "B_fp32_224.onnx"), "source": "export of A"},
        "C": {"model": args.int8_224 or os.path.join(model_dir, "C_int8_224.onnx"),
              "source": "user-supplied INT8 (e.g. QAT)" if args.int8_224 else "PTQ static INT8 of B"},
        "D": {"model": os.path.join(model_dir, "D_fp32_160.onnx"),
              "source": "matched fine-tuning at 160" if args.ckpt_160 else "shared 224 checkpoint at 160"},
        "E": {"model": args.int8_160 or os.path.join(model_dir, "E_int8_160.onnx"),
              "source": "user-supplied INT8 (e.g. QAT)" if args.int8_160 else "PTQ static INT8 of D"},
    }
    if args.rebuild or not os.path.exists(paths["B"]["model"]):
        export(args.ckpt_224, 224, paths["B"]["model"], args.split)
    if args.rebuild or not os.path.exists(paths["D"]["model"]):
        export(ckpt_160, 160, paths["D"]["model"], args.split)
    for q, f in (("C", "B"), ("E", "D")):
        user_supplied = args.int8_224 if q == "C" else args.int8_160
        if not user_supplied and (args.rebuild or not os.path.exists(paths[q]["model"])):
            quantize(paths[f]["model"], args.split, paths[q]["model"], args.num_calib,
                     args.calib_method, seed=args.seed)
    for p in paths.values():
        p.setdefault("size_file", p["model"])
    return paths


def pp(a, b):
    return 100.0 * (a - b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-224", required=True)
    ap.add_argument("--ckpt-160", default=None)
    ap.add_argument("--int8-224", default=None, help="Use this INT8 ONNX (e.g. from qat.py) for C.")
    ap.add_argument("--int8-160", default=None, help="Use this INT8 ONNX (e.g. from qat.py) for E.")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", choices=["val", "test"], default="val",
                    help="Use val while iterating; switch to test once for the final locked run.")
    ap.add_argument("--variants", default="ABCDE")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--bench-images", type=int, default=50)
    ap.add_argument("--num-calib", type=int, default=320)
    ap.add_argument("--calib-method", choices=["minmax", "entropy", "percentile"], default="minmax")
    ap.add_argument("--gate-pp", type=float, default=GATE_MACRO_F1_PP)
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="Re-export / re-quantize even if files exist.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model_dir = os.path.join(args.out, "models")
    split = load_split(args.split)
    items = items_for(split, args.subset)
    paths = build_models(args, model_dir)

    results = {}
    for v in args.variants:
        spec = VARIANTS[v]
        print(f"\n=== Variant {v}: {spec['runtime']} / {spec['precision']} / {spec['img_size']} ===")
        runner = make_runner(paths[v]["model"], threads=0)
        metrics = evaluate(runner, items, spec["img_size"], split["classes"])
        del runner
        print(f"macro-F1 {metrics['macro_f1']:.4f}  " +
              "  ".join(f"{k}:{x['recall']:.3f}" for k, x in metrics["per_class"].items()))
        bench = None
        if not args.skip_benchmark:
            bench = run_benchmark(paths[v]["model"], args.split, spec["img_size"], args.threads,
                                  args.warmup, args.runs, args.bench_images, args.subset,
                                  paths[v]["size_file"])
            print(f"latency median {bench['inference']['median_ms']:.2f} ms  "
                  f"p95 {bench['inference']['p95_ms']:.2f} ms  size {bench['model_size_mb']:.2f} MB  "
                  f"peak mem {bench['peak_inference_memory_mb']:.1f} MB")
        results[v] = {**spec, **paths[v], "metrics": metrics, "benchmark": bench}

    # Quality gate and selection.
    if "A" in results:
        base = results["A"]["metrics"]["macro_f1"]
        for r in results.values():
            r["macro_f1_delta_pp"] = pp(r["metrics"]["macro_f1"], base)
            r["passes_gate"] = r["macro_f1_delta_pp"] >= -args.gate_pp
    passing = [v for v, r in results.items() if r.get("passes_gate") and r["benchmark"]]
    selected = min(passing, key=lambda v: results[v]["benchmark"]["inference"]["median_ms"]) if passing else None

    comparisons = {}
    for name, (x, y) in {"runtime (B vs A)": ("B", "A"), "quantization (C vs B)": ("C", "B"),
                         "resolution FP32 (D vs B)": ("D", "B"), "resolution INT8 (E vs C)": ("E", "C"),
                         "quantization at 160 (E vs D)": ("E", "D")}.items():
        if x in results and y in results:
            c = {"macro_f1_delta_pp": pp(results[x]["metrics"]["macro_f1"], results[y]["metrics"]["macro_f1"])}
            if results[x]["benchmark"] and results[y]["benchmark"]:
                bx, by = results[x]["benchmark"], results[y]["benchmark"]
                c["median_latency_ratio"] = bx["inference"]["median_ms"] / by["inference"]["median_ms"]
                c["size_ratio"] = bx["model_size_mb"] / by["model_size_mb"]
            comparisons[name] = c

    summary = {"subset": args.subset, "gate_pp": args.gate_pp, "selected": selected,
               "comparisons": comparisons, "variants": results,
               "resolution_checkpoint": "matched 160" if args.ckpt_160 else "shared 224"}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    write_tables(results, selected, args)
    print(f"\nSelected variant: {selected}  (results in {args.out}/results.md)")


def write_tables(results, selected, args):
    header = ["variant", "runtime", "precision", "img_size", "macro_f1", "delta_pp_vs_A", "gate"]
    header += [f"recall_{c}" for c in CLASSES]
    header += ["median_ms", "p95_ms", "preprocess_median_ms", "size_mb", "peak_mem_mb"]
    rows = []
    for v, r in results.items():
        b = r["benchmark"]
        rows.append([v, r["runtime"], r["precision"], r["img_size"], f"{r['metrics']['macro_f1']:.4f}",
                     f"{r.get('macro_f1_delta_pp', float('nan')):+.2f}",
                     "pass" if r.get("passes_gate") else "fail"]
                    + [f"{r['metrics']['per_class'][c]['recall']:.3f}" for c in CLASSES]
                    + ([f"{b['inference']['median_ms']:.2f}", f"{b['inference']['p95_ms']:.2f}",
                        f"{b['preprocess']['median_ms']:.2f}", f"{b['model_size_mb']:.2f}",
                        f"{b['peak_inference_memory_mb']:.1f}"] if b else ["-"] * 5))
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(f"Subset: **{args.subset}**, threads={args.threads}, batch=1, "
                f"gate: Macro-F1 loss <= {args.gate_pp} pp vs. A\n\n")
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")
        f.write(f"\nSelected (fastest passing): **{selected}**\n")


if __name__ == "__main__":
    main()

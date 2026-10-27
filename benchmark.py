"""Latency / size / peak-memory benchmark for one model (slide 5 protocol).

Fixed settings: CPU, fixed thread count, batch size 1, warmup + repeated runs.
Preprocessing latency is reported separately from inference latency.
Each measurement runs in a fresh subprocess so peak memory is not shared
between variants.

Example:
    python benchmark.py --model models/E_int8_160.onnx --split splits/rose_split.json --threads 1
"""
import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

from preprocess import items_for, load_rgb, load_split, preprocess_pil


def max_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def summarize_ms(samples_ns):
    a = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "std_ms": float(a.std()), "min_ms": float(a.min()),
            "n": int(a.size)}


def worker(args):
    split = load_split(args.split)
    items = items_for(split, args.subset)[:args.num_images]

    # 1) Preprocessing latency (decode + resize + crop + normalize), measured separately.
    pre_ns, inputs = [], []
    for it in items:
        t0 = time.perf_counter_ns()
        x = preprocess_pil(load_rgb(it["abs_path"]), args.img_size)
        pre_ns.append(time.perf_counter_ns() - t0)
        inputs.append(np.ascontiguousarray(x[None], dtype=np.float32))

    # 2) Import runtime, then record memory before the model is loaded.
    is_onnx = args.model.endswith(".onnx")
    if is_onnx:
        import onnxruntime  # noqa: F401
    else:
        import torch  # noqa: F401
    from inference import make_runner
    rss_before = max_rss_mb()

    t0 = time.perf_counter_ns()
    runner = make_runner(args.model, threads=args.threads)
    load_ms = (time.perf_counter_ns() - t0) / 1e6

    # 3) Warmup + timed batch-1 inference.
    for i in range(args.warmup):
        runner(inputs[i % len(inputs)])
    inf_ns = []
    for i in range(args.runs):
        x = inputs[i % len(inputs)]
        t0 = time.perf_counter_ns()
        runner(x)
        inf_ns.append(time.perf_counter_ns() - t0)
    rss_peak = max_rss_mb()

    size_path = args.size_file or args.model
    return {
        "model": args.model, "runtime": runner.runtime, "img_size": args.img_size,
        "threads": args.threads, "batch_size": 1, "warmup": args.warmup, "runs": args.runs,
        "inference": summarize_ms(inf_ns), "preprocess": summarize_ms(pre_ns),
        "model_load_ms": load_ms,
        "model_file": size_path, "model_size_mb": os.path.getsize(size_path) / 2**20,
        "peak_rss_mb": rss_peak, "rss_before_model_mb": rss_before,
        "peak_inference_memory_mb": rss_peak - rss_before,
        "platform": {"machine": platform.machine(), "processor": platform.processor(),
                     "system": platform.platform(), "python": platform.python_version()},
    }


def run_benchmark(model, split, img_size, threads=1, warmup=20, runs=200, num_images=50,
                  subset="test", size_file=None):
    """Run one benchmark in a fresh Python process and return its JSON result."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = tmp.name
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--model", model, "--split", split,
           "--img-size", str(img_size), "--threads", str(threads), "--warmup", str(warmup),
           "--runs", str(runs), "--num-images", str(num_images), "--subset", subset, "--out", out]
    if size_file:
        cmd += ["--size-file", size_file]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    subprocess.run(cmd, check=True, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
    with open(out) as f:
        result = json.load(f)
    os.remove(out)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--img-size", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--num-images", type=int, default=50)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--size-file", default=None,
                    help="File used for model size (e.g. weights_fp32.pt for a PyTorch checkpoint).")
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        result = worker(args)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        return

    result = run_benchmark(args.model, args.split, args.img_size, args.threads, args.warmup,
                           args.runs, args.num_images, args.subset, args.size_file)
    inf, pre = result["inference"], result["preprocess"]
    print(f"{args.model}: inference median {inf['median_ms']:.2f} ms / p95 {inf['p95_ms']:.2f} ms | "
          f"preprocess median {pre['median_ms']:.2f} ms | size {result['model_size_mb']:.2f} MB | "
          f"peak inference mem {result['peak_inference_memory_mb']:.1f} MB")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()

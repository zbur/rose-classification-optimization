"""Uniform runners for PyTorch checkpoints and ONNX Runtime models.

Both take a preprocessed float32 batch (N, 3, H, W) and return logits (N, C).
Runtimes are imported lazily so a benchmark worker only loads what it needs.
"""
import os

import numpy as np


class OnnxRunner:
    runtime = "onnxruntime"

    def __init__(self, path, threads=1, providers=None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, so, providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.img_size = inp.shape[2] if isinstance(inp.shape[2], int) else None
        self.path = path

    def __call__(self, x):
        return self.session.run(None, {self.input_name: np.ascontiguousarray(x, dtype=np.float32)})[0]


class TorchRunner:
    runtime = "pytorch"

    def __init__(self, path, threads=1, device="cpu"):
        import torch

        from model import load_checkpoint

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.model, meta = load_checkpoint(path, map_location=device)
        self.model.to(device).eval()
        self.device = device
        self.img_size = meta.get("img_size")
        self.path = path

    def __call__(self, x):
        with self.torch.inference_mode():
            out = self.model(self.torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device))
        return out.cpu().numpy()


def make_runner(path, threads=1):
    if os.path.splitext(path)[1].lower() == ".onnx":
        return OnnxRunner(path, threads=threads)
    return TorchRunner(path, threads=threads)

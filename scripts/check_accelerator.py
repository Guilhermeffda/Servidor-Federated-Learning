#!/usr/bin/env python3
"""Real accelerator forward/BatchNorm/backward preflight (no CPU fallback)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fl.runtime import backend_compatibility, get_accelerator_info


def check(device="0", raw=False):
    info = get_accelerator_info(device)
    info["ultralytics"] = version("ultralytics")
    print(json.dumps(info, indent=2), flush=True)
    target = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if target.type == "cpu":
        raise ValueError("Accelerator check requires a GPU; CPU fallback is forbidden")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU is unavailable")
    torch.manual_seed(42)
    x = torch.randn(4, 3, 32, 32, device=target, requires_grad=True)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.BatchNorm2d(8), torch.nn.SiLU(),
    ).to(target).train()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    started = time.perf_counter()
    with nullcontext() if raw else backend_compatibility(device):
        output = model(x)
        loss = output.square().mean()
        loss.backward()
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        elif target.type == "mps":
            torch.mps.synchronize()
    if output.device != x.device or not torch.isfinite(loss):
        raise AssertionError("Output device mismatch or nonfinite loss")
    for name, parameter in [("input", x), *model.named_parameters()]:
        gradient = parameter.grad
        if gradient is None or gradient.device != x.device or not torch.isfinite(gradient).all():
            raise AssertionError(f"Missing/nonfinite/off-device gradient: {name}")
    result = {**info, "raw": raw, "loss": loss.detach().item(),
              "elapsed_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None}
    print(json.dumps(result, indent=2), flush=True)
    print("ACCELERATOR CHECK: OK", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--raw", action="store_true", help="Reproduce without compatibility")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.device, args.raw)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

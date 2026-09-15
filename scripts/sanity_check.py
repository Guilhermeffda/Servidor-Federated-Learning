#!/usr/bin/env python3
"""Mede o custo por epoca do YOLO numa particao real do TACO."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl.partition import count_split_images

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "partitions" / "iid" / "client_0" / "data.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "hardware_benchmark" / "measurement.json"
TOTAL_FEDERATED_EPOCHS = 5 * 50 * 5 * 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(
            f"Particao real ausente: {data}. Rode taco_data/scripts/prepare_experiments.py."
        )
    if args.epochs <= 0:
        raise ValueError("epochs deve ser positivo")
    train_images = count_split_images(data, "train")
    model = YOLO(str(ROOT / "yolov8n.pt"))
    started = time.perf_counter()
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(ROOT / "runs" / "hardware_benchmark"),
        name="taco_iid_client_0",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    seconds_per_epoch = elapsed / args.epochs
    total_seconds = seconds_per_epoch * TOTAL_FEDERATED_EPOCHS
    decision = "colab" if total_seconds > 24 * 3600 else "cpu_local"
    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "data_yaml": str(data.relative_to(ROOT)),
        "train_images": train_images,
        "epochs": args.epochs,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "device": args.device,
        "elapsed_seconds": elapsed,
        "seconds_per_epoch": seconds_per_epoch,
        "estimated_federated_epochs": TOTAL_FEDERATED_EPOCHS,
        "estimated_total_hours": total_seconds / 3600,
        "estimated_total_days": total_seconds / 86400,
        "decision": decision,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

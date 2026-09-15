#!/usr/bin/env python3
"""Treina e avalia a baseline YOLOv8n centralizada no TACO-10."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "centralized" / "data.yaml"
RESULTS = ROOT / "results" / "baseline_centralized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def image_count(path: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(item.suffix.lower() in extensions for item in path.rglob("*"))


def dataset_counts(data_yaml: Path) -> dict[str, int]:
    document = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(document["path"])
    return {
        split: image_count(root / document[split])
        for split in ("train", "val", "test")
    }


def test_ids_hash() -> str:
    path = ROOT / "taco_data" / "data" / "test_global" / "test_image_ids.txt"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(
            f"Dataset centralizado ausente: {data}. Rode taco_data/scripts/prepare_experiments.py."
        )
    counts = dataset_counts(data)
    if counts != {"train": 960, "val": 240, "test": 300}:
        raise ValueError(f"Split inesperado: {counts}")
    print(f"Dataset validado: {counts}; teste SHA-256={test_ids_hash()}")
    if args.dry_run:
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    status_path = RESULTS / "training_status.json"
    started_at = datetime.now(timezone.utc)
    status_path.write_text(
        json.dumps({"status": "running", "started_at": started_at.isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    try:
        if args.resume:
            model = YOLO(str(args.resume.resolve()))
            training = model.train(resume=True)
        else:
            model = YOLO(str(ROOT / "yolov8n.pt"))
            training = model.train(
                data=str(data),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=2,
                seed=42,
                deterministic=True,
                project=str(ROOT / "runs" / "centralized_baseline"),
                name="yolov8n_taco10",
                exist_ok=True,
                plots=True,
            )
        best = Path(model.trainer.best)
        best_model = YOLO(str(best))
        evaluation = best_model.val(
            data=str(data), split="test", imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=2, plots=True,
            project=str(ROOT / "runs" / "centralized_baseline"), name="test_global",
        )
        precision = float(evaluation.box.mp)
        recall = float(evaluation.box.mr)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        shutil.copy2(best, RESULTS / "best.pt")
        training_csv = Path(model.trainer.csv)
        if training_csv.is_file():
            shutil.copy2(training_csv, RESULTS / "training_curve.csv")
        metrics = {
            "status": "complete",
            "scientific_valid": True,
            "dataset": "TACO-10",
            "split_counts": counts,
            "test_image_ids_sha256": test_ids_hash(),
            "epochs": args.epochs,
            "image_size": args.imgsz,
            "batch_size": args.batch,
            "device": args.device,
            "map50": float(evaluation.box.map50),
            "map50_95": float(evaluation.box.map),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "duration_seconds": time.perf_counter() - started,
            "best_checkpoint": str(best),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
        }
        (RESULTS / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        status_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    except Exception as exc:
        status_path.write_text(
            json.dumps({"status": "failed", "error": repr(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()

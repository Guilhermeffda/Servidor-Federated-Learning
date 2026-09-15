#!/usr/bin/env python3
"""Teste isolado do FLClient numa particao TACO real."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fl.client import FLClient
from fl.dataset import resolve_client_dataset
from fl.model import get_weights, set_weights


def main() -> None:
    data_yaml = resolve_client_dataset("client_0", "iid")
    client = FLClient(
        "client_0",
        data_yaml,
        num_train_examples=192,
        model_weights=str(ROOT / "yolov8n.pt"),
    )
    before = get_weights(client.model)
    set_weights(client.model, before)
    after = get_weights(client.model)
    roundtrip_exact = len(before) == len(after) and all(
        np.array_equal(left, right) for left, right in zip(before, after)
    )
    if not roundtrip_exact:
        raise AssertionError("Round-trip dos pesos nao foi exato")

    output_dir = ROOT / "results" / "client_taco_smoke"
    local_runs = output_dir / "local_runs"
    updated, count, metrics = client.fit(
        before,
        {
            "local_epochs": 1,
            "batch_size": 16,
            "image_size": 640,
            "device": "cpu",
            "seed": 42,
            "mu": 0.0,
            "output_dir": str(local_runs),
            "run_name": "client_0_mu_0",
        },
    )
    result = {
        "status": "complete",
        "dataset": "TACO-10 real",
        "data_yaml": str(Path(data_yaml).resolve().relative_to(ROOT)),
        "client_id": "client_0",
        "num_examples": count,
        "num_parameter_tensors": len(updated),
        "weight_roundtrip_exact": roundtrip_exact,
        "dropout_active": False,
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    shutil.rmtree(local_runs, ignore_errors=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Valida os artefatos exigidos pelo DoD da baseline FedAvg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results" / "baseline"
REQUIRED_ROUND_COLUMNS = {
    "round", "loss", "map50", "map50_95", "precision", "recall",
    "num_examples", "elapsed_seconds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = args.results.resolve()
    failures: list[str] = []
    completed = 0
    image_sizes = set()
    for scenario in ("iid", "non_iid"):
        for repetition in (1, 2, 3):
            run_id = f"{scenario}_rep{repetition}"
            run_dir = results / run_id
            try:
                if not (run_dir / "config.json").is_file():
                    raise FileNotFoundError("artefato ausente: config.json")
                status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                if config.get("partition_root"):
                    metadata = ROOT / config["partition_root"] / "metadata.json"
                    if not metadata.is_file():
                        raise FileNotFoundError(f"metadados das particoes ausentes: {metadata}")
                elif not (run_dir / "partitions" / "partition.json").is_file():
                    raise FileNotFoundError("artefato ausente: partitions/partition.json")
                rounds = pd.read_csv(run_dir / "rounds.csv")
                clients = pd.read_csv(run_dir / "clients.csv")
                with Image.open(run_dir / "convergence.png") as image:
                    image.verify()
                    image_sizes.add(image.size)
                if status.get("status") != "complete":
                    raise ValueError("status diferente de complete")
                expected_rows = int(status["rounds_completed"]) + 1
                if len(rounds) != expected_rows:
                    raise ValueError(f"rounds.csv tem {len(rounds)} linhas; esperado {expected_rows}")
                if not REQUIRED_ROUND_COLUMNS.issubset(rounds.columns):
                    raise ValueError("rounds.csv nao possui todas as colunas obrigatorias")
                expected_client_rows = int(status["rounds_completed"]) * int(config["num_clients"])
                if len(clients) != expected_client_rows:
                    raise ValueError("clients.csv nao registra todos os clientes/rounds")
                completed += 1
            except Exception as exc:
                failures.append(f"{run_id}: {exc}")
    if len(image_sizes) != 1:
        failures.append(f"curvas com dimensoes inconsistentes: {sorted(image_sizes)}")
    if failures:
        raise SystemExit("Baseline invalida:\n- " + "\n- ".join(failures))
    print(f"OK: {completed}/6 execucoes completas; curvas consistentes em {image_sizes.pop()}")


if __name__ == "__main__":
    main()

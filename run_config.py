#!/usr/bin/env python3
"""Executa baselines FedAvg reproduziveis para IID e non-IID."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import flwr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import ultralytics
import yaml

from fl.client import FLClient
from fl.model import evaluate_model, get_weights, load_model, set_weights
from fl.partition import count_split_images, load_dataset, materialize_partitions
from fl.server import aggregate_fedavg, get_strategy

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "taco_smoke.yaml")
    parser.add_argument("--scenario", choices=("iid", "non_iid"))
    parser.add_argument("--repetition", type=int, choices=(1, 2, 3))
    parser.add_argument("--all", action="store_true", help="executa as seis combinacoes")
    parser.add_argument("--force", action="store_true", help="sobrescreve uma execucao completa")
    args = parser.parse_args()
    if not args.all and (args.scenario is None or args.repetition is None):
        parser.error("use --all ou informe --scenario e --repetition")
    return args


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "dataset_yaml", "dataset_label", "scientific_valid", "model", "num_clients",
        "rounds", "local_epochs", "batch_size", "image_size", "device", "base_seed",
        "partition_seed", "output_dir",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Campos ausentes na configuracao: {', '.join(missing)}")
    for key in ("num_clients", "rounds", "local_epochs", "batch_size", "image_size"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} deve ser positivo")
    return config


def save_plot(rounds: pd.DataFrame, destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(rounds["round"], rounds["map50"], marker="o", label="mAP@0.5")
    axes[0].plot(rounds["round"], rounds["map50_95"], marker="s", label="mAP@0.5:0.95")
    axes[0].set(xlabel="Round", ylabel="mAP", ylim=(0, 1), title="Convergencia")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(rounds["round"], rounds["loss"], marker="o", color="tab:red")
    axes[1].set(xlabel="Round", ylabel="1 - mAP@0.5", title="Loss de avaliacao (proxy)")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def resolve_run_datasets(
    config: dict, scenario: str, run_dir: Path
) -> tuple[list[tuple[str, int]], str, int]:
    """Usa particoes reais preparadas ou gera particoes para o smoke test."""
    if config.get("partition_root"):
        partition_root = (ROOT / config["partition_root"]).resolve()
        partitions = []
        for index in range(int(config["num_clients"])):
            data_yaml = partition_root / scenario / f"client_{index}" / "data.yaml"
            if not data_yaml.is_file():
                raise FileNotFoundError(f"Particao ausente: {data_yaml}")
            partitions.append((str(data_yaml), count_split_images(data_yaml, "train")))
        evaluation_yaml = (ROOT / config["evaluation_yaml"]).resolve()
        if not evaluation_yaml.is_file():
            raise FileNotFoundError(f"Dataset de avaliacao ausente: {evaluation_yaml}")
        document, _, _ = load_dataset(Path(partitions[0][0]))
        return partitions, str(evaluation_yaml), len(document["names"])

    partitions = materialize_partitions(
        (ROOT / config["dataset_yaml"]).resolve(),
        scenario,
        int(config["num_clients"]),
        int(config["partition_seed"]),
        run_dir / "partitions",
    )
    document, _, _ = load_dataset((ROOT / config["dataset_yaml"]).resolve())
    return partitions, partitions[0][0], len(document["names"])


def run_one(config: dict, scenario: str, repetition: int, force: bool = False) -> dict:
    run_id = f"{scenario}_rep{repetition}"
    output_root = (ROOT / config["output_dir"]).resolve()
    run_dir = output_root / run_id
    status_path = run_dir / "status.json"
    if status_path.exists() and not force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            print(f"[skip] {run_id} ja esta completo")
            return status
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    seed = int(config["base_seed"]) + repetition
    resolved = dict(config)
    resolved.update({"scenario": scenario, "repetition": repetition, "seed": seed})
    (run_dir / "config.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    status_path.write_text(
        json.dumps({"run_id": run_id, "status": "running", "started_at": started.isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        partitions, validation_yaml, num_classes = resolve_run_datasets(
            config, scenario, run_dir
        )
        # Checkpoints locais sao temporarios e geram muitas escritas. Mantemos esse
        # scratch no disco rapido da sessao, mesmo quando output_dir e um Google
        # Drive montado para sobreviver a desconexoes do Colab.
        scratch_dir = ROOT / "runs" / "federated_scratch" / run_id
        # Valida a mesma Strategy que seria usada num servidor Flower distribuido.
        get_strategy(int(config["num_clients"]), config)
        clients = [
            FLClient(f"client_{index}", data_yaml, count, str(ROOT / config["model"]))
            for index, (data_yaml, count) in enumerate(partitions)
        ]
        global_model = load_model(str(ROOT / config["model"]), num_classes=num_classes)
        global_weights = get_weights(global_model)
        round_rows: list[dict] = []
        client_rows: list[dict] = []

        initial = evaluate_model(
            global_model, validation_yaml, int(config["image_size"]),
            int(config["batch_size"]), str(config["device"]),
        )
        round_rows.append({"round": 0, **initial, "elapsed_seconds": 0.0})
        clock = time.perf_counter()
        for server_round in range(1, int(config["rounds"]) + 1):
            updates = []
            for index, client in enumerate(clients):
                fit_config = {
                    "local_epochs": int(config["local_epochs"]),
                    "batch_size": int(config["batch_size"]),
                    "image_size": int(config["image_size"]),
                    "device": str(config["device"]),
                    "seed": seed + server_round * 100 + index,
                    "output_dir": str(scratch_dir),
                    "run_name": f"round_{server_round}_client_{index}",
                    "mu": float(config.get("mu", 0.0)),
                    "nbs": int(config.get("nbs", 64)),
                    **{
                        f"aug_{key}": float(value)
                        for key, value in (config.get("augmentation") or {}).items()
                    },
                }
                weights, count, metrics = client.fit(global_weights, fit_config)
                updates.append((weights, count))
                client_rows.append(
                    {"round": server_round, "client_id": f"client_{index}", "num_examples": count, **metrics}
                )
            global_weights = aggregate_fedavg(updates)
            set_weights(global_model, global_weights)
            metrics = evaluate_model(
                global_model, validation_yaml, int(config["image_size"]),
                int(config["batch_size"]), str(config["device"]),
            )
            round_rows.append({"round": server_round, **metrics, "elapsed_seconds": time.perf_counter() - clock})
            print(f"[{run_id}] round {server_round}/{config['rounds']} mAP50={metrics['map50']:.6f}")

        rounds_frame = pd.DataFrame(round_rows)
        pd.DataFrame(client_rows).to_csv(run_dir / "clients.csv", index=False)
        rounds_frame.to_csv(run_dir / "rounds.csv", index=False)
        save_plot(rounds_frame, run_dir / "convergence.png")
        global_model.save(run_dir / "final.pt")
        # Os checkpoints locais sao intermediarios (centenas de MB em seis runs).
        # Mantemos apenas o checkpoint global e as metricas consolidadas.
        shutil.rmtree(scratch_dir, ignore_errors=True)
        finished = datetime.now(timezone.utc)
        status = {
            "run_id": run_id,
            "status": "complete",
            "scenario": scenario,
            "repetition": repetition,
            "dataset_label": config["dataset_label"],
            "scientific_valid": bool(config["scientific_valid"]),
            "rounds_completed": int(config["rounds"]),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "final_map50": float(rounds_frame.iloc[-1]["map50"]),
            "versions": {
                "python": platform.python_version(), "torch": torch.__version__,
                "flower": flwr.__version__, "ultralytics": ultralytics.__version__,
                "pandas": pd.__version__,
            },
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        return status
    except Exception as exc:
        status_path.write_text(
            json.dumps({"run_id": run_id, "status": "failed", "error": repr(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


def write_summary(config: dict) -> None:
    root = (ROOT / config["output_dir"]).resolve()
    records = []
    for scenario in ("iid", "non_iid"):
        for repetition in (1, 2, 3):
            path = root / f"{scenario}_rep{repetition}" / "status.json"
            if path.exists():
                records.append(json.loads(path.read_text(encoding="utf-8")))
    columns = (
        "run_id", "status", "scenario", "repetition", "dataset_label",
        "scientific_valid", "rounds_completed", "duration_seconds", "final_map50",
    )
    summary_rows = [{key: record.get(key) for key in columns} for record in records]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "summary.csv", index=False)
    completed = summary[summary["status"] == "complete"]
    aggregate = completed.groupby("scenario", as_index=False).agg(
        repetitions=("run_id", "count"),
        map50_mean=("final_map50", "mean"),
        map50_std=("final_map50", "std"),
        duration_mean_seconds=("duration_seconds", "mean"),
    )
    aggregate.to_csv(root / "aggregate.csv", index=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    combinations = (
        [(scenario, repetition) for scenario in ("iid", "non_iid") for repetition in (1, 2, 3)]
        if args.all else [(args.scenario, args.repetition)]
    )
    for scenario, repetition in combinations:
        run_one(config, scenario, repetition, args.force)
        write_summary(config)


if __name__ == "__main__":
    main()

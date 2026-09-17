#!/usr/bin/env python3
"""Identifica combinacoes (cliente, classe) escassas pos-particionamento.

Card P2/S2 "Preparar estrategia de data augmentation para classes escassas
pos-Dirichlet": le as particoes materializadas por prepare_experiments.py, conta
instancias por (cenario, cliente, classe) direto dos labels YOLO e lista as
combinacoes abaixo do limiar. A mitigacao usa as augmentations nativas do
Ultralytics definidas no bloco `augmentation` de configs/baseline_taco.yaml.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITIONS = PROJECT_ROOT / "data" / "partitions"
DEFAULT_OUTPUT = PROJECT_ROOT / "taco_data" / "data" / "partitions" / "augmentation_report.json"
DEFAULT_SVG = PROJECT_ROOT / "taco_data" / "data" / "partitions" / "iid" / "distribution_iid.svg"
DEFAULT_THRESHOLD = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help="minimo de instancias por (cliente, classe) (default: 15)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--svg", type=Path, default=DEFAULT_SVG)
    return parser.parse_args()


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def count_instances(train_manifest: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line in train_manifest.read_text(encoding="utf-8").splitlines():
        image = Path(line.strip())
        if not line.strip():
            continue
        label = label_path(image)
        if not label.is_file():
            raise FileNotFoundError(f"Label ausente para {image}: {label}")
        for row in label.read_text(encoding="utf-8").splitlines():
            fields = row.split()
            if fields:
                counts[int(float(fields[0]))] += 1
    return counts


def write_distribution_svg(path: Path, report: dict, names: dict[int, str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    clients = report["iid"]
    class_ids = sorted(names)
    totals = [sum(client.values()) for client in clients.values()]
    bottoms = np.zeros(len(clients), dtype=float)
    x = np.arange(len(clients))
    figure, axis = plt.subplots(figsize=(11, 7))
    for class_id in class_ids:
        values = np.asarray(
            [client.get(names[class_id], 0) / total if total else 0 for client, total in zip(clients.values(), totals)],
            dtype=float,
        )
        axis.bar(x, values, bottom=bottoms, label=f"{class_id}: {names[class_id]}")
        bottoms += values
    axis.set_xticks(x, [f"{client}\nn={total}" for client, total in zip(clients, totals)])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Within-client fraction of object instances")
    axis.set_xlabel("IID client population")
    axis.set_title("IID partition - object instance distributions")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", metadata={"Creator": "taco_data", "Date": None})
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not PARTITIONS.is_dir():
        raise FileNotFoundError(
            f"Particoes ausentes em {PARTITIONS}. Rode taco_data/scripts/prepare_experiments.py."
        )
    names: dict[int, str] = {}
    report: dict[str, dict] = {}
    critical: list[dict] = []
    for scenario in ("iid", "non_iid"):
        report[scenario] = {}
        for client_dir in sorted((PARTITIONS / scenario).glob("client_*")):
            data_yaml = yaml.safe_load((client_dir / "data.yaml").read_text(encoding="utf-8"))
            names = {int(key): value for key, value in data_yaml["names"].items()}
            counts = count_instances(client_dir / "train.txt")
            per_class = {names[class_id]: counts.get(class_id, 0) for class_id in sorted(names)}
            report[scenario][client_dir.name] = per_class
            for class_name, value in per_class.items():
                if value < args.threshold:
                    critical.append(
                        {
                            "scenario": scenario,
                            "client": client_dir.name,
                            "class": class_name,
                            "train_instances": value,
                        }
                    )
    document = {
        "threshold": args.threshold,
        "note": (
            "Combinacoes abaixo do limiar dependem das augmentations nativas do "
            "Ultralytics (bloco 'augmentation' em configs/baseline_taco.yaml). "
            "Casos com 0 instancias nao sao resolviveis por augmentation e ficam "
            "documentados como limitacao conhecida do cenario non-IID."
        ),
        "critical_combinations": critical,
        "instances_per_client": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_distribution_svg(args.svg, report, names)
    print(f"{len(critical)} combinacoes (cenario, cliente, classe) abaixo de {args.threshold} instancias")
    for item in critical:
        print(f"  {item['scenario']:8s} {item['client']}  {item['class']:22s} {item['train_instances']}")
    print(f"Relatorio: {args.output}")
    print(f"Distribuicao IID: {args.svg}")


if __name__ == "__main__":
    main()

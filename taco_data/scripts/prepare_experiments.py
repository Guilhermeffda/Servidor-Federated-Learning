#!/usr/bin/env python3
"""Converte TACO-10 para YOLO e cria splits centralizado, IID e non-IID."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TACO_ROOT = PROJECT_ROOT / "taco_data"
ANNOTATIONS = TACO_ROOT / "data" / "processed" / "taco10" / "annotations_regrouped.json"
RAW_IMAGES = TACO_ROOT / "data" / "raw" / "taco" / "extracted" / "TACO" / "data"
CENTRALIZED = PROJECT_ROOT / "data" / "centralized"
PARTITIONS = PROJECT_ROOT / "data" / "partitions"
GLOBAL_TEST = TACO_ROOT / "data" / "test_global"
SEED = 42
NUM_CLIENTS = 5
GLOBAL_TEST_RATIO = 0.20
VALIDATION_RATIO_OF_DEVELOPMENT = 0.20
DIRICHLET_ALPHA = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--dirichlet-alpha", type=float, default=DIRICHLET_ALPHA)
    return parser.parse_args()


def dominant_labels(dataset: dict[str, Any]) -> dict[int, int]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        grouped[annotation["image_id"]].append(annotation["category_id"])
    result = {}
    for image in dataset["images"]:
        counts = Counter(grouped[image["id"]])
        result[image["id"]] = min(counts, key=lambda key: (-counts[key], key))
    return result


def split_global(
    image_ids: list[int], labels: dict[int, int], seed: int
) -> tuple[list[int], list[int], list[int]]:
    development, test = train_test_split(
        image_ids,
        test_size=GLOBAL_TEST_RATIO,
        random_state=seed,
        stratify=[labels[image_id] for image_id in image_ids],
    )
    train, validation = train_test_split(
        development,
        test_size=VALIDATION_RATIO_OF_DEVELOPMENT,
        random_state=seed,
        stratify=[labels[image_id] for image_id in development],
    )
    return sorted(train), sorted(validation), sorted(test)


def iid_partition(
    image_ids: list[int], labels: dict[int, int], num_clients: int, seed: int
) -> list[list[int]]:
    splitter = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
    values = np.asarray(image_ids)
    targets = np.asarray([labels[image_id] for image_id in image_ids])
    return [sorted(values[index].tolist()) for _, index in splitter.split(values, targets)]


def non_iid_partition(
    image_ids: list[int],
    labels: dict[int, int],
    num_clients: int,
    seed: int,
    alpha: float,
) -> list[list[int]]:
    if alpha <= 0:
        raise ValueError("dirichlet-alpha deve ser positivo")
    by_class: dict[int, list[int]] = defaultdict(list)
    for image_id in image_ids:
        by_class[labels[image_id]].append(image_id)
    minimum = max(1, len(image_ids) // (num_clients * 4))
    for attempt in range(100):
        rng = np.random.default_rng(seed + attempt)
        clients: list[list[int]] = [[] for _ in range(num_clients)]
        for class_id in sorted(by_class):
            class_images = np.asarray(by_class[class_id])
            rng.shuffle(class_images)
            counts = rng.multinomial(len(class_images), rng.dirichlet([alpha] * num_clients))
            cursor = 0
            for index, count in enumerate(counts):
                clients[index].extend(class_images[cursor : cursor + count].tolist())
                cursor += count
        if min(map(len, clients)) >= minimum:
            return [sorted(client) for client in clients]
    raise RuntimeError("Nao foi possivel gerar particoes non-IID sem clientes vazios")


def safe_name(image: dict[str, Any]) -> str:
    source = Path(image["file_name"])
    return f"{int(image['id']):06d}_{source.name}"


def ensure_symlink(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Imagem TACO ausente: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Link existente aponta para outro arquivo: {destination}")
        return
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == source.read_bytes():
            return
        raise FileExistsError(f"Destino ja existe e nao e link: {destination}")
    try:
        destination.symlink_to(source.resolve())
    except OSError as error:
        if getattr(error, "winerror", None) != 1314:
            raise
        shutil.copy2(source, destination)


def write_label(
    image: dict[str, Any], annotations: list[dict[str, Any]], destination: Path
) -> None:
    lines = []
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        lines.append(
            " ".join(
                [
                    str(annotation["category_id"]),
                    f"{(x + width / 2) / image['width']:.8f}",
                    f"{(y + height / 2) / image['height']:.8f}",
                    f"{width / image['width']:.8f}",
                    f"{height / image['height']:.8f}",
                ]
            )
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + ("\n" if lines else "")
    if destination.exists() and destination.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Label existente diverge do esperado: {destination}")
    destination.write_text(content, encoding="utf-8")


def materialize_split(
    split: str,
    image_ids: list[int],
    images: dict[int, dict[str, Any]],
    annotations: dict[int, list[dict[str, Any]]],
) -> dict[int, Path]:
    paths = {}
    for image_id in image_ids:
        image = images[image_id]
        name = safe_name(image)
        image_destination = CENTRALIZED / "images" / split / name
        label_destination = CENTRALIZED / "labels" / split / Path(name).with_suffix(".txt")
        ensure_symlink(RAW_IMAGES / image["file_name"], image_destination)
        write_label(image, annotations[image_id], label_destination)
        # Nao usar resolve(): o Ultralytics deriva o label trocando /images/ por
        # /labels/. Resolver o symlink faria essa derivacao apontar para o COCO bruto.
        paths[image_id] = image_destination.absolute()
        if split == "test":
            ensure_symlink(image_destination, GLOBAL_TEST / "images" / name)
            global_label = GLOBAL_TEST / "labels" / Path(name).with_suffix(".txt")
            ensure_symlink(label_destination, global_label)
    return paths


def names_mapping(dataset: dict[str, Any]) -> dict[int, str]:
    return {int(category["id"]): category["name"] for category in dataset["categories"]}


def write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_manifest(path: Path, image_ids: list[int], paths: dict[int, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(paths[image_id]) for image_id in image_ids) + "\n", encoding="utf-8")


def class_counts(image_ids: list[int], labels: dict[int, int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(labels[i] for i in image_ids).items())}


def main() -> None:
    args = parse_args()
    dataset = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in dataset["images"]}
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in dataset["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    labels = dominant_labels(dataset)
    train_ids, validation_ids, test_ids = split_global(sorted(images), labels, args.seed)
    assert set(train_ids).isdisjoint(validation_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(validation_ids).isdisjoint(test_ids)
    assert len(train_ids) + len(validation_ids) + len(test_ids) == len(images)

    paths = {}
    paths.update(materialize_split("train", train_ids, images, annotations))
    paths.update(materialize_split("val", validation_ids, images, annotations))
    paths.update(materialize_split("test", test_ids, images, annotations))
    names = names_mapping(dataset)
    write_yaml(
        CENTRALIZED / "data.yaml",
        {"path": str(CENTRALIZED.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": names},
    )
    write_yaml(
        GLOBAL_TEST / "data.yaml",
        {
            # Sem "path", o Ultralytics usa a pasta do proprio YAML. Isso mantem
            # o arquivo portavel entre o computador local e o Colab.
            "train": "images",
            "val": "images",
            "test": "images",
            "names": names,
        },
    )
    (GLOBAL_TEST / "test_image_ids.txt").write_text(
        "\n".join(map(str, test_ids)) + "\n", encoding="utf-8"
    )

    metadata: dict[str, Any] = {
        "seed": args.seed,
        "num_clients": args.num_clients,
        "dirichlet_alpha": args.dirichlet_alpha,
        "split_policy": "64% train, 16% validation, 20% fixed global test",
        "counts": {"train": len(train_ids), "val": len(validation_ids), "test": len(test_ids)},
        "test_ids_sha256": hashlib.sha256(
            ("\n".join(map(str, test_ids)) + "\n").encode()
        ).hexdigest(),
        "scenarios": {},
    }
    for scenario in ("iid", "non_iid"):
        partitioner = iid_partition if scenario == "iid" else non_iid_partition
        if scenario == "iid":
            train_parts = partitioner(train_ids, labels, args.num_clients, args.seed)
            val_parts = partitioner(validation_ids, labels, args.num_clients, args.seed)
        else:
            train_parts = partitioner(train_ids, labels, args.num_clients, args.seed, args.dirichlet_alpha)
            val_parts = partitioner(validation_ids, labels, args.num_clients, args.seed, args.dirichlet_alpha)
        metadata["scenarios"][scenario] = {}
        assigned_train = set()
        assigned_val = set()
        for index, (client_train, client_val) in enumerate(zip(train_parts, val_parts)):
            client_id = f"client_{index}"
            client_dir = PARTITIONS / scenario / client_id
            train_manifest = client_dir / "train.txt"
            val_manifest = client_dir / "val.txt"
            write_manifest(train_manifest, client_train, paths)
            write_manifest(val_manifest, client_val, paths)
            write_yaml(
                client_dir / "data.yaml",
                {
                    "path": str(client_dir.resolve()),
                    "train": str(train_manifest.resolve()),
                    "val": str(val_manifest.resolve()),
                    "test": str((CENTRALIZED / "images" / "test").resolve()),
                    "names": names,
                },
            )
            assigned_train.update(client_train)
            assigned_val.update(client_val)
            metadata["scenarios"][scenario][client_id] = {
                "train_images": len(client_train),
                "val_images": len(client_val),
                "train_primary_class_counts": class_counts(client_train, labels),
                "val_primary_class_counts": class_counts(client_val, labels),
            }
        assert assigned_train == set(train_ids)
        assert assigned_val == set(validation_ids)
        assert assigned_train.isdisjoint(test_ids)
        assert assigned_val.isdisjoint(test_ids)
    PARTITIONS.mkdir(parents=True, exist_ok=True)
    (PARTITIONS / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata["counts"], ensure_ascii=False))
    print(f"Centralizado: {CENTRALIZED / 'data.yaml'}")
    print(f"Particoes: {PARTITIONS}")


if __name__ == "__main__":
    main()

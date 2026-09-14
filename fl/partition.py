"""Particionamento reproduzivel de datasets YOLO em cenarios IID e non-IID."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_root(dataset_yaml: Path, configured: str | None) -> Path:
    if not configured:
        return dataset_yaml.parent.resolve()
    path = Path(configured)
    if path.is_absolute():
        return path
    project_candidate = (Path.cwd() / path).resolve()
    return project_candidate if project_candidate.exists() else (dataset_yaml.parent / path).resolve()


def _expand_split(value: Any, root: Path) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for item in values:
        path = Path(str(item))
        path = path if path.is_absolute() else root / path
        if path.is_dir():
            images.extend(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        elif path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                candidate = Path(line.strip())
                images.append(candidate if candidate.is_absolute() else path.parent / candidate)
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            images.extend(p for p in path.parent.glob(path.name) if p.suffix.lower() in IMAGE_EXTENSIONS)
    resolved = sorted({p.resolve() for p in images})
    missing = [str(p) for p in resolved if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Imagens ausentes no split: {missing[:3]}")
    return resolved


def load_dataset(dataset_yaml: Path) -> tuple[dict[str, Any], list[Path], list[Path]]:
    document = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = _resolve_root(dataset_yaml, document.get("path"))
    train = _expand_split(document["train"], root)
    val = _expand_split(document["val"], root)
    if not train or not val:
        raise ValueError("O dataset precisa ter imagens nos splits train e val")
    return document, train, val


def count_split_images(dataset_yaml: Path, split: str) -> int:
    """Conta imagens de um split sem depender de caches internos do Ultralytics."""
    document = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = _resolve_root(dataset_yaml, document.get("path"))
    return len(_expand_split(document[split], root))


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return image.with_suffix(".txt")


def _dominant_class(image: Path) -> int:
    label = _label_path(image)
    counts: dict[int, int] = {}
    if label.exists():
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                class_id = int(float(fields[0]))
                counts[class_id] = counts.get(class_id, 0) + 1
    return max(counts, key=lambda key: (counts[key], -key)) if counts else -1


def partition_images(
    images: list[Path], scenario: str, num_clients: int, seed: int
) -> list[list[Path]]:
    if len(images) < num_clients:
        raise ValueError(f"Ha {len(images)} imagens para {num_clients} clientes")
    if scenario not in {"iid", "non_iid"}:
        raise ValueError("scenario deve ser 'iid' ou 'non_iid'")
    ordered = list(images)
    if scenario == "iid":
        random.Random(seed).shuffle(ordered)
    else:
        ordered.sort(key=lambda image: (_dominant_class(image), str(image)))
    base, remainder = divmod(len(ordered), num_clients)
    sizes = [base + (index < remainder) for index in range(num_clients)]
    partitions: list[list[Path]] = []
    cursor = 0
    for size in sizes:
        partitions.append(ordered[cursor : cursor + size])
        cursor += size
    return partitions


def materialize_partitions(
    dataset_yaml: Path,
    scenario: str,
    num_clients: int,
    seed: int,
    output_dir: Path,
) -> list[tuple[str, int]]:
    document, train_images, val_images = load_dataset(dataset_yaml)
    partitions = partition_images(train_images, scenario, num_clients, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_manifest = output_dir / "validation.txt"
    val_manifest.write_text("\n".join(map(str, val_images)) + "\n", encoding="utf-8")
    metadata: dict[str, Any] = {"scenario": scenario, "seed": seed, "clients": {}}
    result: list[tuple[str, int]] = []
    for index, images in enumerate(partitions):
        client_id = f"client_{index}"
        client_dir = output_dir / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        train_manifest = client_dir / "train.txt"
        train_manifest.write_text("\n".join(map(str, images)) + "\n", encoding="utf-8")
        client_yaml = client_dir / "data.yaml"
        client_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(client_dir.resolve()),
                    "train": str(train_manifest.resolve()),
                    "val": str(val_manifest.resolve()),
                    "names": document["names"],
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        metadata["clients"][client_id] = {
            "num_images": len(images),
            "dominant_classes": [_dominant_class(image) for image in images],
            "images": [str(image) for image in images],
        }
        result.append((str(client_yaml), len(images)))
    (output_dir / "partition.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result

#!/usr/bin/env python3
"""Create the frozen uniform-random IID partition for TACO-10."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import math
import os
import shutil
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence

import numpy as np


NUMBER_OF_CLIENTS = 5
SEED = 42
VALIDATION_FRACTION = 0.20
EXPECTED_CLASSES = 10
METHOD = "uniform random image partition"
METHOD_NOTE = (
    "Development images are shuffled exactly once with NumPy default_rng(42) "
    "and divided with numpy.array_split; classes do not influence allocation."
)
PRIMARY_CLASS_RULE = (
    "Count annotations by category_id per image; choose the category with the "
    "most instances; break ties using the smallest category_id."
)


class PartitionError(RuntimeError):
    """Indicate an input, integrity, or materialization failure."""


@dataclass(frozen=True)
class Inputs:
    """Validated source data and indices."""

    document: dict[str, Any]
    categories: list[dict[str, Any]]
    images_by_id: dict[int, dict[str, Any]]
    annotations_by_image: dict[int, list[dict[str, Any]]]
    all_ids: set[int]
    test_ids: set[int]
    development_ids: set[int]
    annotations_sha256: str
    test_ids_sha256: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately small command-line interface."""
    parser = argparse.ArgumentParser(
        description="Materialize the deterministic five-client IID TACO-10 split."
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest with bounded memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PartitionError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object with explicit diagnostics."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PartitionError(f"could not load {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise PartitionError(f"{path} root must be an object")
    return document


def record_list(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Require a list composed only of JSON objects."""
    value = document.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise PartitionError(f"{key!r} must be a list of objects")
    return value


def load_test_ids(path: Path) -> set[int]:
    """Load unique integer global-test IDs."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PartitionError(f"could not read test IDs: {exc}") from exc
    parsed: list[int] = []
    for line_number, line in enumerate(lines, start=1):
        value = line.strip()
        if not value:
            continue
        try:
            parsed.append(int(value))
        except ValueError as exc:
            raise PartitionError(
                f"invalid test ID on line {line_number}: {value!r}"
            ) from exc
    duplicates = sorted(key for key, count in Counter(parsed).items() if count > 1)
    if duplicates:
        raise PartitionError(f"duplicate test IDs: {duplicates}")
    return set(parsed)


def validate_inputs(annotations_path: Path, test_ids_path: Path) -> Inputs:
    """Validate source structure, foreign keys, categories, and set invariants."""
    document = load_json_object(annotations_path)
    images = record_list(document, "images")
    annotations = record_list(document, "annotations")
    categories = record_list(document, "categories")
    if len(categories) != EXPECTED_CLASSES:
        raise PartitionError(
            f"expected {EXPECTED_CLASSES} categories, found {len(categories)}"
        )
    if any(not isinstance(item.get("id"), int) for item in categories):
        raise PartitionError("every category ID must be an integer")
    categories = sorted(categories, key=lambda item: item["id"])
    category_ids = [item["id"] for item in categories]
    if category_ids != list(range(EXPECTED_CLASSES)):
        raise PartitionError("category IDs must be exactly 0 through 9")

    image_ids = [item.get("id") for item in images]
    if any(not isinstance(image_id, int) for image_id in image_ids):
        raise PartitionError("every image ID must be an integer")
    if len(image_ids) != len(set(image_ids)):
        raise PartitionError("image IDs must be unique")
    images_by_id = {item["id"]: item for item in images}
    all_ids = set(images_by_id)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id not in all_ids:
            raise PartitionError(f"annotation references unknown image {image_id!r}")
        if category_id not in set(category_ids):
            raise PartitionError(
                f"annotation references unknown category {category_id!r}"
            )
        annotations_by_image[image_id].append(annotation)

    test_ids = load_test_ids(test_ids_path)
    unknown_test = sorted(test_ids - all_ids)
    if unknown_test:
        raise PartitionError(f"test IDs absent from TACO-10: {unknown_test}")
    development_ids = all_ids - test_ids
    if test_ids & development_ids:
        raise PartitionError("test and development IDs overlap")
    if test_ids | development_ids != all_ids:
        raise PartitionError("test and development IDs do not cover the dataset")
    return Inputs(
        document=document,
        categories=categories,
        images_by_id=images_by_id,
        annotations_by_image=dict(annotations_by_image),
        all_ids=all_ids,
        test_ids=test_ids,
        development_ids=development_ids,
        annotations_sha256=sha256_file(annotations_path),
        test_ids_sha256=sha256_file(test_ids_path),
    )


def validate_populations(
    populations: list[list[int]], inputs: Inputs
) -> dict[str, Any]:
    """Require exact, disjoint, balanced development-set coverage."""
    if len(populations) != NUMBER_OF_CLIENTS:
        raise PartitionError("the IID partition must contain exactly five clients")
    if any(not population for population in populations):
        raise PartitionError("an IID client is empty")
    flattened = [image_id for population in populations for image_id in population]
    assigned = set(flattened)
    checks = {
        "assigned_image_occurrences": len(flattened),
        "assigned_unique_images": len(assigned),
        "cross_client_overlap": len(flattened) - len(assigned),
        "global_test_overlap": len(assigned & inputs.test_ids),
        "unassigned_images": len(inputs.development_ids - assigned),
        "unexpected_images": len(assigned - inputs.development_ids),
        "union_equals_development_ids": assigned == inputs.development_ids,
        "each_development_image_exactly_once": (
            len(flattened) == len(inputs.development_ids)
            and assigned == inputs.development_ids
        ),
    }
    sizes = [len(population) for population in populations]
    if max(sizes) - min(sizes) > 1:
        raise PartitionError(f"IID client sizes differ by more than one: {sizes}")
    if not (
        checks["cross_client_overlap"] == 0
        and checks["global_test_overlap"] == 0
        and checks["unassigned_images"] == 0
        and checks["unexpected_images"] == 0
        and checks["each_development_image_exactly_once"]
    ):
        raise PartitionError(f"IID population integrity failure: {checks}")
    return checks


def generate_populations(inputs: Inputs) -> list[list[int]]:
    """Shuffle sorted development IDs once and split them uniformly."""
    shuffled = np.asarray(sorted(inputs.development_ids), dtype=np.int64)
    np.random.default_rng(SEED).shuffle(shuffled)
    return [
        [int(value) for value in part]
        for part in np.array_split(shuffled, NUMBER_OF_CLIENTS)
    ]


def generate_splits(populations: list[list[int]]) -> list[dict[str, Any]]:
    """Generate deterministic internal train/validation splits."""
    splits: list[dict[str, Any]] = []
    for client_id, population in enumerate(populations):
        shuffled = np.asarray(sorted(population), dtype=np.int64)
        np.random.default_rng(SEED + client_id).shuffle(shuffled)
        val_count = max(1, round(len(population) * VALIDATION_FRACTION))
        if val_count >= len(population):
            raise PartitionError(f"client_{client_id} has a degenerate split")
        val_ids = [int(value) for value in shuffled[:val_count]]
        train_ids = [int(value) for value in shuffled[val_count:]]
        if set(train_ids) & set(val_ids) or set(train_ids) | set(val_ids) != set(
            population
        ):
            raise PartitionError(f"client_{client_id} split integrity failure")
        splits.append(
            {
                "client_id": client_id,
                "split_seed": SEED + client_id,
                "all_image_ids": population,
                "train_image_ids": train_ids,
                "val_image_ids": val_ids,
            }
        )
    return splits


def load_frozen_splits(
    manifest_path: Path, inputs: Inputs, input_hashes: dict[str, str]
) -> list[dict[str, Any]] | None:
    """Validate and return authoritative populations and splits from a manifest."""
    if not manifest_path.is_file():
        return None
    manifest = load_json_object(manifest_path)
    expected = {
        "method": METHOD,
        "number_of_clients": NUMBER_OF_CLIENTS,
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "input_hashes": input_hashes,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise PartitionError(f"existing manifest has incompatible {key}")
    clients = manifest.get("clients")
    if not isinstance(clients, list) or len(clients) != NUMBER_OF_CLIENTS:
        raise PartitionError("existing manifest has invalid clients")
    splits: list[dict[str, Any]] = []
    for client_id, client in enumerate(clients):
        if client.get("client_id") != client_id:
            raise PartitionError("existing manifest client order is invalid")
        keys = ("all_image_ids", "train_image_ids", "val_image_ids")
        if any(not isinstance(client.get(key), list) for key in keys):
            raise PartitionError(f"client_{client_id} has invalid frozen IDs")
        if client.get("split_seed") != SEED + client_id:
            raise PartitionError(f"client_{client_id} split seed is incompatible")
        population = client["all_image_ids"]
        train_ids = client["train_image_ids"]
        val_ids = client["val_image_ids"]
        if (
            set(train_ids) & set(val_ids)
            or set(train_ids) | set(val_ids) != set(population)
            or len(val_ids) != max(1, round(len(population) * VALIDATION_FRACTION))
        ):
            raise PartitionError(f"client_{client_id} frozen split is invalid")
        splits.append(
            {
                "client_id": client_id,
                "split_seed": SEED + client_id,
                "all_image_ids": list(population),
                "train_image_ids": list(train_ids),
                "val_image_ids": list(val_ids),
            }
        )
    validate_populations([item["all_image_ids"] for item in splits], inputs)
    return splits


def safe_relative_path(file_name: Any, raw_root: Path) -> Path:
    """Validate and resolve an image path beneath the raw root."""
    if not isinstance(file_name, str) or not file_name.strip() or "\x00" in file_name:
        raise PartitionError(f"invalid image file_name: {file_name!r}")
    posix = PurePosixPath(file_name)
    windows = PureWindowsPath(file_name)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise PartitionError(f"unsafe image file_name: {file_name!r}")
    resolved = (raw_root / Path(file_name)).resolve()
    try:
        return resolved.relative_to(raw_root.resolve())
    except ValueError as exc:
        raise PartitionError(f"image path escapes raw root: {file_name!r}") from exc


def render_label(
    image: dict[str, Any],
    annotations: Iterable[dict[str, Any]],
) -> str:
    """Convert every image annotation to one YOLO line."""
    width = image.get("width")
    height = image.get("height")
    if (
        not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
        or width <= 0
        or height <= 0
    ):
        raise PartitionError(f"image {image.get('id')} has invalid dimensions")
    lines: list[str] = []
    for annotation in annotations:
        bbox = annotation.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, (int, float)) for value in bbox)
        ):
            raise PartitionError(f"annotation {annotation.get('id')} has invalid bbox")
        x, y, box_width, box_height = bbox
        lines.append(
            f"{annotation['category_id']} "
            f"{(x + box_width / 2) / width:.6f} "
            f"{(y + box_height / 2) / height:.6f} "
            f"{box_width / width:.6f} "
            f"{box_height / height:.6f}\n"
        )
    return "".join(lines)


def yaml_text(categories: list[dict[str, Any]]) -> str:
    """Create the portable local train/validation dataset YAML."""
    lines = ["train: images/train", "val: images/val", "", "names:"]
    for category in categories:
        lines.append(f"  {category['id']}: {category['name']}")
    return "\n".join(lines) + "\n"


def build_plan(
    inputs: Inputs, splits: list[dict[str, Any]], raw_root: Path
) -> list[dict[str, Any]]:
    """Resolve all sources and reject destination collisions before writes."""
    plan: list[dict[str, Any]] = []
    destinations: dict[str, list[int]] = defaultdict(list)
    for split in splits:
        client_id = split["client_id"]
        for split_name, key in (("train", "train_image_ids"), ("val", "val_image_ids")):
            for image_id in split[key]:
                image = inputs.images_by_id[image_id]
                relative_image = safe_relative_path(image.get("file_name"), raw_root)
                source = raw_root / relative_image
                if not source.is_file():
                    raise PartitionError(f"raw image is missing: {source}")
                relative_label = relative_image.with_suffix(".txt")
                image_destination = (
                    Path(f"client_{client_id}/images/{split_name}") / relative_image
                )
                label_destination = (
                    Path(f"client_{client_id}/labels/{split_name}") / relative_label
                )
                for destination in (image_destination, label_destination):
                    destinations[destination.as_posix().casefold()].append(image_id)
                plan.append(
                    {
                        "client_id": client_id,
                        "split": split_name,
                        "image_id": image_id,
                        "source": source,
                        "image_destination": image_destination,
                        "label_destination": label_destination,
                        "label": render_label(
                            image, inputs.annotations_by_image.get(image_id, [])
                        ),
                    }
                )
    collisions = {key: ids for key, ids in destinations.items() if len(ids) > 1}
    if collisions:
        raise PartitionError(f"case-insensitive destination collisions: {collisions}")
    return plan


def remove_tree(path: Path) -> None:
    """Remove only a derived tree, including read-only OneDrive entries."""
    if not path.exists():
        return

    def make_writable(function: Any, name: str, _error: Any) -> None:
        os.chmod(name, stat.S_IWRITE)
        function(name)

    os.chmod(path, stat.S_IWRITE)
    shutil.rmtree(path, onerror=make_writable)


def validate_staging(plan: list[dict[str, Any]], staging: Path) -> dict[str, int]:
    """Verify exact image/label inventory, bytes, contents, and line syntax."""
    expected_images = {item["image_destination"].as_posix() for item in plan}
    expected_labels = {item["label_destination"].as_posix() for item in plan}
    actual_images: set[str] = set()
    actual_labels: set[str] = set()
    for path in staging.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(staging).as_posix()
        if "/images/" in f"/{relative}":
            actual_images.add(relative)
        elif "/labels/" in f"/{relative}" and path.suffix == ".txt":
            actual_labels.add(relative)
    if actual_images != expected_images:
        raise PartitionError("staged image inventory differs from the plan")
    if actual_labels != expected_labels:
        raise PartitionError("staged label inventory differs from the plan")
    lines = 0
    for item in plan:
        copied = staging / item["image_destination"]
        label_path = staging / item["label_destination"]
        if not filecmp.cmp(item["source"], copied, shallow=False):
            raise PartitionError(f"copied image differs: {item['image_destination']}")
        actual = label_path.read_text(encoding="utf-8")
        if actual != item["label"]:
            raise PartitionError(f"label differs: {item['label_destination']}")
        for line in actual.splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise PartitionError(f"invalid YOLO line: {item['label_destination']}")
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
            if class_id not in range(EXPECTED_CLASSES) or any(
                not math.isfinite(value) for value in coordinates
            ):
                raise PartitionError(f"invalid YOLO value: {item['label_destination']}")
            lines += 1
    return {
        "missing_files": 0,
        "missing_labels": 0,
        "destination_collisions": 0,
        "validated_label_lines": lines,
    }


def promote(output_root: Path, staging: Path) -> None:
    """Promote all staged clients together and restore old clients on failure."""
    backups = [
        output_root / f".client_{client_id}.backup"
        for client_id in range(NUMBER_OF_CLIENTS)
    ]
    if any(path.exists() for path in backups):
        raise PartitionError("residual IID backup prevents safe promotion")
    backed_up: list[int] = []
    promoted: list[int] = []
    try:
        for client_id in range(NUMBER_OF_CLIENTS):
            current = output_root / f"client_{client_id}"
            if current.exists():
                os.replace(current, backups[client_id])
                backed_up.append(client_id)
        for client_id in range(NUMBER_OF_CLIENTS):
            os.replace(
                staging / f"client_{client_id}",
                output_root / f"client_{client_id}",
            )
            promoted.append(client_id)
    except OSError as exc:
        try:
            for client_id in reversed(promoted):
                remove_tree(output_root / f"client_{client_id}")
            for client_id in backed_up:
                os.replace(backups[client_id], output_root / f"client_{client_id}")
        except OSError as rollback_error:
            raise PartitionError(
                f"promotion failed ({exc}) and rollback failed ({rollback_error})"
            ) from rollback_error
        raise PartitionError(f"promotion failed; old clients restored: {exc}") from exc
    for backup in backups:
        remove_tree(backup)
    remove_tree(staging)


def normalized(counts: list[int]) -> list[float]:
    """Normalize counts to proportions."""
    total = sum(counts)
    return [count / total if total else 0.0 for count in counts]


def jsd(first: list[float], second: list[float]) -> float:
    """Calculate base-2 Jensen-Shannon divergence."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    midpoint = (left + right) / 2

    def divergence(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return (divergence(left) + divergence(right)) / 2


def instance_counts(image_ids: list[int], inputs: Inputs) -> list[int]:
    """Count final TACO-10 instances for images."""
    counts = [0] * EXPECTED_CLASSES
    for image_id in image_ids:
        for annotation in inputs.annotations_by_image.get(image_id, []):
            counts[annotation["category_id"]] += 1
    return counts


def primary_classes(inputs: Inputs) -> dict[int, int]:
    """Calculate the comparison-only primary class for each development image."""
    result: dict[int, int] = {}
    for image_id in sorted(inputs.development_ids):
        counts = Counter(
            annotation["category_id"]
            for annotation in inputs.annotations_by_image.get(image_id, [])
        )
        result[image_id] = (
            min(counts, key=lambda key: (-counts[key], key)) if counts else 1
        )
    return result


def distribution_records(
    counts: list[int], categories: list[dict[str, Any]], count_name: str
) -> list[dict[str, Any]]:
    """Create ordered count/proportion records."""
    proportions = normalized(counts)
    return [
        {
            "category_id": category["id"],
            "category_name": category["name"],
            count_name: counts[index],
            "proportion": proportions[index],
        }
        for index, category in enumerate(categories)
    ]


def client_metrics(
    split: dict[str, Any],
    inputs: Inputs,
    primary_by_image: dict[int, int],
    global_distribution: list[float],
) -> dict[str, Any]:
    """Build scientific counts and IID similarity metrics for one client."""
    all_ids = split["all_image_ids"]
    train_ids = split["train_image_ids"]
    val_ids = split["val_image_ids"]
    total_counts = instance_counts(all_ids, inputs)
    train_counts = instance_counts(train_ids, inputs)
    val_counts = instance_counts(val_ids, inputs)
    primary_counter = Counter(primary_by_image[image_id] for image_id in all_ids)
    primary_counts = [primary_counter[index] for index in range(EXPECTED_CLASSES)]
    local_distribution = normalized(total_counts)
    return {
        "client_id": split["client_id"],
        "split_seed": split["split_seed"],
        "total_images": len(all_ids),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "all_image_ids": all_ids,
        "train_image_ids": train_ids,
        "val_image_ids": val_ids,
        "total_instances": sum(total_counts),
        "train_instances": sum(train_counts),
        "val_instances": sum(val_counts),
        "primary_distribution": distribution_records(
            primary_counts, inputs.categories, "primary_image_count"
        ),
        "full_instance_distribution": distribution_records(
            total_counts, inputs.categories, "instance_count"
        ),
        "train_full_instance_distribution": distribution_records(
            train_counts, inputs.categories, "instance_count"
        ),
        "val_full_instance_distribution": distribution_records(
            val_counts, inputs.categories, "instance_count"
        ),
        "class_coverage": {
            "primary": sum(value > 0 for value in primary_counts),
            "full_instances": sum(value > 0 for value in total_counts),
            "train_full_instances": sum(value > 0 for value in train_counts),
            "val_full_instances": sum(value > 0 for value in val_counts),
        },
        "jsd_to_global": jsd(local_distribution, global_distribution),
        "total_variation_distance_to_global": 0.5
        * sum(
            abs(local - global_value)
            for local, global_value in zip(
                local_distribution, global_distribution, strict=True
            )
        ),
        "maximum_absolute_class_proportion_difference_to_global": max(
            abs(local - global_value)
            for local, global_value in zip(
                local_distribution, global_distribution, strict=True
            )
        ),
    }


def build_manifest(
    inputs: Inputs,
    splits: list[dict[str, Any]],
    input_paths: dict[str, str],
    input_hashes: dict[str, str],
    checks: dict[str, Any],
    materialization: dict[str, int],
) -> dict[str, Any]:
    """Build the deterministic final IID scientific manifest."""
    global_counts = instance_counts(sorted(inputs.development_ids), inputs)
    global_distribution = normalized(global_counts)
    primary_by_image = primary_classes(inputs)
    clients = [
        client_metrics(split, inputs, primary_by_image, global_distribution)
        for split in splits
    ]
    pairwise: list[dict[str, Any]] = []
    for left in range(NUMBER_OF_CLIENTS):
        left_distribution = [
            item["proportion"]
            for item in clients[left]["full_instance_distribution"]
        ]
        for right in range(left + 1, NUMBER_OF_CLIENTS):
            right_distribution = [
                item["proportion"]
                for item in clients[right]["full_instance_distribution"]
            ]
            pairwise.append(
                {
                    "client_a": left,
                    "client_b": right,
                    "jsd": jsd(left_distribution, right_distribution),
                }
            )
    sizes = np.asarray([len(item["all_image_ids"]) for item in splits], dtype=float)
    pairwise_values = [item["jsd"] for item in pairwise]
    integrity = {
        **checks,
        "train_val_overlap": sum(
            len(set(item["train_image_ids"]) & set(item["val_image_ids"]))
            for item in splits
        ),
        "all_clients_nonempty": all(item["all_image_ids"] for item in splits),
        "all_train_nonempty": all(item["train_image_ids"] for item in splits),
        "all_val_nonempty": all(item["val_image_ids"] for item in splits),
        **materialization,
    }
    return {
        "schema_version": 1,
        "method": METHOD,
        "method_note": METHOD_NOTE,
        "number_of_clients": NUMBER_OF_CLIENTS,
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "client_split_seeds": {
            f"client_{client_id}": SEED + client_id
            for client_id in range(NUMBER_OF_CLIENTS)
        },
        "input_paths": input_paths,
        "input_hashes": input_hashes,
        "classes": [
            {"id": category["id"], "name": category["name"]}
            for category in inputs.categories
        ],
        "total_images": len(inputs.all_ids),
        "global_test_images": len(inputs.test_ids),
        "development_images": len(inputs.development_ids),
        "terminology_note": (
            "All reported class distributions use the 10 final TACO-10 classes, "
            "not the 28 original supercategories."
        ),
        "primary_class_rule": PRIMARY_CLASS_RULE,
        "global_development_full_instance_distribution": distribution_records(
            global_counts, inputs.categories, "instance_count"
        ),
        "quantity_metrics": {
            "min_client_images": int(sizes.min()),
            "max_client_images": int(sizes.max()),
            "mean_client_images": float(sizes.mean()),
            "std_client_images": float(sizes.std()),
            "coefficient_of_variation": float(sizes.std() / sizes.mean()),
        },
        "pairwise_jsd": pairwise,
        "mean_pairwise_jsd": float(np.mean(pairwise_values)),
        "max_pairwise_jsd": max(pairwise_values),
        "integrity": integrity,
        "clients": clients,
    }


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """Write readable, sorted, deterministic JSON atomically."""
    temporary = path.with_name(path.name + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise PartitionError(f"could not write manifest: {exc}") from exc


def materialize(
    output_root: Path,
    inputs: Inputs,
    splits: list[dict[str, Any]],
    raw_root: Path,
) -> dict[str, int]:
    """Build, validate, and atomically promote all client datasets."""
    plan = build_plan(inputs, splits, raw_root)
    staging = output_root / ".materialization_staging"
    remove_tree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        data_yaml = yaml_text(inputs.categories)
        for client_id in range(NUMBER_OF_CLIENTS):
            client_root = staging / f"client_{client_id}"
            for relative in (
                "images/train",
                "images/val",
                "labels/train",
                "labels/val",
            ):
                (client_root / relative).mkdir(parents=True)
            (client_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
        for item in plan:
            image_destination = staging / item["image_destination"]
            label_destination = staging / item["label_destination"]
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["source"], image_destination)
            label_destination.write_text(item["label"], encoding="utf-8")
        result = validate_staging(plan, staging)
        promote(output_root, staging)
        return result
    except (OSError, PartitionError, KeyError, TypeError, ValueError):
        remove_tree(staging)
        raise


def run(_args: argparse.Namespace) -> int:
    """Validate inputs, freeze/reuse IID IDs, and materialize clients."""
    root = Path(__file__).resolve().parent.parent
    annotations_path = root / "data/processed/taco10/annotations_regrouped.json"
    test_ids_path = root / "data/test_global/test_image_ids.txt"
    raw_root = root / "data/raw/taco/extracted/TACO/data"
    output_root = root / "data/partitions/iid"
    manifest_path = output_root / "manifest_iid.json"
    inputs = validate_inputs(annotations_path, test_ids_path)
    input_hashes = {
        "annotations_regrouped_sha256": inputs.annotations_sha256,
        "test_image_ids_sha256": inputs.test_ids_sha256,
    }
    splits = load_frozen_splits(manifest_path, inputs, input_hashes)
    reused = splits is not None
    if splits is None:
        populations = generate_populations(inputs)
        validate_populations(populations, inputs)
        splits = generate_splits(populations)
    checks = validate_populations(
        [item["all_image_ids"] for item in splits], inputs
    )
    materialization = materialize(output_root, inputs, splits, raw_root)
    manifest = build_manifest(
        inputs,
        splits,
        {
            "annotations": annotations_path.relative_to(root).as_posix(),
            "test_image_ids": test_ids_path.relative_to(root).as_posix(),
        },
        input_hashes,
        checks,
        materialization,
    )
    write_json_atomic(manifest_path, manifest)
    print("=== IID materialization ===")
    for client in manifest["clients"]:
        print(
            f"client_{client['client_id']}: total={client['total_images']}, "
            f"train={client['train_images']}, val={client['val_images']}, "
            f"instances={client['total_instances']}"
        )
    print(f"Frozen manifest reused: {reused}")
    print(f"Images and labels materialized: {len(inputs.development_ids)} each")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Return a clean nonzero status for expected failures."""
    try:
        return run(parse_args(argv))
    except PartitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

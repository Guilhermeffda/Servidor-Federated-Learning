#!/usr/bin/env python3
"""Analyze candidate Dirichlet alphas for a future non-IID partition."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import logging
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


LOGGER = logging.getLogger("taco-non-iid")
NUMBER_OF_CLIENTS = 5
SEED = 42
CANDIDATE_ALPHAS = (0.1, 0.5, 1.0)
EXPECTED_CLASSES = 10
SELECTED_ALPHA = 0.5
VALIDATION_FRACTION = 0.20
METHOD = "class-wise Dirichlet allocation by primary image class"
METHOD_NOTE = (
    "Adaptation for object detection inspired by Hsu, Qi, and Brown (2019); "
    "this does not reproduce their experimental procedure exactly. Images "
    "are indivisible allocation units and retain all object annotations."
)
PRIMARY_CLASS_RULE = (
    "Count annotations by category_id per image; choose the category with "
    "the most instances; break ties using the smallest category_id."
)


class AnalysisError(RuntimeError):
    """Indicate an invalid input or failed allocation invariant."""


@dataclass(frozen=True)
class Inputs:
    """Validated analysis inputs and their derived indices."""

    document: dict[str, Any]
    categories: list[dict[str, Any]]
    all_ids: set[int]
    test_ids: set[int]
    development_ids: set[int]
    annotations_by_image: dict[int, list[dict[str, Any]]]
    annotations_sha256: str
    test_ids_sha256: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the mutually exclusive analysis/materialization interface."""
    parser = argparse.ArgumentParser(
        description="Analyze candidate Dirichlet alphas for TACO-10."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--analyze",
        action="store_true",
        help="generate deterministic alpha analysis JSON and SVG artifacts",
    )
    modes.add_argument(
        "--materialize",
        action="store_true",
        help="materialize the frozen alpha=0.5 clients and their train/val splits",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest using bounded memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AnalysisError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object or raise a clear analysis error."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"could not load {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise AnalysisError("annotations_regrouped.json root must be an object")
    return document


def require_record_list(document: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Validate one required collection of JSON objects."""
    collection = document.get(name)
    if not isinstance(collection, list):
        raise AnalysisError(f"{name!r} must be a list")
    if any(not isinstance(record, dict) for record in collection):
        raise AnalysisError(f"every item in {name!r} must be an object")
    return collection


def load_test_ids(path: Path) -> set[int]:
    """Load unique integer test IDs without modifying their source file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"could not read test IDs: {exc}") from exc
    parsed: list[int] = []
    for line_number, line in enumerate(lines, start=1):
        value = line.strip()
        if not value:
            continue
        try:
            parsed.append(int(value))
        except ValueError as exc:
            raise AnalysisError(
                f"invalid test image ID at line {line_number}: {value!r}"
            ) from exc
    counts = Counter(parsed)
    duplicates = sorted(image_id for image_id, count in counts.items() if count > 1)
    if duplicates:
        raise AnalysisError(f"duplicate test image IDs: {duplicates}")
    return set(parsed)


def validate_inputs(annotations_path: Path, test_ids_path: Path) -> Inputs:
    """Load inputs and enforce all structural and set invariants."""
    document = load_json_object(annotations_path)
    images = require_record_list(document, "images")
    annotations = require_record_list(document, "annotations")
    categories = require_record_list(document, "categories")
    if len(categories) != EXPECTED_CLASSES:
        raise AnalysisError(
            f"expected {EXPECTED_CLASSES} TACO-10 categories; found {len(categories)}"
        )

    category_ids = [category.get("id") for category in categories]
    if any(not isinstance(category_id, int) for category_id in category_ids):
        raise AnalysisError("every category ID must be an integer")
    if len(set(category_ids)) != len(category_ids):
        raise AnalysisError("category IDs must be unique")
    categories = sorted(categories, key=lambda category: category["id"])
    if [category["id"] for category in categories] != list(range(EXPECTED_CLASSES)):
        raise AnalysisError("TACO-10 category IDs must be exactly 0 through 9")
    valid_category_ids = set(category_ids)

    image_ids = [image.get("id") for image in images]
    if any(not isinstance(image_id, int) for image_id in image_ids):
        raise AnalysisError("every image ID must be an integer")
    all_ids = set(image_ids)
    if len(all_ids) != len(image_ids):
        raise AnalysisError("image IDs must be unique")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id not in all_ids:
            raise AnalysisError(f"annotation references unknown image ID {image_id!r}")
        if category_id not in valid_category_ids:
            raise AnalysisError(
                f"annotation references unknown category ID {category_id!r}"
            )
        annotations_by_image[image_id].append(annotation)

    test_ids = load_test_ids(test_ids_path)
    unknown_test_ids = sorted(test_ids - all_ids)
    if unknown_test_ids:
        raise AnalysisError(f"test IDs absent from TACO-10: {unknown_test_ids}")
    development_ids = all_ids - test_ids
    if test_ids & development_ids:
        raise AnalysisError("test and development IDs overlap")
    if test_ids | development_ids != all_ids:
        raise AnalysisError("test and development IDs do not cover all images")

    return Inputs(
        document=document,
        categories=categories,
        all_ids=all_ids,
        test_ids=test_ids,
        development_ids=development_ids,
        annotations_by_image=dict(annotations_by_image),
        annotations_sha256=sha256_file(annotations_path),
        test_ids_sha256=sha256_file(test_ids_path),
    )


def primary_classes(inputs: Inputs) -> tuple[dict[int, int], dict[int, set[int]]]:
    """Calculate primary and complete class sets for development images."""
    other_ids = [
        category["id"]
        for category in inputs.categories
        if category.get("name") == "Other"
    ]
    if len(other_ids) != 1:
        raise AnalysisError("exactly one TACO-10 category named 'Other' is required")
    primary_by_image: dict[int, int] = {}
    classes_by_image: dict[int, set[int]] = {}
    for image_id in sorted(inputs.development_ids):
        annotations = inputs.annotations_by_image.get(image_id, [])
        counts = Counter(annotation["category_id"] for annotation in annotations)
        classes_by_image[image_id] = set(counts)
        if counts:
            primary_by_image[image_id] = min(
                counts,
                key=lambda category_id: (-counts[category_id], category_id),
            )
        else:
            primary_by_image[image_id] = other_ids[0]
    return primary_by_image, classes_by_image


def ordered_primary_strata(
    primary_by_image: dict[int, int], category_ids: list[int]
) -> dict[int, list[int]]:
    """Shuffle each sorted primary stratum once for reuse by every alpha."""
    grouped: dict[int, list[int]] = {category_id: [] for category_id in category_ids}
    for image_id, category_id in primary_by_image.items():
        grouped[category_id].append(image_id)
    generator = np.random.default_rng(SEED)
    ordered: dict[int, list[int]] = {}
    for category_id in category_ids:
        image_ids = np.asarray(sorted(grouped[category_id]), dtype=np.int64)
        generator.shuffle(image_ids)
        ordered[category_id] = [int(image_id) for image_id in image_ids]
    return ordered


def largest_remainder_counts(proportions: np.ndarray, total: int) -> list[int]:
    """Convert proportions to exact integer counts with deterministic ties."""
    quotas = proportions * total
    floors = np.floor(quotas).astype(np.int64)
    remainder = total - int(floors.sum())
    fractional = quotas - floors
    recipients = sorted(
        range(len(proportions)),
        key=lambda client_id: (-float(fractional[client_id]), client_id),
    )[:remainder]
    for client_id in recipients:
        floors[client_id] += 1
    counts = [int(value) for value in floors]
    if sum(counts) != total:
        raise AnalysisError("largest remainder allocation did not preserve total")
    return counts


def allocate_alpha(
    alpha: float,
    ordered_strata: dict[int, list[int]],
    category_ids: list[int],
) -> tuple[list[list[int]], dict[str, list[float]], dict[str, list[int]]]:
    """Allocate whole images class-wise for one deterministic alpha."""
    generator = np.random.default_rng(SEED)
    clients: list[list[int]] = [[] for _ in range(NUMBER_OF_CLIENTS)]
    sampled_proportions: dict[str, list[float]] = {}
    allocation_counts: dict[str, list[int]] = {}
    for category_id in category_ids:
        proportions = generator.dirichlet(
            alpha * np.ones(NUMBER_OF_CLIENTS, dtype=np.float64)
        )
        counts = largest_remainder_counts(
            proportions,
            len(ordered_strata[category_id]),
        )
        sampled_proportions[str(category_id)] = [
            float(value) for value in proportions
        ]
        allocation_counts[str(category_id)] = counts
        start = 0
        for client_id, count in enumerate(counts):
            stop = start + count
            clients[client_id].extend(ordered_strata[category_id][start:stop])
            start = stop
        if start != len(ordered_strata[category_id]):
            raise AnalysisError(f"class {category_id} allocation is incomplete")
    return clients, sampled_proportions, allocation_counts


def normalized_distribution(counts: list[int]) -> list[float]:
    """Normalize nonnegative counts, returning zeros for an empty total."""
    total = sum(counts)
    if total == 0:
        return [0.0 for _ in counts]
    return [count / total for count in counts]


def normalized_entropy(distribution: list[float]) -> float:
    """Calculate Shannon entropy normalized by ln(K)."""
    entropy = -sum(value * math.log(value) for value in distribution if value > 0)
    return entropy / math.log(len(distribution))


def jensen_shannon_divergence(
    distribution: list[float], global_distribution: list[float]
) -> float:
    """Calculate base-2 Jensen-Shannon divergence without SciPy."""
    first = np.asarray(distribution, dtype=np.float64)
    second = np.asarray(global_distribution, dtype=np.float64)
    midpoint = 0.5 * (first + second)

    def divergence(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return 0.5 * divergence(first) + 0.5 * divergence(second)


def distribution_records(
    category_ids: list[int],
    category_names: dict[int, str],
    counts: list[int],
    count_field: str,
    fraction_field: str,
) -> list[dict[str, Any]]:
    """Build JSON-ready count and fraction records in category-ID order."""
    fractions = normalized_distribution(counts)
    return [
        {
            "category_id": category_id,
            "category_name": category_names[category_id],
            count_field: counts[index],
            fraction_field: fractions[index],
        }
        for index, category_id in enumerate(category_ids)
    ]


def count_instances(
    image_ids: Sequence[int],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    category_ids: list[int],
) -> list[int]:
    """Count all object instances for a set of whole images."""
    positions = {category_id: index for index, category_id in enumerate(category_ids)}
    counts = [0] * len(category_ids)
    for image_id in image_ids:
        for annotation in annotations_by_image.get(image_id, []):
            counts[positions[annotation["category_id"]]] += 1
    return counts


def client_metrics(
    client_id: int,
    image_ids: list[int],
    primary_by_image: dict[int, int],
    inputs: Inputs,
    category_ids: list[int],
    category_names: dict[int, str],
    global_instance_distribution: list[float],
) -> dict[str, Any]:
    """Calculate complete image and instance metrics for one client."""
    primary_counter = Counter(primary_by_image[image_id] for image_id in image_ids)
    primary_counts = [primary_counter[category_id] for category_id in category_ids]
    instance_counts = count_instances(
        image_ids,
        inputs.annotations_by_image,
        category_ids,
    )
    instance_distribution = normalized_distribution(instance_counts)
    if sum(instance_counts):
        largest_index = min(
            range(len(category_ids)),
            key=lambda index: (-instance_counts[index], category_ids[index]),
        )
        largest_class_id: int | None = category_ids[largest_index]
        largest_class_name: str | None = category_names[largest_class_id]
        largest_share = instance_distribution[largest_index]
    else:
        largest_class_id = None
        largest_class_name = None
        largest_share = 0.0
    return {
        "client_id": client_id,
        "client_name": f"client_{client_id}",
        "image_ids": image_ids,
        "total_images": len(image_ids),
        "development_image_fraction": len(image_ids) / len(inputs.development_ids),
        "total_instances": sum(instance_counts),
        "primary_distribution": distribution_records(
            category_ids,
            category_names,
            primary_counts,
            "primary_image_count",
            "primary_image_fraction",
        ),
        "full_instance_distribution": distribution_records(
            category_ids,
            category_names,
            instance_counts,
            "instance_count",
            "instance_fraction",
        ),
        "primary_class_coverage": sum(count > 0 for count in primary_counts),
        "full_instance_class_coverage": sum(count > 0 for count in instance_counts),
        "largest_instance_class_id": largest_class_id,
        "largest_instance_class_name": largest_class_name,
        "largest_instance_class_share": largest_share,
        "normalized_entropy": normalized_entropy(instance_distribution),
        "jsd_to_global": jensen_shannon_divergence(
            instance_distribution,
            global_instance_distribution,
        ),
    }


def integrity_checks(
    clients: list[list[int]], development_ids: set[int], test_ids: set[int]
) -> dict[str, Any]:
    """Verify disjointness, coverage, and absence of global-test leakage."""
    flattened = [image_id for client in clients for image_id in client]
    assigned = set(flattened)
    cross_client_overlaps = len(flattened) - len(assigned)
    checks = {
        "assigned_image_occurrences": len(flattened),
        "assigned_unique_images": len(assigned),
        "each_development_image_exactly_once": (
            len(flattened) == len(development_ids) and assigned == development_ids
        ),
        "union_equals_development_ids": assigned == development_ids,
        "unassigned_images": len(development_ids - assigned),
        "unexpected_assigned_images": len(assigned - development_ids),
        "cross_client_overlaps": cross_client_overlaps,
        "global_test_overlaps": len(assigned & test_ids),
    }
    if not (
        checks["each_development_image_exactly_once"]
        and checks["union_equals_development_ids"]
        and checks["unassigned_images"] == 0
        and checks["unexpected_assigned_images"] == 0
        and checks["cross_client_overlaps"] == 0
        and checks["global_test_overlaps"] == 0
    ):
        raise AnalysisError(f"allocation integrity failure: {checks}")
    return checks


def aggregate_alpha_metrics(
    metrics: list[dict[str, Any]], checks: dict[str, Any]
) -> dict[str, Any]:
    """Summarize quantity skew and label-distribution skew for one alpha."""
    sizes = np.asarray([item["total_images"] for item in metrics], dtype=np.float64)
    entropies = np.asarray(
        [item["normalized_entropy"] for item in metrics], dtype=np.float64
    )
    divergences = np.asarray(
        [item["jsd_to_global"] for item in metrics], dtype=np.float64
    )
    minimum = int(sizes.min())
    maximum = int(sizes.max())
    mean = float(sizes.mean())
    return {
        "quantity_skew": {
            "min_client_images": minimum,
            "max_client_images": maximum,
            "mean_client_images": mean,
            "std_client_images": float(sizes.std()),
            "coefficient_of_variation_client_images": (
                float(sizes.std() / mean) if mean else None
            ),
            "max_min_client_size_ratio": maximum / minimum if minimum else None,
            "empty_clients": int(np.count_nonzero(sizes == 0)),
        },
        "label_distribution_skew": {
            "mean_normalized_entropy": float(entropies.mean()),
            "min_normalized_entropy": float(entropies.min()),
            "max_normalized_entropy": float(entropies.max()),
            "mean_jsd_to_global": float(divergences.mean()),
            "max_jsd_to_global": float(divergences.max()),
            "minimum_primary_class_coverage": min(
                item["primary_class_coverage"] for item in metrics
            ),
            "minimum_full_instance_class_coverage": min(
                item["full_instance_class_coverage"] for item in metrics
            ),
        },
        "integrity": {
            "total_unassigned_images": checks["unassigned_images"],
            "cross_client_overlaps": checks["cross_client_overlaps"],
            "global_test_overlaps": checks["global_test_overlaps"],
        },
    }


def analyze_alpha(
    alpha: float,
    ordered_strata: dict[int, list[int]],
    primary_by_image: dict[int, int],
    inputs: Inputs,
    category_ids: list[int],
    category_names: dict[int, str],
    global_instance_distribution: list[float],
) -> dict[str, Any]:
    """Allocate and measure one alpha candidate."""
    clients, proportions, allocation_counts = allocate_alpha(
        alpha,
        ordered_strata,
        category_ids,
    )
    checks = integrity_checks(clients, inputs.development_ids, inputs.test_ids)
    metrics = [
        client_metrics(
            client_id,
            image_ids,
            primary_by_image,
            inputs,
            category_ids,
            category_names,
            global_instance_distribution,
        )
        for client_id, image_ids in enumerate(clients)
    ]
    return {
        "alpha": alpha,
        "sampled_dirichlet_proportions_by_primary_class": proportions,
        "integer_allocation_counts_by_primary_class": allocation_counts,
        "clients": metrics,
        "aggregate_metrics": aggregate_alpha_metrics(metrics, checks),
        "integrity_checks": checks,
    }


def build_analysis(
    inputs: Inputs,
    annotations_path: Path,
    test_ids_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build the complete deterministic alpha-analysis document."""
    category_ids = [category["id"] for category in inputs.categories]
    category_names = {
        category["id"]: category["name"] for category in inputs.categories
    }
    primary_by_image, classes_by_image = primary_classes(inputs)
    class_cardinalities = [
        len(classes_by_image[image_id]) for image_id in sorted(inputs.development_ids)
    ]
    primary_counter = Counter(primary_by_image.values())
    global_primary_counts = [
        primary_counter[category_id] for category_id in category_ids
    ]
    global_instance_counts = count_instances(
        sorted(inputs.development_ids),
        inputs.annotations_by_image,
        category_ids,
    )
    global_instance_distribution = normalized_distribution(global_instance_counts)
    ordered_strata = ordered_primary_strata(primary_by_image, category_ids)
    alpha_results = [
        analyze_alpha(
            alpha,
            ordered_strata,
            primary_by_image,
            inputs,
            category_ids,
            category_names,
            global_instance_distribution,
        )
        for alpha in CANDIDATE_ALPHAS
    ]

    root = Path(__file__).resolve().parent.parent
    return {
        "method": METHOD,
        "method_note": METHOD_NOTE,
        "number_of_clients": NUMBER_OF_CLIENTS,
        "seed": SEED,
        "candidate_alphas": list(CANDIDATE_ALPHAS),
        "inputs": {
            "annotations_path": annotations_path.relative_to(root).as_posix(),
            "annotations_sha256": inputs.annotations_sha256,
            "test_image_ids_path": test_ids_path.relative_to(root).as_posix(),
            "test_image_ids_sha256": inputs.test_ids_sha256,
        },
        "output_path": output_path.relative_to(root).as_posix(),
        "classes": [
            {"id": category["id"], "name": category["name"]}
            for category in inputs.categories
        ],
        "counts": {
            "total_images": len(inputs.all_ids),
            "test_images": len(inputs.test_ids),
            "development_images": len(inputs.development_ids),
            "test_development_overlap": len(
                inputs.test_ids & inputs.development_ids
            ),
        },
        "primary_class_rule": PRIMARY_CLASS_RULE,
        "multi_label_diagnostics": {
            "images_with_zero_classes": sum(
                value == 0 for value in class_cardinalities
            ),
            "images_with_exactly_one_class": sum(
                value == 1 for value in class_cardinalities
            ),
            "images_with_two_or_more_classes": sum(
                value >= 2 for value in class_cardinalities
            ),
            "maximum_distinct_classes_per_image": max(class_cardinalities),
            "mean_distinct_classes_per_image": float(
                np.mean(class_cardinalities)
            ),
        },
        "global_development_distributions": {
            "primary_images": distribution_records(
                category_ids,
                category_names,
                global_primary_counts,
                "primary_image_count",
                "primary_image_fraction",
            ),
            "full_instances": distribution_records(
                category_ids,
                category_names,
                global_instance_counts,
                "instance_count",
                "instance_fraction",
            ),
        },
        "shared_shuffled_image_order_by_primary_class": {
            str(category_id): ordered_strata[category_id]
            for category_id in category_ids
        },
        "alpha_results": alpha_results,
    }


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """Write readable deterministic JSON atomically."""
    temporary = path.with_name(path.name + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AnalysisError(f"could not write analysis JSON: {exc}") from exc


def write_comparison_svg(path: Path, analysis: dict[str, Any]) -> None:
    """Plot normalized full-instance distributions for every client and alpha."""
    config_root = path.parent / ".matplotlib-cache"
    os.environ["MPLCONFIGDIR"] = str(config_root)
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "taco-niid-alpha-analysis-v1"
    import matplotlib.pyplot as plt

    classes = analysis["classes"]
    labels = [f"{item['id']}: {item['name']}" for item in classes]
    positions = np.arange(len(classes), dtype=np.float64)
    width = 0.16
    colors = plt.get_cmap("tab10").colors[:NUMBER_OF_CLIENTS]
    figure, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    for axis, result in zip(axes, analysis["alpha_results"], strict=True):
        for client_id, client in enumerate(result["clients"]):
            fractions = [
                item["instance_fraction"]
                for item in client["full_instance_distribution"]
            ]
            offset = (client_id - (NUMBER_OF_CLIENTS - 1) / 2) * width
            axis.bar(
                positions + offset,
                fractions,
                width=width,
                color=colors[client_id],
                label=(
                    f"client_{client_id} "
                    f"(n={client['total_images']} images)"
                ),
            )
        axis.set_title(f"alpha = {result['alpha']}")
        axis.set_xlabel("TACO-10 class")
        axis.set_xticks(positions, labels, rotation=50, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Within-client fraction of object instances")
    figure.suptitle(
        "TACO-10 candidate Dirichlet alphas: full instance distributions",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        "Label-distribution skew is normalized per client; legend reports "
        "quantity skew as image counts.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    temporary = path.with_name(path.name + ".part")
    try:
        figure.savefig(
            temporary,
            format="svg",
            metadata={"Creator": "taco_data", "Date": None},
        )
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AnalysisError(f"could not write comparison SVG: {exc}") from exc
    finally:
        plt.close(figure)
        if config_root.exists():
            shutil.rmtree(config_root)


def print_distribution(
    title: str,
    records: list[dict[str, Any]],
    count_field: str,
    fraction_field: str,
) -> None:
    """Print one global distribution table."""
    print(title)
    print("ID | Class | Count | Fraction")
    for record in records:
        print(
            f"{record['category_id']} | {record['category_name']} | "
            f"{record[count_field]} | {record[fraction_field]:.6f}"
        )


def print_report(analysis: dict[str, Any]) -> None:
    """Print stable human-readable development and alpha summaries."""
    counts = analysis["counts"]
    diagnostics = analysis["multi_label_diagnostics"]
    global_distributions = analysis["global_development_distributions"]
    print("=== Development Set ===")
    print(f"Total images       : {counts['total_images']}")
    print(f"Global test images : {counts['test_images']}")
    print(f"Development images : {counts['development_images']}")
    print(f"Test/dev overlap   : {counts['test_development_overlap']}")
    print(
        "Exactly 1 class    : "
        f"{diagnostics['images_with_exactly_one_class']}"
    )
    print(
        "2+ classes        : "
        f"{diagnostics['images_with_two_or_more_classes']}"
    )
    print(
        "Maximum classes   : "
        f"{diagnostics['maximum_distinct_classes_per_image']}"
    )
    print(
        "Mean classes      : "
        f"{diagnostics['mean_distinct_classes_per_image']:.6f}"
    )
    print_distribution(
        "\nPrimary-class images",
        global_distributions["primary_images"],
        "primary_image_count",
        "primary_image_fraction",
    )
    print_distribution(
        "\nFull object instances",
        global_distributions["full_instances"],
        "instance_count",
        "instance_fraction",
    )

    for result in analysis["alpha_results"]:
        print(f"\n=== alpha = {result['alpha']} ===")
        print(
            "Client | Images | Instances | Primary classes | Full classes | "
            "Norm. entropy | JSD global | Largest class share"
        )
        for client in result["clients"]:
            print(
                f"{client['client_name']} | {client['total_images']} | "
                f"{client['total_instances']} | "
                f"{client['primary_class_coverage']} | "
                f"{client['full_instance_class_coverage']} | "
                f"{client['normalized_entropy']:.6f} | "
                f"{client['jsd_to_global']:.6f} | "
                f"{client['largest_instance_class_name']} "
                f"({client['largest_instance_class_share']:.6f})"
            )
        aggregate = result["aggregate_metrics"]
        quantity = aggregate["quantity_skew"]
        label_skew = aggregate["label_distribution_skew"]
        print(
            "Quantity skew: "
            f"min={quantity['min_client_images']}, "
            f"max={quantity['max_client_images']}, "
            f"mean={quantity['mean_client_images']:.6f}, "
            f"std={quantity['std_client_images']:.6f}, "
            f"CV={quantity['coefficient_of_variation_client_images']:.6f}, "
            f"max/min={quantity['max_min_client_size_ratio']}, "
            f"empty={quantity['empty_clients']}"
        )
        print(
            "Label-distribution skew: "
            f"entropy mean/min/max={label_skew['mean_normalized_entropy']:.6f}/"
            f"{label_skew['min_normalized_entropy']:.6f}/"
            f"{label_skew['max_normalized_entropy']:.6f}, "
            f"JSD mean/max={label_skew['mean_jsd_to_global']:.6f}/"
            f"{label_skew['max_jsd_to_global']:.6f}, "
            f"coverage min primary/full="
            f"{label_skew['minimum_primary_class_coverage']}/"
            f"{label_skew['minimum_full_instance_class_coverage']}"
        )

    print("\n=== Alpha Comparison ===")
    print(
        "Alpha | Min/max images | Image CV | Mean entropy | Min entropy | "
        "Mean JSD | Max JSD | Min coverage primary/full"
    )
    for result in analysis["alpha_results"]:
        quantity = result["aggregate_metrics"]["quantity_skew"]
        label_skew = result["aggregate_metrics"]["label_distribution_skew"]
        print(
            f"{result['alpha']} | {quantity['min_client_images']}/"
            f"{quantity['max_client_images']} | "
            f"{quantity['coefficient_of_variation_client_images']:.6f} | "
            f"{label_skew['mean_normalized_entropy']:.6f} | "
            f"{label_skew['min_normalized_entropy']:.6f} | "
            f"{label_skew['mean_jsd_to_global']:.6f} | "
            f"{label_skew['max_jsd_to_global']:.6f} | "
            f"{label_skew['minimum_primary_class_coverage']}/"
            f"{label_skew['minimum_full_instance_class_coverage']}"
        )


def safe_relative_image_path(file_name: Any, raw_root: Path) -> Path:
    """Return a safe relative dataset path while preserving batch folders."""
    if not isinstance(file_name, str) or not file_name.strip() or "\x00" in file_name:
        raise AnalysisError(f"invalid image file_name: {file_name!r}")
    posix_path = PurePosixPath(file_name)
    windows_path = PureWindowsPath(file_name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise AnalysisError(f"unsafe image file_name: {file_name!r}")
    resolved = (raw_root / Path(file_name)).resolve()
    try:
        return resolved.relative_to(raw_root.resolve())
    except ValueError as exc:
        raise AnalysisError(f"image path escapes raw dataset: {file_name!r}") from exc


def render_yolo_label(
    image: dict[str, Any],
    annotations: Iterable[dict[str, Any]],
    category_indices: dict[int, int],
) -> str:
    """Render one deterministic YOLO label using the established formula."""
    width = image.get("width")
    height = image.get("height")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise AnalysisError(f"image {image.get('id')} has invalid dimensions")
    if width <= 0 or height <= 0:
        raise AnalysisError(f"image {image.get('id')} has non-positive dimensions")
    lines: list[str] = []
    for annotation in annotations:
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise AnalysisError(f"annotation {annotation.get('id')} has invalid bbox")
        x, y, box_width, box_height = bbox
        if any(not isinstance(value, (int, float)) for value in bbox):
            raise AnalysisError(
                f"annotation {annotation.get('id')} has nonnumeric bbox"
            )
        category_id = annotation["category_id"]
        lines.append(
            f"{category_indices[category_id]} "
            f"{(x + box_width / 2) / width:.6f} "
            f"{(y + box_height / 2) / height:.6f} "
            f"{box_width / width:.6f} "
            f"{box_height / height:.6f}\n"
        )
    return "".join(lines)


def remove_derived_tree(path: Path) -> None:
    """Remove only a derived tree, including read-only OneDrive entries."""
    if not path.exists():
        return

    def make_writable_and_retry(function: Any, name: str, _error: Any) -> None:
        os.chmod(name, stat.S_IWRITE)
        function(name)

    os.chmod(path, stat.S_IWRITE)
    shutil.rmtree(path, onerror=make_writable_and_retry)


def load_frozen_clients(
    analysis_path: Path, inputs: Inputs
) -> tuple[dict[str, Any], dict[str, Any], list[list[int]]]:
    """Load and validate the exact alpha=0.5 populations from analysis JSON."""
    analysis = load_json_object(analysis_path)
    recorded = analysis.get("inputs")
    if not isinstance(recorded, dict):
        raise AnalysisError("alpha analysis has no valid inputs record")
    if recorded.get("annotations_sha256") != inputs.annotations_sha256:
        raise AnalysisError("alpha analysis annotations hash does not match input")
    if recorded.get("test_image_ids_sha256") != inputs.test_ids_sha256:
        raise AnalysisError("alpha analysis test-ID hash does not match input")
    results = analysis.get("alpha_results")
    if not isinstance(results, list):
        raise AnalysisError("alpha analysis has no alpha_results list")
    selected = [result for result in results if result.get("alpha") == SELECTED_ALPHA]
    if len(selected) != 1:
        raise AnalysisError("alpha analysis must contain exactly one alpha=0.5 result")
    clients = selected[0].get("clients")
    if not isinstance(clients, list) or len(clients) != NUMBER_OF_CLIENTS:
        raise AnalysisError("selected alpha must contain exactly five clients")
    populations: list[list[int]] = []
    for expected_id, client in enumerate(clients):
        if client.get("client_id") != expected_id:
            raise AnalysisError("selected clients are not in canonical ID order")
        image_ids = client.get("image_ids")
        if not isinstance(image_ids, list) or any(
            not isinstance(image_id, int) for image_id in image_ids
        ):
            raise AnalysisError(f"client_{expected_id} has invalid image IDs")
        populations.append(list(image_ids))
    integrity_checks(populations, inputs.development_ids, inputs.test_ids)
    return analysis, selected[0], populations


def deterministic_splits(populations: list[list[int]]) -> list[dict[str, Any]]:
    """Create deterministic 80/20 splits independently within each client."""
    splits: list[dict[str, Any]] = []
    for client_id, population in enumerate(populations):
        shuffled = np.asarray(sorted(population), dtype=np.int64)
        np.random.default_rng(SEED + client_id).shuffle(shuffled)
        validation_count = max(1, round(len(population) * VALIDATION_FRACTION))
        if validation_count >= len(population):
            raise AnalysisError(
                f"client_{client_id} cannot have nonempty train and val"
            )
        validation = [int(value) for value in shuffled[:validation_count]]
        training = [int(value) for value in shuffled[validation_count:]]
        if (
            set(training) & set(validation)
            or set(training) | set(validation) != set(population)
        ):
            raise AnalysisError(f"invalid train/val split for client_{client_id}")
        splits.append(
            {
                "client_id": client_id,
                "seed": SEED + client_id,
                "all_image_ids": list(population),
                "train_image_ids": training,
                "val_image_ids": validation,
            }
        )
    return splits


def reuse_or_create_splits(
    manifest_path: Path,
    populations: list[list[int]],
    expected_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], bool]:
    """Reuse valid manifest splits or create them deterministically."""
    if not manifest_path.is_file():
        return deterministic_splits(populations), False
    manifest = load_json_object(manifest_path)
    if manifest.get("selected_alpha") != SELECTED_ALPHA:
        raise AnalysisError("existing manifest has a different selected alpha")
    if manifest.get("input_hashes") != expected_hashes:
        raise AnalysisError(
            "existing manifest input hashes do not match current inputs"
        )
    records = manifest.get("clients")
    if not isinstance(records, list) or len(records) != NUMBER_OF_CLIENTS:
        raise AnalysisError("existing manifest does not contain five clients")
    splits: list[dict[str, Any]] = []
    pairs = zip(records, populations, strict=True)
    for client_id, (record, population) in enumerate(pairs):
        all_ids = record.get("all_image_ids")
        train_ids = record.get("train_image_ids")
        val_ids = record.get("val_image_ids")
        if (
            all_ids != population
            or not isinstance(train_ids, list)
            or not isinstance(val_ids, list)
        ):
            raise AnalysisError(f"existing client_{client_id} population differs")
        if record.get("split_seed") != SEED + client_id:
            raise AnalysisError(f"existing client_{client_id} seed differs")
        if (
            set(train_ids) & set(val_ids)
            or set(train_ids) | set(val_ids) != set(population)
        ):
            raise AnalysisError(f"existing client_{client_id} split integrity failure")
        if len(val_ids) != max(1, round(len(population) * VALIDATION_FRACTION)):
            raise AnalysisError(f"existing client_{client_id} validation size differs")
        splits.append(
            {
                "client_id": client_id,
                "seed": SEED + client_id,
                "all_image_ids": list(all_ids),
                "train_image_ids": list(train_ids),
                "val_image_ids": list(val_ids),
            }
        )
    return splits, True


def yaml_content(categories: list[dict[str, Any]]) -> str:
    """Build the exact per-client relative-path YOLO configuration."""
    lines = ["train: images/train", "val: images/val", "names:"]
    for index, category in enumerate(categories):
        name = str(category["name"]).replace("'", "''")
        lines.append(f"  {index}: '{name}'")
    return "\n".join(lines) + "\n"


def build_materialization_plan(
    inputs: Inputs,
    splits: list[dict[str, Any]],
    raw_root: Path,
) -> list[dict[str, Any]]:
    """Resolve and collision-check every source and destination before writes."""
    images = require_record_list(inputs.document, "images")
    images_by_id = {image["id"]: image for image in images}
    category_indices = {
        category["id"]: index for index, category in enumerate(inputs.categories)
    }
    plan: list[dict[str, Any]] = []
    destinations: dict[str, list[int]] = defaultdict(list)
    for split in splits:
        client_id = split["client_id"]
        for split_name, key in (("train", "train_image_ids"), ("val", "val_image_ids")):
            for image_id in split[key]:
                image = images_by_id[image_id]
                relative_image = safe_relative_image_path(
                    image.get("file_name"), raw_root
                )
                source = raw_root / relative_image
                if not source.is_file():
                    raise AnalysisError(f"raw image is missing: {source}")
                relative_label = relative_image.with_suffix(".txt")
                image_destination = (
                    Path(f"client_{client_id}/images/{split_name}")
                    / relative_image
                )
                label_destination = (
                    Path(f"client_{client_id}/labels/{split_name}")
                    / relative_label
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
                        "label_content": render_yolo_label(
                            image,
                            inputs.annotations_by_image.get(image_id, []),
                            category_indices,
                        ),
                    }
                )
    collisions = {path: ids for path, ids in destinations.items() if len(ids) > 1}
    if collisions:
        raise AnalysisError(f"case-insensitive destination collisions: {collisions}")
    return plan


def validate_staging(plan: list[dict[str, Any]], staging: Path) -> None:
    """Validate exact staged files, bytes, labels, and YOLO line syntax."""
    expected_images = {item["image_destination"].as_posix() for item in plan}
    expected_labels = {item["label_destination"].as_posix() for item in plan}
    actual_images = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and "/images/" in f"/{path.relative_to(staging).as_posix()}"
    }
    actual_labels = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*.txt")
        if "/labels/" in f"/{path.relative_to(staging).as_posix()}"
    }
    if actual_images != expected_images or actual_labels != expected_labels:
        raise AnalysisError("staging file inventory differs from materialization plan")
    for item in plan:
        copied = staging / item["image_destination"]
        label = staging / item["label_destination"]
        if not filecmp.cmp(item["source"], copied, shallow=False):
            raise AnalysisError(f"copied image differs: {item['image_destination']}")
        actual = label.read_text(encoding="utf-8")
        if actual != item["label_content"]:
            raise AnalysisError(f"label differs: {item['label_destination']}")
        for line in actual.splitlines():
            fields = line.split()
            if len(fields) != 5:
                raise AnalysisError(f"invalid YOLO line in {item['label_destination']}")
            category = int(fields[0])
            values = [float(value) for value in fields[1:]]
            if category not in range(EXPECTED_CLASSES) or any(
                not math.isfinite(value) for value in values
            ):
                raise AnalysisError(
                    f"invalid YOLO value in {item['label_destination']}"
                )


def promote_clients(output_root: Path, staging: Path) -> None:
    """Atomically replace client directories, rolling all of them back on failure."""
    backups = [
        output_root / f".client_{client_id}.backup"
        for client_id in range(NUMBER_OF_CLIENTS)
    ]
    if any(path.exists() for path in backups):
        raise AnalysisError(
            "residual client backup detected; refusing unsafe promotion"
        )
    backed_up: list[int] = []
    promoted: list[int] = []
    try:
        for client_id in range(NUMBER_OF_CLIENTS):
            current = output_root / f"client_{client_id}"
            backup = backups[client_id]
            if current.exists():
                os.replace(current, backup)
                backed_up.append(client_id)
        for client_id in range(NUMBER_OF_CLIENTS):
            source = staging / f"client_{client_id}"
            destination = output_root / f"client_{client_id}"
            os.replace(source, destination)
            promoted.append(client_id)
    except OSError as exc:
        try:
            for client_id in reversed(promoted):
                remove_derived_tree(output_root / f"client_{client_id}")
            for client_id in backed_up:
                os.replace(backups[client_id], output_root / f"client_{client_id}")
        except OSError as rollback_error:
            raise AnalysisError(
                f"promotion failed ({exc}) and rollback failed ({rollback_error})"
            ) from rollback_error
        raise AnalysisError(
            f"promotion failed; previous clients restored: {exc}"
        ) from exc
    for backup in backups:
        remove_derived_tree(backup)
    remove_derived_tree(staging)


def materialization_metrics(
    image_ids: list[int], inputs: Inputs, category_ids: list[int]
) -> dict[str, Any]:
    """Count images, instances, full class distribution, and coverage for a split."""
    counts = count_instances(image_ids, inputs.annotations_by_image, category_ids)
    names = {category["id"]: category["name"] for category in inputs.categories}
    return {
        "images": len(image_ids),
        "instances": sum(counts),
        "full_instance_class_coverage": sum(value > 0 for value in counts),
        "full_instance_distribution": distribution_records(
            category_ids, names, counts, "instance_count", "instance_fraction"
        ),
    }


def build_manifest(
    inputs: Inputs,
    analysis: dict[str, Any],
    selected: dict[str, Any],
    splits: list[dict[str, Any]],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    """Build the deterministic scientific manifest for the materialized clients."""
    category_ids = [category["id"] for category in inputs.categories]
    selected_clients = selected["clients"]
    clients: list[dict[str, Any]] = []
    for split, analyzed in zip(splits, selected_clients, strict=True):
        total = materialization_metrics(split["all_image_ids"], inputs, category_ids)
        train = materialization_metrics(split["train_image_ids"], inputs, category_ids)
        val = materialization_metrics(split["val_image_ids"], inputs, category_ids)
        clients.append(
            {
                "client_id": split["client_id"],
                "client_name": f"client_{split['client_id']}",
                "split_seed": split["seed"],
                "total_images": total["images"],
                "train_images": train["images"],
                "val_images": val["images"],
                "all_image_ids": split["all_image_ids"],
                "train_image_ids": split["train_image_ids"],
                "val_image_ids": split["val_image_ids"],
                "total_instances": total["instances"],
                "train_instances": train["instances"],
                "val_instances": val["instances"],
                "primary_distribution": analyzed["primary_distribution"],
                "full_instance_distribution": total["full_instance_distribution"],
                "train_full_instance_distribution": train["full_instance_distribution"],
                "val_full_instance_distribution": val["full_instance_distribution"],
                "class_coverage": {
                    "primary": analyzed["primary_class_coverage"],
                    "full_instances": total["full_instance_class_coverage"],
                    "train_full_instances": train["full_instance_class_coverage"],
                    "val_full_instances": val["full_instance_class_coverage"],
                },
                "normalized_entropy": analyzed["normalized_entropy"],
                "jsd_to_global": analyzed["jsd_to_global"],
            }
        )
    candidate_comparison = [
        {"alpha": result["alpha"], "aggregate_metrics": result["aggregate_metrics"]}
        for result in analysis["alpha_results"]
    ]
    return {
        "schema_version": 1,
        "method": analysis["method"],
        "method_note": analysis["method_note"],
        "reference": "Hsu, Qi, and Brown (2019)",
        "adaptation_note": (
            "Object-detection adaptation: whole images are allocation units and "
            "all annotations in each allocated image are preserved."
        ),
        "selected_alpha": SELECTED_ALPHA,
        "candidate_alphas": analysis["candidate_alphas"],
        "number_of_clients": NUMBER_OF_CLIENTS,
        "seed": SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "client_split_seeds": {
            f"client_{client_id}": SEED + client_id
            for client_id in range(NUMBER_OF_CLIENTS)
        },
        "primary_class_rule": analysis["primary_class_rule"],
        "train_validation_split": {
            "train_fraction": 1.0 - VALIDATION_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "rounding": "val_count=max(1, round(client_images * 0.20))",
            "seed_rule": "42 + client_id",
        },
        "selection_justification": (
            "alpha=0.5 balances measurable heterogeneity with operational viability: "
            "it has image-count CV 0.544359, mean normalized entropy 0.765593, "
            "mean JSD 0.044516, minimum primary coverage 7/10, full-instance "
            "coverage 10/10, and no empty clients. alpha=0.1 is more concentrated "
            "(CV 0.621205, mean JSD 0.114799, minimum primary coverage 2/10), "
            "while alpha=1.0 is closer to the global distribution (mean normalized "
            "entropy 0.814188 and mean JSD 0.031109)."
        ),
        "quantity_skew_note": (
            "Client sizes are intentionally unequal because class-wise Dirichlet "
            "allocation induces both label-distribution and quantity skew."
        ),
        "input_hashes": input_hashes,
        "input_paths": analysis["inputs"],
        "counts": {
            "total_images": len(inputs.all_ids),
            "global_test_images": len(inputs.test_ids),
            "development_images": len(inputs.development_ids),
            "global_test_overlap": 0,
        },
        "classes": [
            {"id": item["id"], "name": item["name"]}
            for item in inputs.categories
        ],
        "candidate_alpha_comparison": candidate_comparison,
        "selected_alpha_aggregate_metrics": selected["aggregate_metrics"],
        "integrity": {
            **integrity_checks(
                [split["all_image_ids"] for split in splits],
                inputs.development_ids,
                inputs.test_ids,
            ),
            "all_clients_have_nonempty_train": all(
                split["train_image_ids"] for split in splits
            ),
            "all_clients_have_nonempty_val": all(
                split["val_image_ids"] for split in splits
            ),
            "train_val_overlaps": sum(
                len(set(split["train_image_ids"]) & set(split["val_image_ids"]))
                for split in splits
            ),
            "missing_files": 0,
            "missing_labels": 0,
            "destination_collisions": 0,
        },
        "clients": clients,
    }


def write_client_distribution_svg(path: Path, manifest: dict[str, Any]) -> None:
    """Write a deterministic normalized stacked instance-distribution chart."""
    config_root = path.parent / ".matplotlib-materialize-cache"
    os.environ["MPLCONFIGDIR"] = str(config_root)
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "taco-niid-selected-alpha-v1"
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 7))
    bottoms = np.zeros(NUMBER_OF_CLIENTS, dtype=np.float64)
    x = np.arange(NUMBER_OF_CLIENTS)
    for class_index, category in enumerate(manifest["classes"]):
        values = np.asarray(
            [
                client["full_instance_distribution"][class_index]["instance_fraction"]
                for client in manifest["clients"]
            ],
            dtype=np.float64,
        )
        label = f"{category['id']}: {category['name']}"
        axis.bar(x, values, bottom=bottoms, label=label)
        bottoms += values
    labels = [
        f"client_{client['client_id']}\nn={client['total_images']}"
        for client in manifest["clients"]
    ]
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Within-client fraction of object instances")
    axis.set_xlabel("Frozen alpha=0.5 client population")
    axis.set_title("Non-IID Dirichlet partition — alpha = 0.5")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    temporary = path.with_name(path.name + ".part")
    try:
        figure.savefig(
            temporary,
            format="svg",
            metadata={"Creator": "taco_data", "Date": None},
        )
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise AnalysisError(f"could not write client distribution SVG: {exc}") from exc
    finally:
        plt.close(figure)
        remove_derived_tree(config_root)


def materialize_clients(
    root: Path,
    inputs: Inputs,
    analysis_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Stage, validate, promote, document, and plot all frozen clients."""
    analysis, selected, populations = load_frozen_clients(analysis_path, inputs)
    input_hashes = {
        "annotations_regrouped_sha256": inputs.annotations_sha256,
        "test_image_ids_sha256": inputs.test_ids_sha256,
        "alpha_analysis_sha256": sha256_file(analysis_path),
    }
    manifest_path = output_root / "manifest_non_iid.json"
    splits, reused = reuse_or_create_splits(manifest_path, populations, input_hashes)
    raw_root = root / "data/raw/taco/extracted/TACO/data"
    plan = build_materialization_plan(inputs, splits, raw_root)
    staging = output_root / ".materialization_staging"
    remove_derived_tree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        configuration = yaml_content(inputs.categories)
        for client_id in range(NUMBER_OF_CLIENTS):
            client_root = staging / f"client_{client_id}"
            (client_root / "images/train").mkdir(parents=True)
            (client_root / "images/val").mkdir(parents=True)
            (client_root / "labels/train").mkdir(parents=True)
            (client_root / "labels/val").mkdir(parents=True)
            (client_root / "data.yaml").write_text(configuration, encoding="utf-8")
        for item in plan:
            image_destination = staging / item["image_destination"]
            label_destination = staging / item["label_destination"]
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["source"], image_destination)
            label_destination.write_text(item["label_content"], encoding="utf-8")
        validate_staging(plan, staging)
        promote_clients(output_root, staging)
    except (OSError, AnalysisError, KeyError, TypeError, ValueError):
        remove_derived_tree(staging)
        raise
    manifest = build_manifest(inputs, analysis, selected, splits, input_hashes)
    write_json_atomic(manifest_path, manifest)
    write_client_distribution_svg(output_root / "distribution_non_iid.svg", manifest)
    print("=== Frozen alpha=0.5 materialization ===")
    for client in manifest["clients"]:
        print(
            f"client_{client['client_id']}: total={client['total_images']}, "
            f"train={client['train_images']}, val={client['val_images']}, "
            f"instances={client['total_instances']}"
        )
    print(f"Existing deterministic splits reused: {reused}")
    print(f"Materialized images and labels: {len(plan)} each")
    return manifest


def run(args: argparse.Namespace) -> int:
    """Execute exactly one requested pipeline mode."""
    root = Path(__file__).resolve().parent.parent
    annotations_path = root / "data/processed/taco10/annotations_regrouped.json"
    test_ids_path = root / "data/test_global/test_image_ids.txt"
    output_root = root / "data/partitions/non_iid"
    json_path = output_root / "alpha_analysis.json"
    svg_path = output_root / "alpha_comparison.svg"
    inputs = validate_inputs(annotations_path, test_ids_path)
    if args.analyze:
        analysis = build_analysis(inputs, annotations_path, test_ids_path, json_path)
        write_json_atomic(json_path, analysis)
        write_comparison_svg(svg_path, analysis)
        print_report(analysis)
    elif args.materialize:
        materialize_clients(root, inputs, json_path, output_root)
    else:
        raise AnalysisError("no pipeline mode was selected")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Configure logging and return clean nonzero failures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(parse_args(argv))
    except AnalysisError as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

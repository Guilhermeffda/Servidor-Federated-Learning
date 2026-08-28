#!/usr/bin/env python3
"""Audit the extracted aggregate TACO COCO annotations without modifying data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence


LOGGER = logging.getLogger("taco-inspect")
SCENE_FIELD_NAMES = {
    "scene",
    "scenes",
    "scene_tag",
    "scene_tags",
    "background",
    "backgrounds",
    "tag",
    "tags",
}
IMAGE_CORE_FIELDS = ("id", "file_name", "width", "height")
ANNOTATION_CORE_FIELDS = (
    "id",
    "image_id",
    "category_id",
    "bbox",
    "segmentation",
    "area",
    "iscrowd",
)
CATEGORY_CORE_FIELDS = ("id", "name", "supercategory")
IMAGE_METADATA_FIELDS = (
    "flickr_url",
    "flickr_640_url",
    "date_captured",
    "license",
    "coco_url",
)


class AuditError(RuntimeError):
    """Indicate a structural or local error that prevents the audit."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit the extracted aggregate TACO COCO annotations."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help=(
            "diagnostic override for annotations.json; relative paths are "
            "resolved from taco_data"
        ),
    )
    return parser.parse_args(argv)


def relative_text(path: Path, root: Path) -> str:
    """Return a stable POSIX-style path relative to the workspace root."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_annotations(project_root: Path, override: Path | None) -> Path:
    """Resolve the confirmed aggregate annotations path or a safe override."""
    default = (
        project_root
        / "data"
        / "raw"
        / "taco"
        / "extracted"
        / "TACO"
        / "data"
        / "annotations.json"
    )
    candidate = default if override is None else override
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise AuditError("annotations path must remain inside taco_data") from exc
    if not candidate.is_file():
        raise AuditError(f"annotations file does not exist: {candidate}")
    return candidate


def load_coco(path: Path) -> dict[str, Any]:
    """Load and structurally validate the required COCO collections."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"could not load annotations JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise AuditError("annotations JSON root must be an object")
    for collection_name in ("images", "annotations", "categories"):
        collection = document.get(collection_name)
        if not isinstance(collection, list):
            raise AuditError(
                f"required collection {collection_name!r} must be an array"
            )
        if any(not isinstance(record, dict) for record in collection):
            raise AuditError(
                f"every item in {collection_name!r} must be an object"
            )
    return document


def value_key(value: Any) -> tuple[str, str]:
    """Create a hashable, type-aware key for an arbitrary JSON value."""
    return type(value).__name__, json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def display_value(value: Any) -> str:
    """Render a JSON value compactly for stable terminal output."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def duplicate_summary(values: Iterable[Any]) -> tuple[int, int]:
    """Return duplicated distinct values and occurrences beyond the first."""
    counts = Counter(value_key(value) for value in values)
    duplicated_values = sum(count > 1 for count in counts.values())
    excess_records = sum(count - 1 for count in counts.values() if count > 1)
    return duplicated_values, excess_records


def duplicate_details(values: Iterable[Any]) -> list[tuple[Any, int]]:
    """Return duplicated original values and their occurrence counts."""
    counts: Counter[tuple[str, str]] = Counter()
    originals: dict[tuple[str, str], Any] = {}
    for value in values:
        key = value_key(value)
        counts[key] += 1
        originals.setdefault(key, value)
    details = [
        (originals[key], count)
        for key, count in counts.items()
        if count > 1
    ]
    return sorted(details, key=lambda item: display_value(item[0]))


def safe_image_path(dataset_root: Path, file_name: Any) -> Path:
    """Resolve an external image file_name safely below dataset_root."""
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("missing or unusable file_name")
    if "\x00" in file_name:
        raise ValueError("file_name contains a null byte")
    posix_name = PurePosixPath(file_name)
    windows_name = PureWindowsPath(file_name)
    if (
        posix_name.is_absolute()
        or windows_name.is_absolute()
        or windows_name.drive
    ):
        raise ValueError("absolute or drive-qualified file_name")
    if ".." in posix_name.parts or ".." in windows_name.parts:
        raise ValueError("file_name contains path traversal")
    destination = (dataset_root / Path(file_name)).resolve()
    try:
        destination.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError("file_name escapes dataset root") from exc
    return destination


def audit_image_files(
    images: list[dict[str, Any]], dataset_root: Path
) -> dict[str, Any]:
    """Audit referenced image availability without opening image content."""
    available = 0
    missing_names: list[str] = []
    invalid_references: list[tuple[Any, str]] = []
    filesystem_errors: list[tuple[Any, str]] = []
    for image in images:
        file_name = image.get("file_name")
        try:
            destination = safe_image_path(dataset_root, file_name)
        except (ValueError, OSError) as exc:
            invalid_references.append((file_name, str(exc)))
            continue
        try:
            if destination.is_file() and destination.stat().st_size > 0:
                available += 1
            else:
                missing_names.append(file_name)
        except OSError as exc:
            filesystem_errors.append((file_name, str(exc)))
            missing_names.append(file_name)
    return {
        "available": available,
        "missing_names": missing_names,
        "invalid_references": invalid_references,
        "filesystem_errors": filesystem_errors,
    }


def field_presence(records: list[dict[str, Any]]) -> Counter[str]:
    """Count the presence of every observed field across all records."""
    return Counter(field for record in records for field in record)


def print_field_collection(
    label: str,
    records: list[dict[str, Any]],
    core_fields: tuple[str, ...],
) -> None:
    """Print observed field frequencies and highlight COCO core fields."""
    counts = field_presence(records)
    total = len(records)
    core = set(core_fields)
    print(f"{label}:")
    for field in sorted(set(counts) | core):
        suffix = " [COCO relevant]" if field in core else ""
        print(f"  {field}: {counts[field]} / {total}{suffix}")


def category_sort_key(category: dict[str, Any]) -> tuple[int, Any, str]:
    """Return a deterministic category ordering key."""
    identifier = category.get("id")
    if isinstance(identifier, (int, float)) and not isinstance(identifier, bool):
        return 0, identifier, str(category.get("name"))
    return 1, display_value(identifier), str(category.get("name"))


def inspect_scene_metadata(
    document: dict[str, Any], images: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize explicit scene evidence without visual inference."""
    direct_fields = sorted(
        field_presence(images).keys() & SCENE_FIELD_NAMES
    )
    direct: dict[str, dict[str, Any]] = {}
    for field in direct_fields:
        populated = [
            image[field]
            for image in images
            if field in image and image[field] not in (None, "", [], {})
        ]
        direct[field] = {
            "with_value": len(populated),
            "without_value": len(images) - len(populated),
            "distinct": sorted(
                {display_value(value) for value in populated}
            ),
            "multiple": sum(
                isinstance(value, (list, tuple, set)) and len(value) > 1
                for value in populated
            ),
        }

    scene_annotations = document.get("scene_annotations")
    scene_categories = document.get("scene_categories")
    result: dict[str, Any] = {
        "direct": direct,
        "structure_present": False,
    }
    if not (
        isinstance(scene_annotations, list)
        and all(isinstance(item, dict) for item in scene_annotations)
        and isinstance(scene_categories, list)
        and all(isinstance(item, dict) for item in scene_categories)
    ):
        return result

    image_ids = {value_key(image.get("id")) for image in images}
    category_names = {
        value_key(category.get("id")): category.get("name")
        for category in scene_categories
    }
    values_per_image: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    background_counts: Counter[tuple[str, str]] = Counter()
    records_with_multiple = 0
    invalid_background_values = 0
    for record in scene_annotations:
        image_key = value_key(record.get("image_id"))
        background_ids = record.get("background_ids")
        if not isinstance(background_ids, list):
            invalid_background_values += 1
            continue
        if len(background_ids) > 1:
            records_with_multiple += 1
        for background_id in background_ids:
            background_key = value_key(background_id)
            values_per_image[image_key].add(background_key)
            background_counts[background_key] += 1
    referenced_images = set(values_per_image)
    result.update(
        {
            "structure_present": True,
            "annotation_records": len(scene_annotations),
            "category_records": len(scene_categories),
            "images_with_values": len(referenced_images & image_ids),
            "images_without_values": len(image_ids - referenced_images),
            "orphan_image_ids": len(referenced_images - image_ids),
            "images_with_multiple_values": sum(
                len(values) > 1 for values in values_per_image.values()
            ),
            "records_with_multiple_values": records_with_multiple,
            "invalid_background_values": invalid_background_values,
            "background_counts": background_counts,
            "category_names": category_names,
        }
    )
    return result


def discover_auxiliary_artifacts(
    project_root: Path, taco_root: Path, canonical: Path
) -> dict[str, Any]:
    """Find auxiliary annotations and CSV files inside extracted TACO only."""
    annotations = sorted(
        (
            path
            for path in taco_root.rglob("annotations.json")
            if path.resolve() != canonical.resolve()
        ),
        key=lambda path: relative_text(path, project_root),
    )
    csv_files = sorted(
        taco_root.rglob("*.csv"),
        key=lambda path: relative_text(path, project_root),
    )
    macosx = taco_root.parent / "__MACOSX"
    return {
        "annotations": annotations,
        "csv_files": csv_files,
        "macosx_exists": macosx.is_dir(),
    }


def print_limited_examples(
    label: str, values: list[Any], limit: int = 20
) -> None:
    """Print a bounded deterministic list and indicate truncation."""
    if not values:
        return
    print(f"{label} (showing up to {limit}):")
    for value in values[:limit]:
        print(f"  - {value}")
    if len(values) > limit:
        print(f"  ... truncated; {len(values) - limit} additional item(s)")


def print_report(
    project_root: Path,
    annotations_path: Path,
    document: dict[str, Any],
) -> None:
    """Compute and print the complete deterministic audit report."""
    images = document["images"]
    annotations = document["annotations"]
    categories = document["categories"]
    dataset_root = annotations_path.parent
    taco_root = dataset_root.parent

    image_id_keys = [value_key(image.get("id")) for image in images]
    image_id_set = set(image_id_keys)
    category_id_keys = [value_key(category.get("id")) for category in categories]
    category_id_set = set(category_id_keys)
    annotation_image_keys = [
        value_key(annotation.get("image_id")) for annotation in annotations
    ]
    annotation_category_keys = [
        value_key(annotation.get("category_id")) for annotation in annotations
    ]
    image_files = audit_image_files(images, dataset_root)

    duplicate_file_names = duplicate_summary(
        image.get("file_name") for image in images
    )
    duplicate_image_ids = duplicate_summary(image.get("id") for image in images)
    duplicate_annotation_ids = duplicate_summary(
        annotation.get("id") for annotation in annotations
    )
    duplicate_category_ids = duplicate_summary(
        category.get("id") for category in categories
    )
    duplicate_category_names = duplicate_summary(
        category.get("name") for category in categories
    )
    annotated_image_ids = set(annotation_image_keys)
    images_without_annotations = sum(
        image_id not in annotated_image_ids for image_id in image_id_keys
    )
    orphan_image_references = [
        key for key in annotation_image_keys if key not in image_id_set
    ]
    orphan_category_references = [
        key for key in annotation_category_keys if key not in category_id_set
    ]
    annotation_counts = Counter(annotation_category_keys)

    print("=== Dataset ===")
    print(f"Annotations: {relative_text(annotations_path, project_root)}")
    print(f"Dataset root: {relative_text(dataset_root, project_root)}")
    print(f"Top-level keys: {', '.join(sorted(document))}")

    print("\n=== Images ===")
    print(f"Image records: {len(images)}")
    print(f"Locally available: {image_files['available']}")
    print(f"Missing locally: {len(image_files['missing_names'])}")
    print(f"Invalid path references: {len(image_files['invalid_references'])}")
    print(
        "Duplicated file_name values: "
        f"{duplicate_file_names[0]} "
        f"({duplicate_file_names[1]} excess record(s))"
    )
    print(f"Images without annotations: {images_without_annotations}")
    print(
        "Duplicated image ID values: "
        f"{duplicate_image_ids[0]} "
        f"({duplicate_image_ids[1]} excess record(s))"
    )
    print_limited_examples("Missing file_name values", image_files["missing_names"])
    invalid_examples = [
        f"{display_value(name)}: {reason}"
        for name, reason in image_files["invalid_references"]
    ]
    print_limited_examples("Invalid path references", invalid_examples)

    print("\n=== Annotations ===")
    print(f"Annotation records: {len(annotations)}")
    print(
        "Duplicated annotation ID values: "
        f"{duplicate_annotation_ids[0]} "
        f"({duplicate_annotation_ids[1]} excess record(s))"
    )
    annotation_id_duplicates = duplicate_details(
        annotation.get("id") for annotation in annotations
    )
    if annotation_id_duplicates:
        rendered_duplicates = ", ".join(
            f"{display_value(identifier)} ({count} occurrences)"
            for identifier, count in annotation_id_duplicates
        )
        print(f"Duplicated annotation IDs: {rendered_duplicates}")
    print(f"References to missing image IDs: {len(orphan_image_references)}")
    print(
        "Distinct missing image IDs referenced: "
        f"{len(set(orphan_image_references))}"
    )
    print(f"References to missing category IDs: {len(orphan_category_references)}")
    print(
        "Distinct missing category IDs referenced: "
        f"{len(set(orphan_category_references))}"
    )
    annotation_fields = field_presence(annotations)
    print(f"Annotations with bbox: {annotation_fields['bbox']}")
    print(f"Annotations with segmentation: {annotation_fields['segmentation']}")

    print("\n=== Categories ===")
    print(f"Category records: {len(categories)}")
    print(
        "Duplicated category ID values: "
        f"{duplicate_category_ids[0]} "
        f"({duplicate_category_ids[1]} excess record(s))"
    )
    print(
        "Duplicated category name values: "
        f"{duplicate_category_names[0]} "
        f"({duplicate_category_names[1]} excess record(s))"
    )
    print("ID | Category | Supercategory | Instances")
    for category in sorted(categories, key=category_sort_key):
        category_key = value_key(category.get("id"))
        print(
            f"{display_value(category.get('id'))} | "
            f"{category.get('name', '<missing>')} | "
            f"{category.get('supercategory', '<missing>')} | "
            f"{annotation_counts[category_key]}"
        )

    category_supercategories: Counter[str] = Counter()
    category_to_supercategory: dict[tuple[str, str], str] = {}
    for category in categories:
        supercategory = str(category.get("supercategory", "<missing>"))
        category_supercategories[supercategory] += 1
        category_to_supercategory.setdefault(
            value_key(category.get("id")), supercategory
        )
    instance_supercategories: Counter[str] = Counter()
    for annotation in annotations:
        category_key = value_key(annotation.get("category_id"))
        if category_key in category_to_supercategory:
            instance_supercategories[category_to_supercategory[category_key]] += 1

    print("\n=== Supercategories ===")
    print(f"Unique supercategories: {len(category_supercategories)}")
    print("Supercategory | Categories | Instances")
    for supercategory in sorted(category_supercategories):
        print(
            f"{supercategory} | {category_supercategories[supercategory]} | "
            f"{instance_supercategories[supercategory]}"
        )

    print("\n=== Observed fields ===")
    print_field_collection("Images", images, IMAGE_CORE_FIELDS)
    print_field_collection("Annotations", annotations, ANNOTATION_CORE_FIELDS)
    print_field_collection("Categories", categories, CATEGORY_CORE_FIELDS)

    print("\n=== Image metadata ===")
    image_field_counts = field_presence(images)
    for field in IMAGE_METADATA_FIELDS:
        print(f"{field}: {image_field_counts[field]} / {len(images)}")
    license_counts = Counter(
        display_value(image["license"])
        for image in images
        if "license" in image
    )
    if license_counts:
        print("Observed image-record license values:")
        for value, count in sorted(license_counts.items()):
            print(f"  {value}: {count}")
    else:
        print("No license values found directly in image records.")

    scene = inspect_scene_metadata(document, images)
    print("\n=== Scene metadata ===")
    if not scene["direct"]:
        print(
            "No explicit scene/background tag fields found in the aggregate "
            "COCO image records."
        )
    else:
        for field, summary in scene["direct"].items():
            print(
                f"Image field {field}: with value={summary['with_value']}, "
                f"without value={summary['without_value']}, "
                f"multiple values={summary['multiple']}"
            )
            print(f"  Distinct values: {', '.join(summary['distinct'])}")
    if scene["structure_present"]:
        print("Top-level scene_annotations/scene_categories found.")
        print(f"Scene annotation records: {scene['annotation_records']}")
        print(f"Scene category records: {scene['category_records']}")
        print(f"Images with scene values: {scene['images_with_values']}")
        print(f"Images without scene values: {scene['images_without_values']}")
        print(f"Orphan scene image IDs: {scene['orphan_image_ids']}")
        print(
            "Images with multiple distinct scene/background IDs: "
            f"{scene['images_with_multiple_values']}"
        )
        print(
            "Scene records containing multiple background IDs: "
            f"{scene['records_with_multiple_values']}"
        )
        print(
            "Scene records with invalid background_ids structure: "
            f"{scene['invalid_background_values']}"
        )
        print("Scene/background ID | Name | Occurrences")
        all_background_keys = sorted(
            scene["background_counts"], key=lambda key: (key[0], key[1])
        )
        for background_key in all_background_keys:
            name = scene["category_names"].get(background_key, "<unmatched>")
            print(
                f"{background_key[1]} | {name} | "
                f"{scene['background_counts'][background_key]}"
            )
    else:
        print("No usable top-level scene annotation structure found.")

    auxiliary = discover_auxiliary_artifacts(
        project_root, taco_root, annotations_path
    )
    print("\n=== Auxiliary artifacts ===")
    print(f"__MACOSX exists: {auxiliary['macosx_exists']}")
    print(
        "Other annotations.json files inside TACO: "
        f"{len(auxiliary['annotations'])}"
    )
    for path in auxiliary["annotations"]:
        print(f"  - {relative_text(path, project_root)}")
    print(f"CSV files inside TACO: {len(auxiliary['csv_files'])}")
    for path in auxiliary["csv_files"]:
        print(f"  - {relative_text(path, project_root)}")

    print("\n=== Integrity observations ===")
    print(f"Missing image files: {len(image_files['missing_names'])}")
    print(f"Unsafe/invalid image references: {len(image_files['invalid_references'])}")
    print(f"Image filesystem errors: {len(image_files['filesystem_errors'])}")
    print(f"Duplicate image IDs: {duplicate_image_ids[0]}")
    print(f"Duplicate annotation IDs: {duplicate_annotation_ids[0]}")
    print(f"Duplicate category IDs: {duplicate_category_ids[0]}")
    print(f"Orphan annotation image references: {len(orphan_image_references)}")
    print(
        "Orphan annotation category references: "
        f"{len(orphan_category_references)}"
    )


def run(args: argparse.Namespace) -> int:
    """Resolve inputs, load the aggregate COCO file, and print its audit."""
    project_root = Path(__file__).resolve().parent.parent
    annotations_path = resolve_annotations(project_root, args.annotations)
    document = load_coco(annotations_path)
    print_report(project_root, annotations_path, document)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Configure logging and convert expected failures to a nonzero exit."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(parse_args(argv))
    except AuditError as exc:
        LOGGER.error("%s", exc)
        return 1
    except OSError as exc:
        LOGGER.error("local filesystem error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

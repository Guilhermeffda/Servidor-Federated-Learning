#!/usr/bin/env python3
"""Analyze scarce non-IID train classes and preview native augmentations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REQUIRED_ULTRALYTICS = "8.4.121"
SELECTED_ALPHA = 0.5
SEED = 42
MAX_PREVIEWS = 12
PARAMETERS = {
    "hsv_h": 0.015,
    "hsv_s": 0.50,
    "hsv_v": 0.30,
    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.25,
    "flipud": 0.0,
    "fliplr": 0.50,
    "mosaic": 0.50,
}
SEVERITY_ORDER = {"absent": 0, "critical": 1, "scarce": 2}


class StrategyError(RuntimeError):
    """Indicate an invalid frozen input or preview-generation failure."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the small, parameter-free command-line interface."""
    parser = argparse.ArgumentParser(
        description="Prepare and preview native Ultralytics augmentations."
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise StrategyError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object with an explicit error."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StrategyError(f"could not load {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise StrategyError(f"{path} root must be an object")
    return document


def validate_manifest(
    manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Validate fields needed for train-only scarcity analysis."""
    if manifest.get("selected_alpha") != SELECTED_ALPHA:
        raise StrategyError("manifest selected_alpha must be 0.5")
    if manifest.get("number_of_clients") != 5:
        raise StrategyError("manifest must contain five clients")
    classes = manifest.get("classes")
    if not isinstance(classes, list) or len(classes) != 10:
        raise StrategyError("manifest must describe ten TACO-10 classes")
    class_names: dict[int, str] = {}
    for expected_id, category in enumerate(classes):
        if not isinstance(category, dict) or category.get("id") != expected_id:
            raise StrategyError("manifest category IDs must be exactly 0 through 9")
        name = category.get("name")
        if not isinstance(name, str) or not name:
            raise StrategyError("manifest category names must be nonempty strings")
        class_names[expected_id] = name
    clients = manifest.get("clients")
    if not isinstance(clients, list) or len(clients) != 5:
        raise StrategyError("manifest clients must be a five-item list")
    for client_id, client in enumerate(clients):
        if not isinstance(client, dict) or client.get("client_id") != client_id:
            raise StrategyError("manifest clients must be in canonical ID order")
        train_ids = client.get("train_image_ids")
        distribution = client.get("train_full_instance_distribution")
        if not isinstance(train_ids, list) or any(
            not isinstance(image_id, int) for image_id in train_ids
        ):
            raise StrategyError(f"client_{client_id} train IDs are invalid")
        if not isinstance(distribution, list) or len(distribution) != 10:
            raise StrategyError(f"client_{client_id} train distribution is invalid")
        for category_id, record in enumerate(distribution):
            if (
                not isinstance(record, dict)
                or record.get("category_id") != category_id
                or not isinstance(record.get("instance_count"), int)
                or record["instance_count"] < 0
            ):
                raise StrategyError(
                    f"client_{client_id} train class {category_id} is invalid"
                )
    return clients, class_names


def scarce_pairs(
    clients: list[dict[str, Any]], class_names: dict[int, str]
) -> list[dict[str, Any]]:
    """Classify train-only client/class pairs under the frozen policy."""
    pairs: list[dict[str, Any]] = []
    for client in clients:
        for record in client["train_full_instance_distribution"]:
            count = record["instance_count"]
            if count == 0:
                severity = "absent"
            elif count <= 9:
                severity = "critical"
            elif count <= 15:
                severity = "scarce"
            else:
                continue
            category_id = record["category_id"]
            pairs.append(
                {
                    "severity": severity,
                    "instances": count,
                    "client_id": client["client_id"],
                    "category_id": category_id,
                    "category_name": class_names[category_id],
                }
            )
    return sorted(
        pairs,
        key=lambda item: (
            SEVERITY_ORDER[item["severity"]],
            item["instances"],
            item["client_id"],
            item["category_id"],
        ),
    )


def load_image_metadata(root: Path) -> tuple[dict[int, str], dict[int, set[int]]]:
    """Load paths and class sets solely to resolve frozen train preview IDs."""
    document = load_json(root / "data/processed/taco10/annotations_regrouped.json")
    images = document.get("images")
    annotations = document.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise StrategyError("TACO-10 metadata collections are invalid")
    paths: dict[int, str] = {}
    classes: dict[int, set[int]] = defaultdict(set)
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("id"), int):
            raise StrategyError("invalid TACO-10 image metadata")
        if not isinstance(image.get("file_name"), str):
            raise StrategyError("invalid TACO-10 image file_name")
        paths[image["id"]] = image["file_name"]
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise StrategyError("invalid TACO-10 annotation metadata")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id not in paths or category_id not in range(10):
            raise StrategyError("invalid TACO-10 annotation reference")
        classes[image_id].add(category_id)
    return paths, dict(classes)


def select_previews(
    clients: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    classes_by_image: dict[int, set[int]],
) -> list[dict[str, Any]]:
    """Greedily cover critical then scarce pairs with deterministic train images."""
    targets = {
        (pair["client_id"], pair["category_id"]): pair["severity"]
        for pair in pairs
        if pair["severity"] != "absent"
    }
    candidates: list[dict[str, Any]] = []
    for client in clients:
        client_id = client["client_id"]
        for image_id in sorted(client["train_image_ids"]):
            covered = {
                (client_id, category_id)
                for category_id in classes_by_image.get(image_id, set())
                if (client_id, category_id) in targets
            }
            if covered:
                candidates.append(
                    {
                        "client_id": client_id,
                        "image_id": image_id,
                        "covers": covered,
                    }
                )
    random.Random(SEED).shuffle(candidates)
    remaining = set(targets)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < MAX_PREVIEWS:
        best = max(
            candidates,
            key=lambda item: (
                sum(targets[pair] == "critical" for pair in item["covers"] & remaining),
                len(item["covers"] & remaining),
            ),
        )
        newly_covered = best["covers"] & remaining
        if not newly_covered:
            break
        selected.append(
            {
                "client_id": best["client_id"],
                "image_id": best["image_id"],
                "covered_pairs": sorted(newly_covered),
            }
        )
        remaining -= newly_covered
        candidates.remove(best)
    return selected


def strategy_yaml() -> str:
    """Return the fixed candidate configuration with methodological comments."""
    return """# Moderate HSV variation for plausible lighting and color changes.
hsv_h: 0.015
hsv_s: 0.50
hsv_v: 0.30
# Low rotation and translation preserve urban-scene geometry.
degrees: 5.0
translate: 0.05
# Conservative scale because many litter objects are small.
scale: 0.25
# Vertical flips are implausible for urban scenes; horizontal flips remain valid.
flipud: 0.0
fliplr: 0.50
# Moderate Mosaic trades visual diversity against apparent small-object size.
mosaic: 0.50
"""


def validate_ultralytics() -> tuple[str, Any]:
    """Confirm the installed version and native recognition of every parameter."""
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG, get_cfg

    if ultralytics.__version__ != REQUIRED_ULTRALYTICS:
        raise StrategyError(
            f"Ultralytics {REQUIRED_ULTRALYTICS} required; "
            f"found {ultralytics.__version__}"
        )
    try:
        configuration = get_cfg(DEFAULT_CFG, overrides=PARAMETERS)
    except (KeyError, TypeError, ValueError) as exc:
            raise StrategyError(
                f"Ultralytics rejected augmentation parameters: {exc}"
            ) from exc
    for name, expected in PARAMETERS.items():
        if getattr(configuration, name, None) != expected:
            raise StrategyError(f"Ultralytics did not retain parameter {name}")
    return ultralytics.__version__, configuration


def remove_tree(path: Path) -> None:
    """Remove a derived preview/cache tree, including read-only entries."""
    if not path.exists():
        return

    def make_writable(function: Any, name: str, _error: Any) -> None:
        os.chmod(name, stat.S_IWRITE)
        function(name)

    os.chmod(path, stat.S_IWRITE)
    shutil.rmtree(path, onerror=make_writable)


def remove_loader_caches(client_root: Path) -> None:
    """Remove only cache files generated transiently by YOLODataset."""
    for path in client_root.rglob("*.cache"):
        if path.is_file():
            try:
                path.chmod(stat.S_IWRITE)
                path.unlink()
            except OSError as exc:
                raise StrategyError(
                    f"could not remove loader cache {path}: {exc}"
                ) from exc


def generate_previews(
    root: Path,
    selected: list[dict[str, Any]],
    image_paths: dict[int, str],
    class_names: dict[int, str],
    configuration: Any,
) -> list[str]:
    """Create original/augmented sheets through native YOLODataset transforms."""
    import torch
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.plotting import plot_images

    output_root = root / "data/partitions/non_iid/augmentation_previews"
    staging = root / "data/partitions/non_iid/.augmentation_preview_staging"
    remove_tree(staging)
    staging.mkdir(parents=True)
    generated: list[str] = []
    by_client: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for selection in selected:
        by_client[selection["client_id"]].append(selection)
    try:
        for client_id in sorted(by_client):
            client_root = root / f"data/partitions/non_iid/client_{client_id}"
            checked = check_det_dataset(
                str(client_root / "data.yaml"), autodownload=False
            )
            original_dataset = YOLODataset(
                img_path=checked["train"],
                data=checked,
                task="detect",
                augment=False,
                cache=False,
                prefix=f"preview client_{client_id} original: ",
            )
            augmented_dataset = YOLODataset(
                img_path=checked["train"],
                data=checked,
                task="detect",
                augment=True,
                hyp=configuration,
                cache=False,
                prefix=f"preview client_{client_id} augmented: ",
            )
            indices = {
                Path(path).resolve().as_posix().casefold(): index
                for index, path in enumerate(original_dataset.im_files)
            }
            for selection in by_client[client_id]:
                image_id = selection["image_id"]
                expected = (
                    client_root / "images/train" / image_paths[image_id]
                ).resolve()
                key = expected.as_posix().casefold()
                if key not in indices:
                    raise StrategyError(
                        f"client_{client_id} train image {image_id} "
                        "was not discovered"
                    )
                index = indices[key]
                random.seed(SEED + image_id)
                np.random.seed(SEED + image_id)
                torch.manual_seed(SEED + image_id)
                original = original_dataset[index]
                random.seed(SEED + image_id)
                np.random.seed(SEED + image_id)
                torch.manual_seed(SEED + image_id)
                augmented = augmented_dataset[index]
                labels: dict[str, Any] = {}
                labels["cls"] = torch.cat((original["cls"], augmented["cls"]), 0)
                labels["bboxes"] = torch.cat(
                    (original["bboxes"], augmented["bboxes"]), 0
                )
                labels["batch_idx"] = torch.cat(
                    (
                        torch.zeros_like(original["batch_idx"]),
                        torch.ones_like(augmented["batch_idx"]),
                    ),
                    0,
                )
                images = torch.stack((original["img"], augmented["img"]), 0)
                critical_names = [
                    class_names[category_id]
                    for pair_client, category_id in selection["covered_pairs"]
                    if pair_client == client_id
                ]
                file_name = f"client_{client_id}_image_{image_id}.jpg"
                result = plot_images(
                    labels,
                    images=images,
                    paths=[
                        f"original | client_{client_id} | image_id={image_id}",
                        "native augmented | " + ", ".join(critical_names),
                    ],
                    fname=str(staging / file_name),
                    names=class_names,
                    save=True,
                    show_conf=False,
                )
                if hasattr(result, "join"):
                    result.join()
                if not (staging / file_name).is_file():
                    raise StrategyError(f"native preview was not written: {file_name}")
                generated.append(file_name)
            remove_loader_caches(client_root)
        remove_tree(output_root)
        os.replace(staging, output_root)
    except (OSError, KeyError, TypeError, ValueError, StrategyError):
        remove_tree(staging)
        raise
    return generated


def report_text(
    manifest_hash: str,
    version: str,
    pairs: list[dict[str, Any]],
    previews: list[str],
) -> str:
    """Build the deterministic Markdown report."""
    counts = Counter(pair["severity"] for pair in pairs)
    clients = sorted({pair["client_id"] for pair in pairs})
    classes = sorted({pair["category_name"] for pair in pairs})
    lines = [
        "# Estratégia de data augmentation pós-Dirichlet",
        "",
        f"- SHA-256 do manifesto non-IID: `{manifest_hash}`",
        f"- Ultralytics: `{version}`",
        "- Selected alpha: `0.5`",
        "- Fonte da análise de escassez: somente `train_full_instance_distribution`.",
        "- Política: absent = 0; critical = 1–9; scarce = 10–15 instâncias.",
        "- Pares absent/critical/scarce: "
        f"{counts['absent']}/{counts['critical']}/{counts['scarce']}",
        "- Clientes envolvidos: "
        f"{', '.join(f'client_{item}' for item in clients) or 'nenhum'}",
        f"- Classes envolvidas: {', '.join(classes) or 'nenhuma'}",
        "",
        "## Combinações sinalizadas",
        "",
        "| Severity | Instâncias | Cliente | ID | Classe TACO-10 |",
        "|---|---:|---:|---:|---|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair['severity']} | {pair['instances']} | "
            f"client_{pair['client_id']} | {pair['category_id']} | "
            f"{pair['category_name']} |"
        )
    lines.extend(
        [
            "",
            "## Configuração candidata",
            "",
            "```yaml",
            strategy_yaml().rstrip(),
            "```",
            "",
            "HSV moderado amplia variações plausíveis de iluminação e cor. A "
            "rotação e translação são baixas. A escala é conservadora devido aos "
            "objetos pequenos. Flip vertical fica desabilitado para cenas urbanas "
            "e o horizontal permanece permitido. Mosaic moderado equilibra "
            "diversidade visual com o risco de reduzir o tamanho aparente de "
            "objetos pequenos.",
            "",
            "Os parâmetros foram aceitos pela configuração nativa do Ultralytics "
            "8.4.121 e não são ajustados automaticamente.",
            "",
            "## Previews nativos",
            "",
            "Os previews usam `YOLODataset` com `augment=True`, a configuração "
            "candidata e `plot_images` para desenhar as bounding boxes originais "
            "e transformadas. Nenhum modelo, treino ou peso é utilizado.",
            "",
        ]
    )
    lines.extend(f"- `augmentation_previews/{name}`" for name in previews)
    lines.extend(
        [
            "",
            "## Inspeção visual",
            "",
            "Verificar se as boxes continuam alinhadas; objetos pequenos permanecem "
            "reconhecíveis; cortes não são excessivos; scale não reduz o lixo a "
            "tamanho impraticável; Mosaic preserva instâncias pequenas; cores e "
            "brilho permanecem plausíveis; flips são semanticamente plausíveis; "
            "e não há artefatos graves.",
            "",
            "**visual_validation_status: pending_review**",
            "",
            "## Limitações conhecidas",
            "",
            "1. Augmentation não corrige ausência total de uma classe.",
            "2. Augmentation não equivale à aquisição de novos exemplos reais.",
            "3. A configuração é global ao pipeline de treino, não class-specific.",
            "4. Classes extremamente raras continuam sujeitas a alta variância.",
            "5. Mosaic pode reduzir o tamanho aparente de objetos pequenos e "
            "depende de validação visual.",
            "6. As augmentations atuam nas imagens amostradas durante o treino, não "
            "alteram IDs e não aumentam por si sós a frequência de amostragem de "
            "classes.",
            "7. Exemplos raros continuam com baixa diversidade semântica; a estratégia "
            "aumenta diversidade visual, mas não rebalanceia classes.",
            "",
        ]
    )
    return "\n".join(lines)


def write_atomic(path: Path, content: str) -> None:
    """Write deterministic UTF-8 text atomically."""
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise StrategyError(f"could not write {path}: {exc}") from exc


def run(_args: argparse.Namespace) -> int:
    """Create configuration, scarcity report, and native preview sheets."""
    root = Path(__file__).resolve().parent.parent
    output_root = root / "data/partitions/non_iid"
    config_root = output_root / ".ultralytics-augmentation-config"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_root)
    os.environ["MPLCONFIGDIR"] = str(output_root / ".matplotlib-augmentation-cache")
    manifest_path = output_root / "manifest_non_iid.json"
    manifest = load_json(manifest_path)
    clients, class_names = validate_manifest(manifest)
    manifest_hash = sha256_file(manifest_path)
    pairs = scarce_pairs(clients, class_names)
    image_paths, classes_by_image = load_image_metadata(root)
    selected = select_previews(clients, pairs, classes_by_image)
    version, configuration = validate_ultralytics()
    previews = generate_previews(
        root, selected, image_paths, class_names, configuration
    )
    write_atomic(output_root / "augmentation_strategy.yaml", strategy_yaml())
    write_atomic(
        output_root / "augmentation_report.md",
        report_text(manifest_hash, version, pairs, previews),
    )
    remove_tree(output_root / ".matplotlib-augmentation-cache")
    remove_tree(config_root)
    counts = Counter(pair["severity"] for pair in pairs)
    print(f"Ultralytics: {version}")
    print(f"Manifest SHA-256: {manifest_hash}")
    print(
        "Pairs absent/critical/scarce: "
        f"{counts['absent']}/{counts['critical']}/{counts['scarce']}"
    )
    print(f"Native preview sheets: {len(previews)}")
    print("visual_validation_status: pending_review")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Return a clean nonzero status for expected failures."""
    try:
        return run(parse_args(argv))
    except StrategyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

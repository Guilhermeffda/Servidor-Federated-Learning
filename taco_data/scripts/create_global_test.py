from __future__ import annotations

import filecmp
import json
import os
import shutil
import stat
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS_FILE = (
    ROOT / "data/processed/taco10/annotations_regrouped.json"
)

RAW_DATA_ROOT = (
    ROOT / "data/raw/taco/extracted/TACO/data"
)

OUTPUT_ROOT = ROOT / "data/test_global"
TEST_IDS_FILE = OUTPUT_ROOT / "test_image_ids.txt"
IMAGES_OUTPUT = OUTPUT_ROOT / "images"
LABELS_OUTPUT = OUTPUT_ROOT / "labels"
IMAGES_STAGING = OUTPUT_ROOT / ".images.staging"
LABELS_STAGING = OUTPUT_ROOT / ".labels.staging"
IMAGES_BACKUP = OUTPUT_ROOT / ".images.backup"
LABELS_BACKUP = OUTPUT_ROOT / ".labels.backup"

TEST_SIZE = 0.20
RANDOM_STATE = 42


class PipelineError(RuntimeError):
    """Indica uma falha que impede materializar o teste global."""


def load_dataset():
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_categories(dataset):
    """
    Define uma categoria principal para cada imagem.

    Como uma imagem pode conter várias categorias, utiliza-se a
    categoria com maior número de instâncias naquela imagem.
    Em caso de empate, utiliza-se a categoria de menor ID.
    """
    categories = {
        category["id"]: category["name"]
        for category in dataset["categories"]
    }

    image_annotations = {
        image["id"]: []
        for image in dataset["images"]
    }

    for annotation in dataset["annotations"]:
        image_id = annotation["image_id"]

        if image_id in image_annotations:
            image_annotations[image_id].append(
                annotation["category_id"]
            )

    image_primary_category = {}

    for image_id, category_ids in image_annotations.items():
        if not category_ids:
            image_primary_category[image_id] = "Other"
            continue

        counts = {}

        for category_id in category_ids:
            counts[category_id] = counts.get(category_id, 0) + 1

        primary_category_id = min(
            counts,
            key=lambda category_id: (-counts[category_id], category_id)
        )

        image_primary_category[image_id] = categories[
            primary_category_id
        ]

    return image_primary_category


def can_stratify(labels):
    """
    Verifica se cada categoria possui pelo menos duas imagens,
    requisito mínimo para uma divisão estratificada.
    """
    counts = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    return all(count >= 2 for count in counts.values())


def load_authoritative_test_ids(
    path: Path,
    all_ids: set[int],
    expected_size: int,
) -> set[int]:
    """Carrega e valida o conjunto de teste já materializado."""
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed_ids = []

    for line_number, line in enumerate(lines, start=1):
        value = line.strip()
        if not value:
            continue
        try:
            parsed_ids.append(int(value))
        except ValueError as exc:
            raise PipelineError(
                f"ID inválido na linha {line_number} de {path}: {value!r}"
            ) from exc

    test_ids = set(parsed_ids)
    if len(test_ids) != len(parsed_ids):
        duplicates = sorted(
            image_id
            for image_id in test_ids
            if parsed_ids.count(image_id) > 1
        )
        raise PipelineError(f"IDs de teste duplicados: {duplicates}")
    if len(test_ids) != expected_size:
        raise PipelineError(
            f"Tamanho inesperado do teste: {len(test_ids)} "
            f"(esperado: {expected_size})"
        )
    unknown_ids = sorted(test_ids - all_ids)
    if unknown_ids:
        raise PipelineError(
            f"IDs de teste inexistentes no dataset: {unknown_ids}"
        )
    return test_ids


def select_test_ids(dataset):
    """Reutiliza IDs autoritativos ou cria o split quando ainda não existe."""
    image_ids = [image["id"] for image in dataset["images"]]
    all_ids = set(image_ids)
    if len(all_ids) != len(image_ids):
        raise PipelineError("O dataset contém IDs de imagem duplicados")
    expected_size = round(len(image_ids) * TEST_SIZE)

    if TEST_IDS_FILE.is_file():
        test_ids = load_authoritative_test_ids(
            TEST_IDS_FILE,
            all_ids,
            expected_size,
        )
        return test_ids, None, False

    image_primary_category = build_image_categories(dataset)
    labels = [image_primary_category[image_id] for image_id in image_ids]
    stratified = can_stratify(labels)
    split_arguments = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    if stratified:
        split_arguments["stratify"] = labels
    _, selected_ids = train_test_split(image_ids, **split_arguments)
    test_ids = set(selected_ids)
    if len(test_ids) != expected_size:
        raise PipelineError(
            f"Tamanho inesperado do teste: {len(test_ids)} "
            f"(esperado: {expected_size})"
        )
    return test_ids, stratified, True


def safe_relative_image_path(file_name: Any) -> Path:
    """Valida file_name e preserva seu caminho relativo ao dataset."""
    if not isinstance(file_name, str) or not file_name.strip():
        raise PipelineError(f"file_name inválido: {file_name!r}")
    if "\x00" in file_name:
        raise PipelineError("file_name contém byte nulo")
    posix_path = PurePosixPath(file_name)
    windows_path = PureWindowsPath(file_name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise PipelineError(f"file_name inseguro: {file_name!r}")
    source = (RAW_DATA_ROOT / Path(file_name)).resolve()
    try:
        return source.relative_to(RAW_DATA_ROOT.resolve())
    except ValueError as exc:
        raise PipelineError(
            f"file_name escapa do dataset raw: {file_name!r}"
        ) from exc


def render_yolo_label(
    image: dict[str, Any],
    annotations: Iterable[dict[str, Any]],
    category_indices: dict[int, int],
) -> str:
    """Converte as annotations de uma imagem usando a fórmula existente."""
    lines = []
    width = image["width"]
    height = image["height"]

    for annotation in annotations:
        x, y, bbox_width, bbox_height = annotation["bbox"]
        center_x = (x + bbox_width / 2) / width
        center_y = (y + bbox_height / 2) / height
        normalized_width = bbox_width / width
        normalized_height = bbox_height / height
        category_id = category_indices[annotation["category_id"]]
        lines.append(
            f"{category_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{normalized_width:.6f} "
            f"{normalized_height:.6f}\n"
        )
    return "".join(lines)


def build_materialization_plan(dataset, test_images):
    """Constrói e valida todos os destinos antes de copiar arquivos."""
    categories = {
        category["id"]: index
        for index, category in enumerate(dataset["categories"])
    }

    images_by_id = {
        image["id"]: image
        for image in dataset["images"]
    }

    annotations_by_image = defaultdict(list)

    for annotation in dataset["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    plan = []
    image_destinations = defaultdict(list)
    label_destinations = defaultdict(list)
    for image_id in sorted(test_images):
        image = images_by_id[image_id]
        relative_image = safe_relative_image_path(image["file_name"])
        source_image = RAW_DATA_ROOT / relative_image
        if not source_image.is_file():
            raise FileNotFoundError(
                f"Imagem não encontrada: {source_image}"
            )
        relative_label = relative_image.with_suffix(".txt")
        label_content = render_yolo_label(
            image,
            annotations_by_image[image_id],
            categories,
        )
        plan.append(
            (
                image_id,
                source_image,
                relative_image,
                relative_label,
                label_content,
            )
        )
        image_destinations[relative_image.as_posix().casefold()].append(
            (image_id, image["file_name"])
        )
        label_destinations[relative_label.as_posix().casefold()].append(
            (image_id, image["file_name"])
        )

    collisions = []
    for kind, destinations in (
        ("image", image_destinations),
        ("label", label_destinations),
    ):
        for destination, sources in destinations.items():
            if len(sources) > 1:
                collisions.append((kind, destination, sources))
    if collisions:
        details = "; ".join(
            f"{kind} {destination}: {sources}"
            for kind, destination, sources in collisions
        )
        raise PipelineError(f"Colisões de destino detectadas: {details}")
    return plan


def remove_derived_tree(path: Path) -> None:
    """Remove uma árvore derivada, inclusive quando marcada read-only."""
    if not path.exists():
        return

    def make_writable_and_retry(function, name, _error_info):
        os.chmod(name, stat.S_IWRITE)
        function(name)

    os.chmod(path, stat.S_IWRITE)
    shutil.rmtree(path, onerror=make_writable_and_retry)


def remove_staging() -> None:
    """Remove somente staging derivado e incompleto."""
    for path in (IMAGES_STAGING, LABELS_STAGING):
        remove_derived_tree(path)


def validate_materialization(
    plan,
    images_root: Path,
    labels_root: Path,
) -> tuple[int, int, int]:
    """Valida arquivos, conteúdo e contagens de uma materialização."""
    image_files = [path for path in images_root.rglob("*") if path.is_file()]
    label_files = [path for path in labels_root.rglob("*") if path.is_file()]
    if len(image_files) != len(plan) or len(label_files) != len(plan):
        raise PipelineError(
            "Contagem inválida no staging: "
            f"{len(image_files)} imagens e {len(label_files)} labels "
            f"para {len(plan)} IDs"
        )

    label_lines = 0
    for _, source, relative_image, relative_label, expected_label in plan:
        staged_image = images_root / relative_image
        staged_label = labels_root / relative_label
        if not filecmp.cmp(source, staged_image, shallow=False):
            raise PipelineError(
                f"Imagem no staging diverge da origem: {relative_image}"
            )
        actual_label = staged_label.read_text(encoding="utf-8")
        if actual_label != expected_label:
            raise PipelineError(
                f"Label no staging diverge das annotations: {relative_label}"
            )
        label_lines += len(actual_label.splitlines())
    expected_annotations = sum(
        len(label_content.splitlines())
        for _, _, _, _, label_content in plan
    )
    if label_lines != expected_annotations:
        raise PipelineError(
            f"Staging contém {label_lines} linhas YOLO; "
            f"esperadas {expected_annotations}"
        )
    return len(image_files), len(label_files), label_lines


def promote_staging() -> None:
    """Promove staging validado, restaurando outputs antigos em caso de erro."""
    if IMAGES_BACKUP.exists() or LABELS_BACKUP.exists():
        raise PipelineError(
            "Backup residual detectado; promoção abortada para preservar dados"
        )
    images_backed_up = False
    labels_backed_up = False
    images_promoted = False
    labels_promoted = False
    try:
        if IMAGES_OUTPUT.exists():
            os.replace(IMAGES_OUTPUT, IMAGES_BACKUP)
            images_backed_up = True
        if LABELS_OUTPUT.exists():
            os.replace(LABELS_OUTPUT, LABELS_BACKUP)
            labels_backed_up = True
        os.replace(IMAGES_STAGING, IMAGES_OUTPUT)
        images_promoted = True
        os.replace(LABELS_STAGING, LABELS_OUTPUT)
        labels_promoted = True
    except OSError as exc:
        try:
            if images_promoted and IMAGES_OUTPUT.exists():
                remove_derived_tree(IMAGES_OUTPUT)
            if labels_promoted and LABELS_OUTPUT.exists():
                remove_derived_tree(LABELS_OUTPUT)
            if images_backed_up and IMAGES_BACKUP.exists():
                os.replace(IMAGES_BACKUP, IMAGES_OUTPUT)
            if labels_backed_up and LABELS_BACKUP.exists():
                os.replace(LABELS_BACKUP, LABELS_OUTPUT)
        except OSError as rollback_error:
            raise PipelineError(
                f"Falha na promoção ({exc}) e no rollback ({rollback_error})"
            ) from rollback_error
        raise PipelineError(f"Falha na promoção; rollback concluído: {exc}") from exc

    if IMAGES_BACKUP.exists():
        remove_derived_tree(IMAGES_BACKUP)
    if LABELS_BACKUP.exists():
        remove_derived_tree(LABELS_BACKUP)


def create_yolo_labels(dataset, test_images):
    """Materializa imagens e labels em staging e promove após validação."""
    plan = build_materialization_plan(dataset, test_images)
    remove_staging()
    IMAGES_STAGING.mkdir(parents=True)
    LABELS_STAGING.mkdir(parents=True)
    try:
        for _, source, relative_image, relative_label, label_content in plan:
            image_destination = IMAGES_STAGING / relative_image
            label_destination = LABELS_STAGING / relative_label
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, image_destination)
            label_destination.write_text(label_content, encoding="utf-8")
        counts = validate_materialization(
            plan,
            IMAGES_STAGING,
            LABELS_STAGING,
        )
        promote_staging()
        return counts
    except (OSError, PipelineError):
        remove_staging()
        raise


def write_data_yaml():
    content = """path: .
train: images
val: images
test: images

names:
  0: Can
  1: Other
  2: Bottle
  3: Bottle cap
  4: Cup
  5: Lid
  6: Plastic bag + wrapper
  7: Pop tab
  8: Straw
  9: Cigarette
"""

    with open(
        OUTPUT_ROOT / "data.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(content)


def write_test_image_ids(test_images):
    with open(
        OUTPUT_ROOT / "test_image_ids.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for image_id in sorted(test_images):
            f.write(f"{image_id}\n")


def write_readme(
    dataset,
    test_images,
    test_annotations,
    stratified,
):
    categories = {
        category["id"]: category["name"]
        for category in dataset["categories"]
    }

    counts = {
        name: 0
        for name in categories.values()
    }

    test_image_set = set(test_images)

    for annotation in dataset["annotations"]:
        if annotation["image_id"] in test_image_set:
            category_name = categories[annotation["category_id"]]
            counts[category_name] += 1

    lines = [
        "# Teste Global — TACO-10",
        "",
        "Este diretório contém o conjunto de teste global utilizado "
        "para avaliar os modelos treinados no projeto FedLitter.",
        "",
        "## Divisão",
        "",
        f"- Imagens totais do TACO: {len(dataset['images'])}",
        f"- Imagens de teste: {len(test_images)}",
        f"- Proporção: {TEST_SIZE:.0%}",
        f"- Seed: {RANDOM_STATE}",
        f"- Instâncias no teste: {test_annotations}",
        "",
        "O teste global foi separado antes da criação das partições "
        "dos clientes. Portanto, nenhuma imagem deste conjunto deve "
        "ser utilizada no treinamento ou validação de qualquer cliente.",
        "",
        "## Estratificação",
        "",
    ]

    if stratified:
        lines.extend(
            [
                "A divisão foi realizada com `train_test_split` do "
                "scikit-learn utilizando `random_state=42` e "
                "estratificação.",
                "",
                "Como uma imagem pode conter várias categorias, foi "
                "definida uma categoria principal por imagem: a "
                "categoria com maior número de instâncias na imagem. "
                "Essa categoria foi utilizada apenas para orientar "
                "a estratificação.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Não foi possível realizar uma estratificação segura, "
                "pois pelo menos uma categoria não possuía imagens "
                "suficientes para formar os grupos de treino e teste.",
                "",
                "Nesse caso, o split foi realizado sem estratificação, "
                "mantendo a seed fixa em 42 para garantir "
                "reprodutibilidade.",
                "",
            ]
        )

    lines.extend(
        [
            "## Distribuição das instâncias",
            "",
            "| Categoria | Instâncias |",
            "|---|---:|",
        ]
    )

    for name, count in counts.items():
        lines.append(f"| {name} | {count} |")

    lines.extend(
        [
            "",
            "## Controle de sobreposição",
            "",
            "Os IDs das imagens selecionadas para o teste global são "
            "armazenados em `test_image_ids.txt`.",
            "",
            "O particionamento dos clientes deverá consultar essa lista "
            "e verificar, por meio de `assert`, que nenhuma imagem do "
            "teste global foi atribuída a um cliente.",
            "",
            "O objetivo é garantir que FedAvg, FedProx e FedTrimmed "
            "sejam avaliados exatamente sobre os mesmos exemplos, "
            "sem vazamento de dados entre treinamento e teste.",
        ]
    )

    with open(
        OUTPUT_ROOT / "README.md",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(lines) + "\n")


def main():
    dataset = load_dataset()

    images = dataset["images"]

    if len(images) != 1500:
        print(
            f"Aviso: eram esperadas 1500 imagens, "
            f"mas foram encontradas {len(images)}."
        )

    all_ids = {image["id"] for image in images}
    test_ids, stratified, created_split = select_test_ids(dataset)
    development_ids = all_ids - test_ids

    if not test_ids.isdisjoint(development_ids):
        raise PipelineError("Existe sobreposição entre teste e desenvolvimento")
    if test_ids | development_ids != all_ids:
        raise PipelineError(
            "Teste e desenvolvimento não recompõem todos os IDs do dataset"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    copied_images, copied_labels, copied_annotations = create_yolo_labels(
        dataset,
        test_ids,
    )

    if created_split:
        write_test_image_ids(test_ids)
        write_data_yaml()
        write_readme(
            dataset,
            test_ids,
            copied_annotations,
            stratified,
        )

    print("Teste global criado com sucesso!")
    print(f"Imagens totais: {len(images)}")
    print(f"Imagens de teste: {len(test_ids)}")
    print(f"Imagens de desenvolvimento: {len(development_ids)}")
    print(f"Sobreposição teste/desenvolvimento: {len(test_ids & development_ids)}")
    print(f"Arquivos de imagem: {copied_images}")
    print(f"Arquivos de label: {copied_labels}")
    print(f"Instâncias de teste: {copied_annotations}")
    print(f"IDs autoritativos reutilizados: {not created_split}")
    if created_split:
        print(f"Estratificação utilizada: {stratified}")
    print(f"Seed: {RANDOM_STATE}")
    print(f"Diretório: {OUTPUT_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except (PipelineError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    

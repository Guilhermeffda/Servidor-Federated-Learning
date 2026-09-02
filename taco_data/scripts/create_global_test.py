from pathlib import Path
import json
import shutil

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS_FILE = (
    ROOT / "data/processed/taco10/annotations_regrouped.json"
)

RAW_DATA_ROOT = (
    ROOT / "data/raw/taco/extracted/TACO/data"
)

OUTPUT_ROOT = ROOT / "data/test_global"

TEST_SIZE = 0.20
RANDOM_STATE = 42


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


def create_yolo_labels(dataset, test_images):
    categories = {
        category["id"]: index
        for index, category in enumerate(dataset["categories"])
    }

    images_by_id = {
        image["id"]: image
        for image in dataset["images"]
    }

    annotations_by_image = {}

    for annotation in dataset["annotations"]:
        image_id = annotation["image_id"]

        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []

        annotations_by_image[image_id].append(annotation)

    images_output = OUTPUT_ROOT / "images"
    labels_output = OUTPUT_ROOT / "labels"

    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_annotations = 0

    for image_id in test_images:
        image = images_by_id[image_id]

        source_image = RAW_DATA_ROOT / image["file_name"]

        if not source_image.is_file():
            raise FileNotFoundError(
                f"Imagem não encontrada: {source_image}"
            )

        destination_image = images_output / source_image.name
        shutil.copy2(source_image, destination_image)

        label_file = labels_output / f"{source_image.stem}.txt"

        width = image["width"]
        height = image["height"]

        with open(label_file, "w", encoding="utf-8") as f:
            for annotation in annotations_by_image.get(image_id, []):
                x, y, bbox_width, bbox_height = annotation["bbox"]

                center_x = (x + bbox_width / 2) / width
                center_y = (y + bbox_height / 2) / height
                normalized_width = bbox_width / width
                normalized_height = bbox_height / height

                category_id = categories[annotation["category_id"]]

                f.write(
                    f"{category_id} "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{normalized_width:.6f} "
                    f"{normalized_height:.6f}\n"
                )

                copied_annotations += 1

        copied_images += 1

    return copied_images, copied_annotations


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

    image_ids = [image["id"] for image in images]

    image_primary_category = build_image_categories(dataset)

    labels = [
        image_primary_category[image_id]
        for image_id in image_ids
    ]

    stratified = can_stratify(labels)

    if stratified:
        train_ids, test_ids = train_test_split(
            image_ids,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
    else:
        train_ids, test_ids = train_test_split(
            image_ids,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    train_ids = set(train_ids)
    test_ids = set(test_ids)

    # Garantia de que uma imagem não pertence aos dois conjuntos.
    assert train_ids.isdisjoint(test_ids), (
        "Erro: existe sobreposição entre treino e teste global."
    )

    expected_test_size = round(len(images) * TEST_SIZE)

    assert len(test_ids) == expected_test_size, (
        f"Tamanho inesperado do teste: {len(test_ids)} "
        f"(esperado: {expected_test_size})"
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    copied_images, copied_annotations = create_yolo_labels(
        dataset,
        test_ids,
    )

    write_data_yaml()
    write_test_image_ids(test_ids)
    write_readme(
        dataset,
        test_ids,
        copied_annotations,
        stratified,
    )

    print("Teste global criado com sucesso!")
    print(f"Imagens totais: {len(images)}")
    print(f"Imagens de teste: {len(test_ids)}")
    print(f"Instâncias de teste: {copied_annotations}")
    print(f"Estratificação utilizada: {stratified}")
    print(f"Seed: {RANDOM_STATE}")
    print(f"Diretório: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
    
from pathlib import Path
import csv
import json


ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS_FILE = ROOT / "data/raw/taco/extracted/TACO/data/annotations.json"
MAP_FILE = ROOT / "data/raw/taco/extracted/TACO/detector/taco_config/map_10.csv"
OUTPUT_FILE = ROOT / "data/processed/taco10/annotations_regrouped.json"


# 1. Carrega o dataset original
with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)


# 2. Carrega o mapeamento oficial TACO-10
mapping = {}

with open(MAP_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f)

    for row in reader:
        if len(row) >= 2:
            original_category = row[0].strip()
            target_category = row[1].strip()
            mapping[original_category] = target_category


# 3. Mapeia ID original -> nome original
original_categories = {
    category["id"]: category["name"]
    for category in dataset["categories"]
}


# 4. Obtém as 10 categorias TACO-10 na ordem do map_10.csv
target_categories = list(dict.fromkeys(mapping.values()))

target_ids = {
    name: index
    for index, name in enumerate(target_categories)
}


# 5. Atualiza as categorias das anotações
for annotation in dataset["annotations"]:
    original_name = original_categories[annotation["category_id"]]
    target_name = mapping[original_name]

    annotation["category_id"] = target_ids[target_name]


# 6. Substitui a lista de categorias pelo TACO-10
dataset["categories"] = [
    {
        "id": target_ids[name],
        "name": name,
        "supercategory": name
    }
    for name in target_categories
]


# 7. Cria o diretório de saída
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# 8. Salva o novo COCO JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)


# 9. Mostra a quantidade de anotações por classe
counts = {name: 0 for name in target_categories}

for annotation in dataset["annotations"]:
    category_name = next(
        name for name, category_id in target_ids.items()
        if category_id == annotation["category_id"]
    )
    counts[category_name] += 1


print("TACO-10 gerado com sucesso!")
print(f"Arquivo: {OUTPUT_FILE}")
print("\nAnotações por categoria:")

for name, count in counts.items():
    print(f"  {name}: {count}")
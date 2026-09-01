# Dados — TACO

Esta pasta contém os dados utilizados no projeto FedLitter.

## TACO

O dataset utilizado é o **TACO (Trash Annotations in Context)**, originalmente composto por 60 categorias.

Para os experimentos, as categorias são agrupadas segundo o mapeamento oficial **TACO-10**, utilizado no trabalho original do TACO:

* Bottle
* Bottle cap
* Can
* Cigarette
* Cup
* Lid
* Other
* Plastic bag + wrapper
* Pop tab
* Straw

O paper descreve essa taxonomia como **9 supercategorias + uma classe residual `Other Litter`**. No mapeamento oficial (`map_10.csv`), essa classe aparece como `Other`.

## Instalando o dataset
cd taco_data
py scripts\download_dataset.py
py scripts\inspect_dataset.py


## Processamento

Os dados originais ficam em:

```text
data/raw/taco/
```

O agrupamento para TACO-10 é realizado por:

```bash
python scripts/regroup_categories.py
```

O resultado é salvo em:

```text
data/processed/taco10/annotations_regrouped.json
```

O arquivo de saída mantém o formato **COCO JSON**, alterando apenas a categorização das anotações para o esquema TACO-10.

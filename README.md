# FedLitter

Projeto de Trabalho de Conclusão de Curso sobre **Aprendizado Federado para detecção de resíduos urbanos**.

## Objetivo

Investigar o uso de Aprendizado Federado para treinar modelos de detecção de resíduos sem centralizar as imagens dos clientes.

O projeto compara diferentes estratégias de agregação e avalia o impacto delas no desempenho do modelo.

## Dataset

É utilizado o dataset **TACO (Trash Annotations in Context)**.

O TACO possui originalmente 60 categorias. Para os experimentos, elas são agrupadas no esquema **TACO-10**, composto por 10 categorias:

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

O processamento dos dados está documentado em `taco_data/data/README.md`.

## Estrutura

```text
Servidor-Federated-Learning/
├── taco_data/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   └── scripts/
│
├── fl/
├── server/
├── client/
└── README.md
```

## Dados

Para baixar e preparar o TACO:

```bash
python taco_data/scripts/download_dataset.py
python taco_data/scripts/inspect_dataset.py
python taco_data/scripts/regroup_categories.py
```

O último comando gera:

```text
taco_data/data/processed/taco10/annotations_regrouped.json
```

## Aprendizado Federado

O projeto utiliza uma arquitetura cliente-servidor para simular diferentes participantes do treinamento federado.

Cada cliente realiza treinamento local e envia os parâmetros do modelo ao servidor. O servidor agrega os modelos utilizando as estratégias avaliadas no projeto.

## Tecnologias

* Python
* PyTorch
* Flower
* COCO / pycocotools
* TACO
* Docker

## Projeto

Este repositório contém a implementação e os experimentos desenvolvidos para o TCC.

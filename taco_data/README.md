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

```bash
cd taco_data
py scripts\download_dataset.py
py scripts\inspect_dataset.py
```

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

Para converter as anotações para YOLO e gerar, de forma reproduzível, o dataset
centralizado e as partições IID/non-IID dos cinco clientes:

```bash
python scripts/prepare_experiments.py
```

O script usa seed 42 e registra a política, as contagens e o hash dos IDs do teste
em `../data/partitions/metadata.json` (a pasta gerada é deliberadamente ignorada
pelo Git por conter caminhos e dados locais).

## Divisão dos dados

O conjunto de dados é dividido em duas etapas.

### 1. Teste global

Antes de qualquer particionamento entre clientes, 20% das imagens são
isoladas como conjunto de teste global.

O TACO possui 1.500 imagens, portanto:

- 300 imagens são reservadas para o teste global;
- 1.200 imagens permanecem disponíveis para desenvolvimento e
  particionamento entre clientes.

O teste global não participa do treinamento nem da validação dos clientes.

A divisão utiliza `train_test_split` do scikit-learn com:

- `test_size=0.20`;
- `random_state=42`.

A seed é fixa para garantir que todas as estratégias de agregação sejam
avaliadas exatamente sobre os mesmos exemplos.

### 2. Treino e validação dos clientes

As 1.200 imagens restantes formam o conjunto de desenvolvimento.

Essas imagens serão posteriormente distribuídas entre os clientes. Após
a criação das partições de cada cliente, os dados de cada cliente serão
divididos em:

- **Treino:** utilizado para atualizar localmente o modelo;
- **Validação:** utilizada para acompanhar o desempenho durante o
  treinamento e auxiliar na análise de convergência.

A validação é realizada dentro das partições dos clientes e não utiliza
nenhuma imagem pertencente ao teste global.

A divisão final seguirá o formato:

```text
TACO — 1.500 imagens
│
├── Teste global — 300 imagens (20%)
│   └── usado somente na avaliação final
│
└── Desenvolvimento — 1.200 imagens (80%)
    │
    └── Particionamento entre clientes
        │
        ├── Cliente 1
        │   ├── Treino
        │   └── Validação
        │
        ├── Cliente 2
        │   ├── Treino
        │   └── Validação
        │
        └── ...
```

Essa separação garante que o teste global permaneça completamente
independente do treinamento federado.

#### Por que separar dessa forma?

O teste global precisa ser o mesmo para todas as execuções do projeto.
Assim, FedAvg, FedProx e FedTrimmed são comparados sobre exatamente os
mesmos exemplos.

A validação, por outro lado, pertence ao processo de desenvolvimento
dos clientes e será utilizada para acompanhar o treinamento antes da
avaliação final.

Dessa forma:

Treino: aprendizado dos modelos locais;
Validação: acompanhamento e análise durante o desenvolvimento;
Teste global: avaliação final e comparação entre as estratégias
de agregação.

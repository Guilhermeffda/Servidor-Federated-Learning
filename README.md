# FedLitter

Aprendizado Federado para detecção de resíduos urbanos — comparação de estratégias
de agregação (FedAvg, FedProx e FedTrimmed) sob cenários IID e non-IID.

Trabalho de Conclusão de Curso — Bacharelado em Ciência da Computação, PUCPR.

---

## Documentação

Este repositório tem **três documentos de referência**. Não duplique informação
entre eles.

| Documento | Para quê |
|---|---|
| `README.md` (este) | Visão geral, dataset, arquitetura, status macro |
| [`docs/INSTRUCOES.md`](docs/INSTRUCOES.md) | **Como rodar tudo**, do zero ao experimento final |
| [`docs/STATUS.md`](docs/STATUS.md) | Status detalhado por área (P1–P5) e por card |

Documentos técnicos de apoio:

- [`fl/CONTRACT.md`](fl/CONTRACT.md) — contrato servidor ↔ cliente (pesos, config, métricas)
- [`taco_data/data/README.md`](taco_data/data/README.md) — auditoria estrutural do TACO
- [`notes/hardware_benchmark.md`](notes/hardware_benchmark.md) — custo por época e viabilidade da grade

---

## Pergunta de pesquisa

Em que medida a escolha da estratégia de agregação federada (FedAvg, FedProx,
FedTrimmed) afeta o desempenho de um detector de resíduos treinado sobre clientes
heterogêneos, e qual valor do termo proximal μ do FedProx maximiza esse desempenho.

**Hipóteses.** H1: sob non-IID, FedProx supera FedAvg. H2: FedTrimmed é mais estável
que FedAvg sob atualizações extremas, mesmo sem clientes maliciosos. H3: a diferença
entre estratégias é menor no cenário IID.

---

## Dataset

**TACO (Trash Annotations in Context)** — 1.500 imagens, 4.784 anotações de
instância em formato COCO.

As 60 categorias originais são agrupadas no esquema **TACO-10**. O artigo original
descreve a taxonomia como 9 supercategorias mais uma classe residual chamada
*Other Litter*; o mapeamento oficial dos autores (`map_10.csv`) nomeia essa classe
residual como **`Other`**. Este projeto segue o mapeamento oficial:

| ID | Classe | Instâncias |
|---:|---|---:|
| 0 | Can | 273 |
| 1 | Other | 1.727 |
| 2 | Bottle | 439 |
| 3 | Bottle cap | 289 |
| 4 | Cup | 192 |
| 5 | Lid | 87 |
| 6 | Plastic bag + wrapper | 850 |
| 7 | Pop tab | 99 |
| 8 | Straw | 161 |
| 9 | Cigarette | 667 |

### Divisão dos dados

```
TACO — 1.500 imagens
├── Teste global — 300 (20%, estratificado, seed 42) → avaliação final de TODAS as estratégias
└── Desenvolvimento — 1.200
    ├── Centralizado: 960 treino / 240 validação        (baseline de referência)
    └── Federado: 5 clientes × {IID, non-IID}
```

O teste global é separado **antes** de qualquer particionamento e é idêntico para
FedAvg, FedProx, FedTrimmed e para a baseline centralizada — é o que garante
comparação justa.

**Cenário IID:** `StratifiedKFold` sobre as imagens de desenvolvimento, preservando
a proporção de classes em cada cliente.

**Cenário non-IID:** alocação de Dirichlet por classe primária, **α = 0,5**.
Candidatos 0,1 / 0,5 / 1,0 foram analisados; 0,5 foi escolhido por equilibrar
heterogeneidade mensurável e viabilidade operacional (CV de imagens 0,544; entropia
normalizada média 0,766; JSD média 0,045; cobertura mínima 7/10 classes primárias;
nenhum cliente vazio). Justificativa completa em
`taco_data/data/partitions/non_iid/manifest_non_iid.json`.

---

## Arquitetura

```
fl/
├── CONTRACT.md   # contrato servidor ↔ cliente
├── model.py      # YOLOv8n ↔ Flower: get/set_weights, train_local, evaluate, termo proximal
├── client.py     # FLClient (NumPyClient): treino local por round
├── server.py     # FedAvg, on_fit_config_fn, evaluate_metrics_aggregation_fn
├── dataset.py    # resolve data/partitions/{iid,non_iid}/client_X/data.yaml
└── partition.py  # particionamento genérico (usado só pelo smoke test)

run_config.py     # runner: orquestra rounds, agrega, salva métricas e curvas
```

Modelo: **YOLOv8n** (Ultralytics), transfer learning a partir de `yolov8n.pt`.
Orquestração: **Flower**. Treino: **PyTorch**.

O termo proximal do FedProx é injetado no `loss` do trainer do Ultralytics antes do
backward (`fl/model.py::_install_fedprox_loss`), ativado quando `mu > 0`.

---

## Status macro

| Área | Status |
|---|---|
| Aquisição, auditoria e reagrupamento do TACO | ✅ Completo |
| Teste global (300 imagens, estratificado) | ✅ Completo |
| Partições IID e non-IID (Dirichlet α=0,5) | ✅ Completo |
| Relatório de classes escassas + augmentation | ✅ Completo |
| Cliente federado (troca de pesos + treino local) | ✅ Completo |
| Agregação FedAvg + config por round + métricas | ✅ Completo |
| Termo proximal do FedProx (código) | ✅ Completo |
| Pipeline validado ponta a ponta (smoke test) | ✅ Completo |
| **Baseline centralizada (50 épocas)** | ⬜ **Pendente de execução** |
| **6 baselines federadas TACO-10 (50 rounds)** | ⬜ **Pendente de execução** |
| **Execução com μ > 0 (FedProx real)** | ⬜ **Pendente de execução** |
| Infraestrutura de execução paralela (P5) | ⬜ Não iniciado |
| FedTrimmed | ⛔ Sprint 3 |

> **O código está pronto; os experimentos científicos não foram rodados.**
> Todos os resultados hoje em `results/baseline/` estão marcados
> `scientific_valid: false` — são smoke tests de pipeline, não resultados do artigo.

Detalhamento por card em [`docs/STATUS.md`](docs/STATUS.md).

---

## Riscos abertos

**Tamanho da grade experimental.** 42 execuções × 50 rounds × 5 épocas × 5 clientes
= 52.500 épocas de treino. Medição real: 160 s/época em CPU, 50 s/época em GPU (MPS)
sobre uma partição de 192 imagens. Mesmo em GPU, isso dá ~30 dias sequenciais numa
máquina só. **Precisa de decisão do grupo** — paralelizar entre integrantes, reduzir
a grade de μ de 5 para 3 valores, ou reduzir rounds de 50 para 30. Ver
`notes/hardware_benchmark.md`.

**Classes escassas pós-Dirichlet.** Combinações (cliente, classe) com menos de 15
instâncias estão catalogadas em `taco_data/data/partitions/augmentation_report.json`.
Mitigação via augmentations nativas do Ultralytics, configuradas em
`configs/baseline_taco.yaml`. Casos com 0 instâncias não são resolvíveis por
augmentation e ficam como limitação declarada do cenário non-IID.

---

## Escopo — o que este projeto NÃO faz

- Não simula clientes maliciosos nem ataques Byzantine à agregação
- Não compara outros agregadores robustos além do FedTrimmed (Krum, mediana geométrica)
- Não implanta em hardware físico real — tudo é simulação em lote, máquina única
- Não oferece garantia formal de privacidade (privacidade diferencial, agregação
  segura ficam fora do escopo)
  

### RESULTADOS DE BENCHMARK JA OBTIDOS:

| Pessoa | s/epoca | tempo estimado |
|---|---:|---:|
| GPU CAMILA | 20,28s | 12 dias |
| GPU GUILHERME | 18,60s | 11 dias |

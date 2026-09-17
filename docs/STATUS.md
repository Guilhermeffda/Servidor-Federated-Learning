# Status por área (P1–P5)

Atualizado em 16/09/2026 — fim da Sprint 2.

**Legenda:** ✅ concluído · 🟡 código pronto, execução pendente · ⬜ não iniciado · ⛔ Sprint 3+

---

## Resumo

| Área | Responsabilidade | Sprint 2 |
|---|---|---|
| **P1** | Revisão bibliográfica, redação, formatação | ✅ Concluído |
| **P2** | Dataset, partições, augmentation | ✅ Concluído |
| **P3** | Servidor, agregação, FedProx (lado servidor) | 🟡 Código completo, execução pendente |
| **P4** | Cliente, modelo, experimentos | 🟡 Código completo, execução pendente |
| **P5** | Infraestrutura paralela, estatística, figuras | ⬜ Não iniciado |

> **O gargalo da Sprint 2 não é código — é tempo de GPU.** Todo o pipeline está
> implementado e validado ponta a ponta. Nenhuma execução científica foi concluída.
> Ver a seção [Bloqueio crítico](#bloqueio-crítico) no fim.

---

## P1 — Revisão bibliográfica e redação

| Card | Status | Evidência |
|---|---|---|
| Fechar leitura de Li et al. 2020 (convergência non-IID) e Li et al. 2023 (agregação robusta) | ✅ | Ambos citados e fundamentando H1/H2 no planejamento |

Referências verificadas e com metadados completos: McMahan 2017, Lu & Chen 2022
(*Waste Management* 142:29–43), Khan et al. 2024 (*Sci. Rep.* 14:11816),
Wang et al. 2026 (*IJECES* 17(5):355–366), Proença & Simões 2020, Hsu et al. 2019,
Li et al. 2020 (FedProx/MLSys), Yin et al. 2018, Li, Ngai & Voigt 2023
(*IEEE T. Big Data* 10(6):975–988), Li et al. 2020 (ICLR), Kairouz et al. 2021,
Lin et al. 2014.

**Próximo (Sprint 3):** rascunho de Trabalhos Relacionados + Introdução.

---

## P2 — Dataset e particionamento

| Card | Status | Evidência |
|---|---|---|
| Reagrupar TACO em TACO-10 | ✅ | `regroup_categories.py` usa o `map_10.csv` oficial; saída auditada |
| Separar 20% como teste global | ✅ | 300 imagens, 1.068 instâncias, estratificado, seed 42 |
| Partição IID (5 clientes) | ✅ | `StratifiedKFold`; `manifest_iid.json` com JSD par a par |
| Partição non-IID via Dirichlet | ✅ | α=0,5 escolhido entre {0,1 / 0,5 / 1,0}; `alpha_comparison.svg` |
| Data augmentation para classes escassas | ✅ | `augmentation_report.json` + bloco `augmentation` no config |

### DoD do card de Dirichlet — conferido

- [x] 5 pastas non-IID geradas e carregáveis pelo Ultralytics
- [x] α justificado por escrito — CV 0,544; entropia normalizada 0,766; JSD 0,045;
      cobertura mínima 7/10 classes primárias
- [x] Gráfico de distribuição em formato vetorial (`.svg`)
- [x] Nenhum cliente degenerado (o gerador rejeita e re-sorteia se algum cliente
      ficar abaixo de `len(imagens)/(num_clients*4)`)
- [x] Nenhuma sobreposição com o teste global (assert automático + teste em
      `tests/test_prepare_experiments.py`)

### Limitação declarada

Combinações (cliente, classe) com **0 instâncias** no cenário non-IID não são
resolvíveis por augmentation e ficam documentadas como limitação conhecida — não
como falha do particionamento. É o comportamento esperado de α=0,5.

---

## P3 — Servidor e agregação

| Card | Status | Evidência |
|---|---|---|
| Implementar agregação FedAvg | ✅ | `fl/server.py`, FedAvg nativo do Flower configurado |
| `on_fit_config_fn` por round | ✅ | `make_fit_config_fn` envia `local_epochs`, `batch_size`, `image_size`, `device`, `mu`, `nbs` + `aug_*` |
| `evaluate_metrics_aggregation_fn` | ✅ | `weighted_evaluate_metrics`, ponderada por `num_examples` |
| Teste unitário de agregação | ✅ | `test_fedavg_is_weighted_by_num_examples` |
| `min_fit_clients` ajustado para 5 | ✅ | Parametrizado por `num_clients` no config |
| Termo proximal do FedProx (μ no config) | ✅ | Campo `mu` no contrato e no `on_fit_config_fn` |
| `fl/CONTRACT.md` atualizado | ✅ | Documenta `mu`, `nbs` e as chaves `aug_*` |
| **Baseline FedAvg end-to-end (50 rounds, IID + non-IID)** | 🟡 | Pipeline validado em smoke test; **execução científica pendente** |
| **FedProx com μ fixo rodando até o fim** | 🟡 | Código pronto; **nenhuma execução com μ > 0 registrada** |

> **Nota sobre `start_server` vs simulação.** O card previa migrar de
> `fl.server.start_server` para `fl.simulation.start_simulation`. A implementação
> seguiu outro caminho: `run_config.py` orquestra os rounds diretamente, chamando
> `fit`/`evaluate` dos clientes e agregando com `aggregate_fedavg` (a mesma função
> do FedAvg do Flower). O resultado é equivalente, mais simples de instrumentar e
> mais fácil de retomar após queda de sessão. `fl/server.py::get_strategy` continua
> disponível para um servidor Flower distribuído, se algum dia for necessário.

---

## P4 — Cliente, modelo e experimentos

| Card | Status | Evidência |
|---|---|---|
| Setup YOLOv8n local + sanity check | ✅ | `scripts/sanity_check.py` |
| Esqueleto do FLClient | ✅ | Superado pela implementação real |
| Portar `sprint2_preview` → `fl/`, remover dropout | ✅ | `fl/model.py` + `fl/client.py` ativos |
| Round-trip exato dos pesos | ✅ | 355 tensores, `test_yolo_weight_roundtrip_is_exact` |
| Suporte a `mu` (termo proximal) | ✅ | Injetado no `loss` antes do backward, não como pós-ajuste |
| Treino isolado numa partição TACO real | ✅ | `results/client_taco_smoke/result.json` |
| Remedir benchmark com TACO real | ✅ | 160 s/época CPU; 50 s/época GPU (MPS) |
| **Baseline centralizada (50 épocas)** | 🟡 | Runner pronto, dry-run validado (960/240/300); **treino final pendente** |
| **6 baselines FedAvg TACO-10** | 🟡 | 6 smoke tests concluídos com `scientific_valid: false`; **execuções reais pendentes** |

### Sobre o split da baseline centralizada

O card mencionava 70/15/15. Foi mantido o **64/16/20** já documentado no
repositório, porque o teste global de 20% (300 imagens) precisa ser **exatamente o
mesmo** usado para avaliar os modelos federados. Avaliar centralizado e federado em
conjuntos diferentes invalidaria a comparação. Decisão deliberada, não desvio.

### Bug corrigido em 15/09/2026 — leia antes de usar qualquer resultado antigo

`fl/model.py::train_local` não sincronizava os pesos treinados de volta para o
modelo do cliente: o Ultralytics treina uma cópia interna e `save=False` não
recarrega o checkpoint. Cada cliente devolvia os pesos globais **inalterados** e
todas as execuções federadas eram no-ops silenciosos — as métricas ficavam
bit-idênticas em todos os rounds.

- Corrigido por `_sync_trained_weights`
- Coberto por `test_train_local_changes_trainable_parameters`
- Os 6 smoke tests foram reexecutados com o código corrigido
- **Nenhum resultado científico foi afetado, porque nenhum havia sido gerado ainda**

---

## P5 — Infraestrutura paralela, estatística e figuras

| Card | Status |
|---|---|
| Preparar infraestrutura de execução paralela | ⬜ Não iniciado |
| Benchmark próprio, comparável ao de P4 | ⬜ Não iniciado |
| Validar `run_config.py` com saída em formato idêntico | ⬜ Não iniciado |

Nenhum artefato de P5 existe no repositório. **É o único card de Sprint 2 sem
nenhum progresso** — e é justamente o que destravaria a divisão de carga
computacional descrita no bloqueio abaixo.

Para começar: seções [1](INSTRUCOES.md#1-instalação),
[2](INSTRUCOES.md#2-preparação-dos-dados), [4](INSTRUCOES.md#4-smoke-tests) e
[5](INSTRUCOES.md#5-benchmark-de-hardware) do guia de instruções. O DoD é ter um
`measurement.json` próprio e uma execução de `configs/taco_smoke.yaml` cujos
artefatos batam em formato com os de P4.

---

## Bloqueio crítico

**A grade experimental não cabe no prazo numa máquina só.**

```
42 execuções × 50 rounds × 5 épocas × 5 clientes = 52.500 épocas
```

| Hardware | s/época | Grade completa |
|---|---:|---:|
| CPU (i7-1165G7) | 160,2 | 97 dias |
| GPU (M1 Pro / MPS) | 50,1 | 30 dias |
| GPU NVIDIA (estimado otimista) | ~25 | ~15 dias |

### RESULTADOS DE BENCHMARK JA OBTIDOS:

| Pessoa | s/epoca | tempo estimado |
|---|---:|---:|
| GPU CAMILA | 20,28s | 12 dias |

Opções, em ordem de preferência:

1. **Paralelizar entre integrantes** — já planejado, mas insuficiente sozinho
   (3 máquinas ≈ 10 dias). Depende de P5 sair do zero.
2. **Reduzir a grade de μ** de 5 para 3 valores — já previsto como primeiro corte
   no checkpoint de risco de ~20–22/09.
3. **Reduzir rounds** de 50 para 30, se a curva de convergência estabilizar antes.

Essa decisão precisa ser tomada **antes** de comprometer horas de GPU. Cada
integrante deve rodar o benchmark na própria máquina e levar o número ao grupo.

---

## Sprint 3 — o que vem

| Card | Área |
|---|---|
| Implementar FedTrimmed (trimmed mean, β=0,4) | P3 |
| Varredura de μ ∈ {0, 0.001, 0.01, 0.1, 1} — 30 execuções | P3/P4/P5 |
| Identificar μ* por cenário | P4/P5 |
| Rascunho de Trabalhos Relacionados + Introdução | P1 |
| **Checkpoint de risco (~20–22/09)** — validar ritmo, cortar grade se atrasado | Todos |

> O planejamento do artigo usa **β = 0,4** para o FedTrimmed (k=1 de cada lado com
> 5 clientes). Alguns cards antigos citam β=0,2 — está desatualizado; vale 0,4.
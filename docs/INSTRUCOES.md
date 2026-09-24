# Instruções de execução

Tudo que precisa ser rodado no projeto, na ordem correta. Cada etapa diz **quando**
rodar, **quanto tempo leva** e **como saber que deu certo**.

> Todos os comandos assumem que você está na **raiz do repositório** e com o
> ambiente virtual ativado.

---

## Índice

1. [Instalação](#1-instalação)
2. [Preparação dos dados](#2-preparação-dos-dados) — roda uma vez
3. [Validação da instalação](#3-validação-da-instalação) — roda uma vez
4. [Smoke tests](#4-smoke-tests) — sempre que mexer no pipeline
5. [Benchmark de hardware](#5-benchmark-de-hardware) — uma vez por máquina
6. [Experimentos científicos](#6-experimentos-científicos) — os resultados do artigo
7. [Validação de artefatos](#7-validação-de-artefatos)
8. [Solução de problemas](#8-solução-de-problemas)

---

## 1. Instalação

Python 3.12 ou superior.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\activate

# GPU NVIDIA (CUDA 12.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# ou, se não tiver GPU:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate

# Linux + NVIDIA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# macOS (Apple Silicon — usa MPS, já vem no build padrão):
# pip install torch torchvision

pip install -r requirements.txt
```

### Conferir se a GPU foi reconhecida

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

Se `CUDA: True`, o PyTorch está vendo a GPU NVIDIA corretamente. Guarde o
resultado — ele define o valor de `device` nos passos seguintes.

Em `configs/taco_smoke.yaml` (linha `device:`) e em cada `configs/*.yaml` que você
for rodar (Seção 6), ajuste `device` conforme seu hardware:

| Hardware | `device` |
|---|---|
| NVIDIA | `"0"` |
| Apple Silicon | `"mps"` |
| Sem GPU | `"cpu"` |

> Este projeto roda exclusivamente em máquinas próprias com GPU (Seção 5 confirma
> por quê). Não há caminho de execução via Google Colab — foi removido deste guia
> porque nenhuma máquina do grupo precisa dele.

---

## 2. Preparação dos dados

**Roda uma vez por máquina — cada pessoa do grupo precisa rodar isso na sua
própria máquina antes de participar dos experimentos.** Baixa ~2,7 GB e
materializa todos os splits.

Execute **nesta ordem** — cada script depende da saída do anterior:

```bash
# 1. Baixa o TACO do Zenodo, valida MD5/SHA-256 e extrai  (~2,7 GB, 10-30 min)
python taco_data/scripts/download_dataset.py

# 2. Auditoria estrutural (não modifica nada) → atualiza taco_data/data/README.md
python taco_data/scripts/inspect_dataset.py

# 3. Reagrupa as 60 categorias em TACO-10 usando o map_10.csv oficial
python taco_data/scripts/regroup_categories.py

# 4. Materializa splits YOLO: centralizado, teste global, partições IID e non-IID
python taco_data/scripts/prepare_experiments.py
```

### O que o passo 4 gera

```
data/
├── centralized/          # 960 treino / 240 val / 300 teste + data.yaml
└── partitions/
    ├── iid/client_0..4/      # cada um com data.yaml
    ├── non_iid/client_0..4/  # Dirichlet α=0,5
    └── metadata.json

taco_data/data/test_global/   # 300 imagens + data.yaml + test_image_ids.txt
```

As imagens são **symlinks** para `taco_data/data/raw/`, não cópias — não ocupa
espaço duplicado, mas não mova a pasta `raw` depois.

**Importante para quem for rodar a grade em paralelo (Seção 6.4):** a
`partition_seed: 42` é fixa no config, então as partições IID e non-IID geradas
por `prepare_experiments.py` são **idênticas em qualquer máquina** — Camila,
Guilherme e Anabelly treinam sobre os mesmos 5 clientes, só em computadores
diferentes.

### Como saber que deu certo

```bash
python -c "
import yaml,pathlib
for s in ['iid','non_iid']:
    for i in range(5):
        p=pathlib.Path(f'data/partitions/{s}/client_{i}/data.yaml')
        assert p.is_file(), f'FALTANDO: {p}'
    print(f'{s}: 5 clientes OK')
print('centralizado:', pathlib.Path('data/centralized/data.yaml').is_file())
print('teste global:', pathlib.Path('taco_data/data/test_global/data.yaml').is_file())
"
```

### Análise de α (opcional — gera a figura do artigo)

```bash
python taco_data/scripts/partition_non_iid.py --analyze
```

Saídas: `taco_data/data/partitions/non_iid/alpha_analysis.json` e
`alpha_comparison.svg`. Só precisa rodar uma vez, por qualquer pessoa — o
resultado não depende de máquina.

### Relatório de classes escassas (opcional)

```bash
python taco_data/scripts/augmentation_report.py
```

Lista combinações (cliente, classe) com poucas instâncias →
`taco_data/data/partitions/augmentation_report.json`.

---

## 3. Validação da instalação

```bash
python -m pytest
```

Nove testes devem passar. Cobrem: média ponderada do FedAvg, contrato do
`on_fit_config_fn`, agregação de métricas, disjunção e reprodutibilidade das
partições, termo proximal do FedProx, round-trip exato dos pesos, e — importante —
**regressão de que o treino local realmente altera os pesos**.

> Esse último teste existe porque houve um bug em que o `model.train()` do
> Ultralytics treinava uma cópia interna e os pesos treinados não voltavam para o
> modelo do cliente, tornando todas as execuções federadas no-ops silenciosos. Não
> remova esse teste.

---

## 4. Smoke tests

Rode sempre que mexer no pipeline, **antes** de comprometer horas de GPU — e
**sempre**, antes de rodar sua fatia da grade pela primeira vez (Seção 6.4).

### 4.1 Smoke test sobre as partições TACO reais (minutos)

**Este é o teste que importa** — valida partições → treino local → agregação →
avaliação no teste global:

```bash
python run_config.py --config configs/taco_smoke.yaml --scenario iid --repetition 1
```

5 clientes, 2 rounds, 416 px. Saída em `results/taco_smoke/`.

Valide o formato:

```bash
python scripts/validate_run_format.py --candidate results/taco_smoke/iid_rep1
```

Espera `✅ FORMATO VALIDADO`. Se não der, não siga para a Seção 6 — algo no seu
ambiente está diferente do esperado.

### 4.2 Cliente isolado numa partição real

```bash
python scripts/test_taco_client.py
```

Confirma que um cliente treina sobre uma partição TACO real e que o round-trip dos
355 tensores de peso é exato. Saída: `results/client_taco_smoke/result.json`.

---

## 5. Benchmark de hardware

**Cada integrante roda na própria máquina** antes de aceitar execuções da grade.

```bash
# NVIDIA
python scripts/sanity_check.py --epochs 1 --imgsz 640 --batch 16 --device 0
# Apple Silicon
python scripts/sanity_check.py --epochs 1 --imgsz 640 --batch 16 --device mps
# CPU
python scripts/sanity_check.py --epochs 1 --imgsz 640 --batch 16 --device cpu
```

Mede o tempo por época sobre `data/partitions/iid/client_0/data.yaml` (192 imagens
de treino) e grava `results/hardware_benchmark/measurement.json`.

### Estimativa por execução completa

Uma execução (50 rounds × 5 épocas locais × 5 clientes) equivale a 1.250
"época-equivalentes" de treino:

```
tempo_por_época × 1.250
```

### RESULTADOS DE BENCHMARK REAIS JÁ OBTIDOS

| Pessoa | s/época | Tempo por execução | Grade completa solo (42 execuções) |
|---|---:|---:|---:|
| Camila | 20,28s | ~7,0h | ~12 dias |
| Guilherme | 18,60s | ~6,5h | ~11 dias |
| Anabelly | *(rodar antes de pegar sua fatia — Seção 6.4)* | — | — |

> **Anabelly: rode o benchmark antes de seguir para a Seção 6.4.** A divisão de
> carga assume um tempo por execução parecido com o de Camila/Guilherme
> (~18-20s/época). Se sua GPU for muito mais lenta ou mais rápida, avise o grupo
> antes de começar — a fatia de 10 execuções pode precisar ser rebalanceada.

---

## 6. Experimentos científicos

> ⚠️ Só rode depois que os passos 2 a 5 estiverem OK. São dezenas de horas de GPU.

### 6.1 Baseline centralizada (referência de comparação) _NAO PRECISA RODAR DE NOVO_

> Já concluída — ver `results/baseline_centralized/`. **----> Não precisa rodar de novo <----** , a
> menos que o dataset ou o modelo mudem.

> ```bash
> python scripts/train_centralized.py --epochs 50 --imgsz 640 --batch 16 --device 0
> python scripts/analyze_centralized_baseline.py
> ```

### 6.2 O que falta rodar agora: FedAvg + varredura de FedProx

A grade completa do artigo tem 42 execuções (FedAvg + FedProx + FedTrimmed).

O fedprox será feito com a varredura dos termos 0.01, 0.1, 0.2, 0.5, 1
0.01, 0.1, 0.2, 0.5, 1

**FedTrimmed é da Sprint 3** (ainda não implementado) — o que dá para rodar agora é o FedAvg

| Config | μ | Execuções | Status |
|---|---:|---:|---|
| `configs/baseline_taco.yaml` | 0 (= FedAvg) | 6 | Pronto |
| **Total** | | **30** | |

Cada config gera 6 execuções (IID × 3 repetições + non-IID × 3 repetições) em
`results/<output_dir>/<cenário>_rep<n>/`, sempre no mesmo formato — os 4 configs
novos são clones de `baseline_taco.yaml`, só mudando `mu` e `output_dir`.

### 6.3 Rodar uma fatia manualmente

Sintaxe geral:

```bash
# Um config inteiro (6 execuções)
python run_config.py --config configs/<nome>.yaml --all

# Uma execução específica
python run_config.py --config configs/<nome>.yaml --scenario iid --repetition 1
python run_config.py --config configs/<nome>.yaml --scenario non_iid --repetition 2
```

### 6.4 Divisão entre Camila, Guilherme e Anabelly

As 30 execuções ficam divididas em **10 para cada pessoa**, com base no benchmark
da Seção 5. Cada execução é independente — pode rodar em qualquer ordem, pausar
entre uma e outra, e retomar depois (o runner pula execuções já `complete`).

**Camila — `configs/baseline_taco.yaml` inteiro + metade do `fedprox_mu0_001`:**

```bash
python run_config.py --config configs/baseline_taco.yaml --all
python run_config.py --config configs/fedprox_mu0_001.yaml --scenario iid --repetition 1
python run_config.py --config configs/fedprox_mu0_001.yaml --scenario iid --repetition 2
python run_config.py --config configs/fedprox_mu0_001.yaml --scenario non_iid --repetition 1
python run_config.py --config configs/fedprox_mu0_001.yaml --scenario non_iid --repetition 2
```
Status Camila: 

[X] FedAvg Completo
[] Metade do Fedprox 0.01

---

**Guilherme — resto do `fedprox_mu0_001` + `fedprox_mu0_01` inteiro + metade do `fedprox_mu0_1`:**

```bash
python run_config.py --config configs/fedprox_mu0_001.yaml --scenario iid --repetition 3
python run_config.py --config configs/fedprox_mu0_001.yaml --scenario non_iid --repetition 3
python run_config.py --config configs/fedprox_mu0_01.yaml --all
python run_config.py --config configs/fedprox_mu0_1.yaml --scenario iid --repetition 1
python run_config.py --config configs/fedprox_mu0_1.yaml --scenario iid --repetition 2
```

**Anabelly — resto do `fedprox_mu0_1` + `fedprox_mu1` inteiro:**

```bash
python run_config.py --config configs/fedprox_mu0_1.yaml --scenario iid --repetition 3
python run_config.py --config configs/fedprox_mu0_1.yaml --scenario non_iid --repetition 1
python run_config.py --config configs/fedprox_mu0_1.yaml --scenario non_iid --repetition 2
python run_config.py --config configs/fedprox_mu0_1.yaml --scenario non_iid --repetition 3
python run_config.py --config configs/fedprox_mu1.yaml --all
```

Com os tempos medidos (Seção 5), cada fatia de 10 execuções leva **cerca de 3
dias de GPU rodando sem parar** — contra ~12 dias se uma pessoa só tentasse a
grade inteira sozinha. Isso pressupõe a máquina ligada e sem hibernar durante o
treino; se você precisa usar o computador para outra coisa no meio, o runner
retoma de onde parou (execução por execução), só não pausa uma execução em
andamento.

### 6.5 Sincronizando resultados entre as três máquinas

Cada execução escreve num subdiretório próprio
(`results/<config>/<cenário>_rep<n>/`) — como ninguém escreve na pasta de outra
pessoa, não há conflito de merge. Depois de terminar sua fatia:

```bash
git add results/
git status   # confira que NAO esta adicionando final.pt (checkpoints, pesados)
git commit -m "resultados: <seu nome>, <config>, 10 execucoes"
git push
```

As outras duas pessoas rodam `git pull` para ver os resultados de todo mundo. Se
`final.pt` aparecer no `git status` para commit, adicione `results/**/final.pt` ao
`.gitignore` antes de commitar — são checkpoints de ~6 MB cada, 30 deles não
compensam versionar.

Depois que as 30 execuções estiverem no repositório (de qualquer uma das três
máquinas), gere o resumo consolidado:

```bash
python scripts/validate_baseline.py
```

### 6.6 FedTrimmed (Sprint 3, ainda não roda)

Depende da implementação do card de FedTrimmed. Quando pronto, deve seguir o
mesmo padrão: um `configs/fedtrimmed.yaml` clonado de `baseline_taco.yaml`, mais 6
execuções, divididas entre as três máquinas do mesmo jeito.

---

## 7. Validação de artefatos

Depois de qualquer lote de execuções:

```bash
python scripts/validate_baseline.py
```

Verifica que as execuções geraram todos os artefatos exigidos e que as curvas são
consistentes.

---

## 8. Solução de problemas

**`FileNotFoundError: Particao TACO nao encontrada`**
O passo 2 não foi concluído nessa máquina. Rode
`python taco_data/scripts/prepare_experiments.py`.

**`FileExistsError: Destino ja existe e nao e link`**
Uma execução anterior deixou `data/` sujo. Apague `data/centralized/` e
`data/partitions/` e rode `prepare_experiments.py` de novo. Não apague
`taco_data/data/raw/` — é o download de 2,7 GB.

**Métricas idênticas em todos os rounds**
Sinal do bug de sincronização de pesos. Confirme que `_sync_trained_weights` existe
em `fl/model.py` e que `python -m pytest` passa. Resultados assim são inválidos.

**Loss não muda / otimizador nunca dá passo**
Com partições pequenas, o acumulo padrão do Ultralytics (`nbs=64`) pode nunca
atingir o número de iterações necessário. Nesses casos, defina `nbs` igual ao
`batch_size` no config. Ver `fl/CONTRACT.md`.

**Out of memory na GPU**
Reduza `batch_size` de 16 para 8 e mantenha `nbs: 64`. Registre a mudança no
`config.json` da execução — batch diferente muda a comparabilidade. Se você mudar
isso, avise as outras duas pessoas: as execuções deixam de ser comparáveis entre
si se o batch size divergir entre máquinas.

**`[skip] <run_id> ja esta completo` mas eu queria refazer**
Use `--force`:
```bash
python run_config.py --config configs/<nome>.yaml --scenario iid --repetition 1 --force
```

**Resultado de outra pessoa não aparece depois do `git pull`**
Confirme que ela deu `git push` (Seção 6.5) e que você está na mesma branch.
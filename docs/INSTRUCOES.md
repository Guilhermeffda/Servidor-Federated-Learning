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

se `CUDA: True` confirma que o PyTorch está vendo a GPU NVIDIA corretamente Guarde o resultado — ele define o valor de `device` nos passos seguintes:

Em `Servidor-Federated-Learning\configs\taco_smoke.yaml`, ajustar `device:` (linha 17) para o device correspondente:

| Hardware | `device` |
|---|---|
| NVIDIA | `"0"` |
| Apple Silicon | `"mps"` |
| Sem GPU | `"cpu"` |

---

## 2. Preparação dos dados

**Roda uma vez por máquina.** Baixa ~2,7 GB e materializa todos os splits.

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

Produz a comparação entre α = 0,1 / 0,5 / 1,0 com métricas de heterogeneidade e a
figura vetorial usada na seção de Materiais e Métodos:

```bash
python taco_data/scripts/partition_non_iid.py --analyze
```

Saídas: `taco_data/data/partitions/non_iid/alpha_analysis.json` e
`alpha_comparison.svg`.

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

Rode sempre que mexer no pipeline, **antes** de comprometer horas de GPU.

### 4.1 Smoke test do pipeline TACO (minutos)

```bash
python run_config.py --all
```

Usa `configs/taco_smoke.yaml` (partições TACO reais, 5 clientes, 2 rounds, CPU).
Saída em `results/taco_smoke/`, marcada `scientific_valid: false`.

### 4.2 Smoke test sobre as partições TACO reais (minutos)

**Este é o teste que importa** — valida partições → treino local → agregação →
avaliação no teste global:

```bash
python run_config.py --config configs/taco_smoke.yaml --scenario iid --repetition 1
```

5 clientes, 2 rounds, 416 px, CPU. Saída em `results/taco_smoke/`.

### 4.3 Cliente isolado numa partição real

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

### Estimativa da grade completa

```
tempo_por_época × 5 épocas × 50 rounds × 5 clientes × 42 execuções
```

Referências estimadas: **160 s/época em CPU** (i7-1165G7) → 97 dias;
**50 s/época em GPU** (M1 Pro/MPS) → 30 dias. Anote o seu número e leve ao grupo —
a divisão das execuções depende disso.


#### RESULTADOS DE BENCHMARK REAIS JA OBTIDOS:

| Pessoa | s/epoca | tempo estimado |
|---|---:|---:|
| GPU CAMILA | 20,28s | 12 dias |
| GPU GUILHERME | 18,60s | 11 dias |

---

## 6. Experimentos científicos

> ⚠️ Só rode depois que os passos 2 a 5 estiverem OK. São dezenas de horas de GPU.

Antes de começar, ajuste `device` em `configs/baseline_taco.yaml` para o hardware
da sua máquina (`"0"`, `"mps"` ou `"cpu"`).

### 6.1 Baseline centralizada (referência de comparação)

Treina YOLOv8n com todos os dados juntos, sem federação:

```bash
python scripts/train_centralized.py --epochs 50 --imgsz 640 --batch 16 --device 0
```

Saídas em `results/baseline_centralized/`: `metrics.json`, `training_status.json`,
`training_curve.csv`, `best.pt`.


Obter resultados do baseline:

```bash
python scripts/analyze_centralized_baseline.py
```

### 6.2 Baselines federadas FedAvg — 6 execuções

```bash
# Tudo de uma vez (IID ×3 + non-IID ×3)
python run_config.py --config configs/baseline_taco.yaml --all

# Ou individualmente, para dividir entre máquinas:
python run_config.py --config configs/baseline_taco.yaml --scenario iid --repetition 1
python run_config.py --config configs/baseline_taco.yaml --scenario iid --repetition 2
python run_config.py --config configs/baseline_taco.yaml --scenario iid --repetition 3
python run_config.py --config configs/baseline_taco.yaml --scenario non_iid --repetition 1
python run_config.py --config configs/baseline_taco.yaml --scenario non_iid --repetition 2
python run_config.py --config configs/baseline_taco.yaml --scenario non_iid --repetition 3
```

Saída por execução em `results/baseline_taco/<cenário>_rep<n>/`:

| Arquivo | Conteúdo |
|---|---|
| `config.json` | configuração e seed resolvidas |
| `rounds.csv` | métricas globais por round (inclui round 0) |
| `clients.csv` | loss local e nº de exemplos por cliente/round |
| `convergence.png` | curva de convergência |
| `status.json` | estado, duração, versões, métrica final |
| `final.pt` | checkpoint agregado (não versionado) |

Na raiz do `output_dir`: `summary.csv` (as 6 execuções) e `aggregate.csv`
(média ± desvio por cenário).

### 6.3 FedProx com μ fixo

O termo proximal já está implementado. Para rodar, copie
`configs/baseline_taco.yaml`, mude `mu` e `output_dir`:

```bash
cp configs/baseline_taco.yaml configs/fedprox_mu001.yaml
# edite: mu: 0.01   e   output_dir: results/fedprox_mu001
python run_config.py --config configs/fedprox_mu001.yaml --all
```

A varredura completa (μ ∈ {0, 0.001, 0.01, 0.1, 1}) é da Sprint 3.

### Sobre o fallback usando collab

Rode tudo local, seguindo a seção 6 do INSTRUCOES.md, com device selecionado no configs/baseline_taco.yaml. O notebook do Colab só entra em cena se, na hora de dividir as 42 execuções entre os 5 integrantes (o bloqueio do P5 que mencionei), alguém não tiver GPU própria — aí essa pessoa roda a fatia dela no Colab em vez de ficar travada em CPU.

### Execução no Colab

> `notebooks/P4_sprint2_colab.ipynb` executa a baseline centralizada e as seis
> federadas. Monte o Google Drive para persistir resultados entre sessões — o Colab
> desconecta por inatividade.

---

## 7. Validação de artefatos

Depois de qualquer lote de execuções:

```bash
python scripts/validate_baseline.py
```

Verifica que as 6 execuções geraram todos os artefatos exigidos e que as curvas são
consistentes.

---

## 8. Solução de problemas

**`FileNotFoundError: Particao TACO nao encontrada`**
O passo 2 não foi concluído. Rode `python taco_data/scripts/prepare_experiments.py`.

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
`config.json` da execução — batch diferente muda a comparabilidade.

**Sessão do Colab caiu no meio**
Cada execução (`--scenario X --repetition N`) é independente. Rode só as que
faltaram; não precisa refazer as concluídas.
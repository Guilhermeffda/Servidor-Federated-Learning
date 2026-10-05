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

Se `CUDA: True`, o PyTorch está vendo uma GPU via CUDA **ou ROCm**. Guarde o
resultado — ele define o valor de `device` nos passos seguintes.

Em `configs/taco_smoke.yaml` (linha `device:`) e em cada `configs/*.yaml` que você
for rodar (Seção 6), ajuste `device` conforme seu hardware:

| Hardware | `device` |
|---|---|
| NVIDIA | `"0"` |
| AMD Radeon / ROCm | `"0"` |
| Apple Silicon | `"mps"` |
| Sem GPU | `"cpu"` |

### Windows nativo + AMD Radeon RX 7600 (gfx1102)

Use Python 3.12 x64 e caminho curto fora do OneDrive. Não instale wheels CUDA
nesse ambiente. ROCm usa as APIs `torch.cuda.*` e o Ultralytics pode mostrar
`CUDA:0 (AMD Radeon RX 7600)`; `torch.version.hip is not None` distingue ROCm.

Baseline de instalação: ROCm 10.0.0, torch 2.13.0+rocm10.0.0,
torchvision 0.28.0+rocm10.0.0. Fonte: [instalação oficial AMD](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
Confira os requisitos de driver da AMD antes de reproduzir em outra máquina.
Os gates abaixo são necessários para declarar o ambiente operacional.

Exemplo com Python instalado em `F:\Python312` e checkout em `F:\sfl\project`:

```powershell
New-Item -ItemType Directory -Force F:\rocm-temp,F:\pip-cache | Out-Null
$env:TEMP="F:\rocm-temp"
$env:TMP="F:\rocm-temp"
$env:PIP_CACHE_DIR="F:\pip-cache"
$env:PYTHONUTF8="1"
$env:YOLO_CONFIG_DIR="F:\sfl\yolo-config"
Set-Location F:\sfl\project
& F:\Python312\python.exe -m venv F:\sfl\.venv
& F:\sfl\.venv\Scripts\Activate.ps1
python --version
python -c "import tempfile; print(tempfile.gettempdir())"
python -m pip install --no-cache-dir --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,device-gfx1102]==10.0.0" "torch[device-gfx1102]==2.13.0+rocm10.0.0" "torchvision[device-gfx1102]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
python -m pip install --no-cache-dir -c constraints-windows-rocm.txt -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Use sempre as constraints ao instalar dependências nesse ambiente. Sem elas,
faixas mínimas de versões não garantem que o resolver preserve a build ROCm.
Após a validação completa, fixamos também Ultralytics 8.4.164 e Flower 1.38.0
nessas constraints AMD. O arquivo de requisitos genérico permanece multiplataforma.
Não use `device-all`: instale somente gfx1102. Torchaudio não participa do treino.

Antes dos testes do projeto, reproduza o comportamento sem compatibilidade:

```powershell
python scripts/check_accelerator.py --raw
python scripts/check_accelerator.py --output results/accelerator_check.json
python -m pytest
```

O repro original falha em MIOpen BatchNorm com `HIPRTC_ERROR_COMPILATION` e
`'type_traits' file not found`. `fl/runtime.py` limita o fallback ao dispatch de
BatchNorm em Windows + HIP disponível + GPU disponível + tensor GPU no contexto
de execução. A operação `torch.native_batch_norm` preserva autograd, buffers e
parâmetros; convoluções continuam em MIOpen. O contexto restaura o callable até
em exceções e usa ContextVar para isolar threads. Não altera flags globais.
NVIDIA, CPU, MPS e Linux ROCm mantêm seu caminho original. JIT/torch.compile e
SyncBatchNorm distribuído não fazem parte deste pipeline YOLO eager validado.

Também foi reproduzido um treino curto que terminou sem alterar pesos: os três
passos do otimizador foram descartados por overflow de gradientes com a escala
inicial AMP 65536. No caminho Windows ROCm, novos treinos limitam essa escala
numérica a 1024 via a API pública do GradScaler. A loss é desescalada antes do
passo; AMP, loss, learning rate, batch, nbs e épocas permanecem iguais. Os três
passos então tiveram gradientes finitos e atualizaram 183 parâmetros treináveis.
Checkpoints retomados preservam sua escala salva. Esse ajuste é registrado
explicitamente para reprodução, sem modificar configs científicos.

Compatibilidade FedProx: o loss vetorial do Ultralytics é somado antes de adicionar
o termo proximal escalar, garantindo `loss_detecção + mu/2 * ||w-w_global||²`.
Isso corrige um bug anterior de broadcasting que aplicava 3×mu nesta versão.
Essa correção matemática vale em todos os hardwares e está registrada no
[relatório de validação](VALIDACAO_WINDOWS_ROCM.md); demais adaptações são
restritas ao backend Windows ROCm.

A [documentação MIOpen](https://rocm.docs.amd.com/projects/MIOpen/en/docs-7.2.2/how-to/use-nhwc-batchnorm-in-pytorch.html)
documenta o fallback nativo usando `cudnn.flags(enabled=False)`. No build testado,
o argumento `cudnn_enabled=False` sozinho não evitou MIOpen; por isso usamos a
operação ATen nativa sem alterar a flag global. Não instale Visual Studio para
resolver esse erro antes de executar o repro e o preflight.

Após preparar os dados (Seção 2), execute nesta ordem:

```powershell
python scripts/test_taco_client.py --device 0
python run_config.py --config configs/taco_smoke.yaml --scenario iid --repetition 1 --force
python scripts/validate_run_format.py --candidate results/taco_smoke/iid_rep1
python scripts/sanity_check.py --epochs 1 --imgsz 640 --batch 16 --device 0
# Regressão adicional FedProx real, sem alterar configs científicos:
python scripts/test_taco_client.py --device 0 --mu 1
```

Detecção ou inferência não provam treino: preflight, testes, cliente, smoke,
artefatos e benchmark 640/batch16 precisam passar. Não interprete resultados
antigos versionados de outras máquinas como evidência da RX 7600.

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

As imagens usam **symlinks** para `taco_data/data/raw/` quando permitido. No
Windows, sem privilégio de symlink (erro 1314), o preparador usa cópia segura.
Não mova a pasta `raw` depois de materializar links.

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

Todos os testes devem passar. Cobrem: média ponderada do FedAvg, contrato do
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

### 6.2 O que falta rodar agora: varredura de FedProx

FedAvg já rodou (6/6). Falta a varredura de FedProx, μ ∈ {0,01 / 0,1 / 0,2 / 0,5 / 1}
— 5 valores × 6 execuções (IID×3 + non-IID×3) = **30 execuções**.

| Config | μ | Execuções | Onda | Responsável |
|---|---:|---:|---|---|
| `configs/fedprox_mu0_01.yaml` | 0,01 | 6 | 1 (prioridade) | Camila |
| `configs/fedprox_mu0_2.yaml` | 0,2 | 6 | 1 (prioridade) | Guilherme |
| `configs/fedprox_mu1.yaml` | 1,0 | 6 | 1 (prioridade) | Anabelly |
| `configs/fedprox_mu0_1.yaml` | 0,1 | 6 | 2 | veremos depois |
| `configs/fedprox_mu0_5.yaml` | 0,5 | 6 | 2 | veremos depois |

A Onda 1 cobre o primeiro, o do meio e o último valor do grid — dá uma leitura
inicial da forma da curva (μ pequeno vs. médio vs. grande) antes de preencher os
pontos intermediários na Onda 2.

### 6.3 Rodar uma fatia manualmente

```bash
# Um config inteiro (6 execuções)
python run_config.py --config configs/<nome>.yaml --all

# Uma execução específica — prefira isso a --all se a energia na sua região não
# for confiável (ver Seção 8, "Recuperação após queda de energia")
python run_config.py --config configs/<nome>.yaml --scenario iid --repetition 1
python run_config.py --config configs/<nome>.yaml --scenario non_iid --repetition 2
```

### 6.4 Onda 1 — um FedProx inteiro para cada pessoa

```bash
# Camila
python run_config.py --config configs/fedprox_mu0_01.yaml --all

# Guilherme
python run_config.py --config configs/fedprox_mu0_2.yaml --all

# Anabelly
python run_config.py --config configs/fedprox_mu1.yaml --all
```

18 execuções no total, 6 por pessoa, ~3 dias de GPU contínua cada (Seção 5). Cada
`--all` já pula sozinho qualquer execução que porventura já esteja `complete`.

### 6.5 Onda 2 — μ = 0,1 e μ = 0,5

Só começa depois que a Onda 1 terminar e a curva μ pequeno/médio/grande já der
alguma leitura. Divisão a combinar quando chegar lá — provavelmente uma pessoa fica
com μ=0,1 inteiro (6 execuções) e outra com μ=0,5 inteiro (6 execuções), e a
terceira ajuda a rodar `scripts/validate_baseline.py` e a consolidar os resultados
da Onda 1 enquanto isso.

### 6.6 Sincronizando resultados entre as três máquinas

Cada execução escreve num subdiretório próprio
(`results/<config>/<cenário>_rep<n>/`) — como ninguém escreve na pasta de outra
pessoa, não há conflito de merge. Depois de terminar sua fatia:

```bash
git add results/
git status   # confira que NAO esta adicionando final.pt (checkpoints, pesados)
git commit -m "resultados: <seu nome>, fedprox mu=<valor>, 6 execucoes"
git push
```

As outras duas pessoas rodam `git pull` para ver os resultados de todo mundo. Se
`final.pt` aparecer no `git status`, adicione `results/**/final.pt` ao
`.gitignore` antes de commitar.

### 6.7 FedTrimmed (Sprint 3 — concluído)

Implementado: `fl/server.py::aggregate_fedtrimmed` (trimmed mean coordenada a
coordenada, β=0,4 → k=1 de cada lado com 5 clientes, média ponderada por
`num_examples` dos 3 restantes) e a Strategy `FedTrimmed(FedAvg)` que a usa em
`aggregate_fit`. Smoke test (`configs/fedtrimmed_smoke.yaml`) validado
ponta a ponta em IID e non-IID.

**As 6 execuções científicas já rodaram** (`configs/fedtrimmed.yaml`, GPU
Guilherme, 27/09/2026) e estão em `results/fedtrimmed/` — validadas por
`scripts/validate_baseline.py --results results/fedtrimmed`. mAP50 médio final:
0,268 (IID, σ=0,019) e 0,250 (non-IID, σ=0,009). Não precisa rodar de novo.

Config científico usado, clonado de `baseline_taco.yaml` (mesma partição, mesmo
modelo, mesmos rounds/épocas/augmentation; difere em `strategy: fedtrimmed`,
`beta: 0.4` e `output_dir`):

```bash
python run_config.py --config configs/fedtrimmed.yaml --all
```

6 execuções (IID×3 + non-IID×3); caso precise refazer, a dividir entre as três máquinas do mesmo jeito
que o FedProx (Seção 6.4).

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

**Recuperação após queda de energia**
`run_config.py` só marca uma execução como recuperável (`[skip] ... ja esta
completo`) depois que ela termina os 50 rounds inteiros — não existe checkpoint
por round. Se a energia cair no meio de uma execução, ao rodar de novo ela
reinicia do round 1, perdendo o tempo já gasto nessa execução específica (as
outras execuções já completas do mesmo `--all` continuam sendo puladas
normalmente). Para limitar o prejuízo de uma queda, prefira rodar execução por
execução (`--scenario X --repetition N`) em vez de `--all` de uma vez, e considere
um nobreak se a energia na sua região for instável — 6-7h é bastante tempo de
exposição por execução.

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
Pare e registre erro, VRAM disponível, pico de memória e batch testado. Não
reduza batch automaticamente: isso exige decisão científica do grupo e uma
configuração explicitamente documentada. No caminho Windows ROCm, o wrapper
impede o retry do Ultralytics que reduziria o batch silenciosamente.

**`[skip] <run_id> ja esta completo` mas eu queria refazer**
Use `--force`:
```bash
python run_config.py --config configs/<nome>.yaml --scenario iid --repetition 1 --force
```

**Resultado de outra pessoa não aparece depois do `git pull`**
Confirme que ela deu `git push` (Seção 6.5) e que você está na mesma branch.

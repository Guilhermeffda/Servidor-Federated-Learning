# FedLitter — FL Core (Cliente)

Aprendizado Federado para detecção e mapeamento de descarte irregular de resíduos em
vias urbanas, via câmeras veiculares. Este repositório é o **FL Core** do projeto (par
servidor/cliente Flower); este README documenta especificamente a parte do **P4
(Rodrigo) — FL Core Cliente**, seguindo o planejamento oficial do quadro Trello.

## Sobre o projeto (contexto geral)

**Pergunta de pesquisa:** em que medida um modelo de detecção de descarte irregular de
resíduos, treinado via Aprendizado Federado (FedAvg) sobre uma frota simulada de
veículos de borda, mantém qualidade comparável a um treinamento centralizado — e
permanece robusto sob conectividade intermitente dos clientes — sem que nenhuma imagem
bruta seja centralizada?

- **Dataset:** [pLitterStreet](https://arxiv.org/pdf/2401.14719) (Mandhati et al., 2024)
  — 13.000+ imagens de lixo/descarte em vias urbanas, coletadas por câmeras veiculares
  (Tailândia/Sri Lanka/Vietnã), anotações em **COCO JSON**, com metadados de
  geolocalização. Distribuído via [Zenodo](https://zenodo.org/records/8288500)
  (`images.zip`, ~13,7 GB — não há amostra pequena oficial separada).
- **Modelo:** YOLOv8n (Ultralytics), transfer learning a partir de `yolov8n.pt`
  (pré-treinado em COCO).
- **Arquitetura:** cada veículo é um cliente FL. Treina YOLOv8n localmente sobre as
  imagens capturadas e envia **apenas os pesos do modelo** ao servidor — a imagem bruta
  nunca é centralizada. O servidor agrega via **FedAvg** nativo do Flower (McMahan et
  al., 2017) — sem reimplementação própria do algoritmo.
- **Simulação:** 3 a 5 "clientes virtuais" (cada um = uma região/rota de coleta),
  tudo rodando numa única máquina via **Flower + PyTorch**. **Windows nativo, sem
  WSL/Docker/Linux.** TensorFlow Federated está proibido no projeto (sem suporte nativo
  a Windows) — decisão fechada pelo grupo: exclusivamente Flower + PyTorch.

### Equipe (P1–P5)

| Pessoa | Papel | Responsabilidade central |
|---|---|---|
| P1 | Literatura, validação e escrita do artigo |
| P2 | Dados (download, inspeção, particionamento, GeoJSON) |
| P3 | Servidor FL (`FLServer`, FedAvg, integração) — ver [`server.py`](server.py) |
| **P4 (eu)** | **Cliente FL (`FLClient`, YOLOv8n, treino local, baseline centralizado, experimentos)** |
| P5 | Dashboard |

### O que fica fora do escopo (P4 e geral)

- Comparação entre algoritmos de agregação (só FedAvg — não reimplementar do zero).
- Heterogeneidade estatística entre clientes como eixo central de pesquisa.
- Implantação em hardware físico real — tudo é simulado em software.
- WSL, Docker ou TensorFlow Federated.

## Cronograma

| Sprint | Início | Fim | Foco relevante para P4 |
|---|---|---|---|
| **1 — Kickoff e Dados** | 12/08 | 27/08 | **Em andamento** — setup YOLOv8n, esqueleto do `FLClient`, baseline centralizado |
| 2 — Pipeline FL Funcional | 28/08 | 12/09 | Implementar `FLClient` de verdade (pesos, treino, simulação de 3-5 clientes) |
| 3 — Experimentos e Dashboard Avançado | 13/09 | 03/10 | Dropout/conectividade intermitente, experimentos |
| 4 — Integração e Redação | 04/10 | 24/10 | Integração com servidor/dashboard; escrever seção do artigo |
| 5 — Validação e Ajustes Finais | 25/10 | **31/10 (entrega)** | Ajustes finais e revisão conjunta |

## Ambiente e restrições técnicas

- **Windows nativo**, Python (ver [pendência de versão](#pendências-e-decisões-abertas)),
  virtualenv em `C:\venvs\fedlitter` (não em `.venv` dentro do projeto — ver nota abaixo).
- Dependências em [`requirements.txt`](requirements.txt), fixadas via `pip freeze`.

## Como instalar

```bash
python -m venv C:\venvs\fedlitter
C:\venvs\fedlitter\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ No Windows, se a instalação do `torch` falhar com
> `OSError: [WinError 206] O nome do arquivo ou a extensão é muito grande`, é o limite de
> caminho do Windows (o pacote tem arquivos de licença com caminhos bem profundos). É por
> isso que o venv fica em `C:\venvs\fedlitter` (caminho curto) em vez de dentro do
> projeto — criar o venv num caminho longo (`Documents\...\Servidor-Federated-Learning\.venv`)
> reproduz o erro.

## Como usar

```bash
# Card 1 — sanity check do ambiente YOLOv8n (mede tempo/época nesta máquina)
python scripts\sanity_check.py

# Subir o servidor (P3)
python server.py
```

`fl/client.py` ainda é um esqueleto (Card 2, sem lógica de treino — ver seção de
status). A versão completa e já testada (pesos reais, treino local, simulação de
frota, dropout) está guardada em [`sprint2_preview/`](sprint2_preview/README.md) até o
Sprint 2 ser autorizado.

## Estrutura do repositório

```
Servidor-Federated-Learning/
├── server.py                  # Servidor Flower (P3) — FedAvg, orquestração de rounds
├── fl/
│   ├── __init__.py
│   ├── client.py                # FLClient (P4) — esqueleto Sprint 1, Card 2
│   └── CONTRACT.md               # Contrato de interface servidor↔cliente
├── scripts/
│   └── sanity_check.py            # Card 1 — valida ambiente YOLOv8n + benchmark
├── data/
│   └── mini_test/                  # Mini-subset PROVISÓRIO (coco8, não é pLitterStreet)
├── notes/
│   └── hardware_benchmark.md        # Tempo/época medido (Card 1)
├── results/
│   └── baseline_centralized/         # Métricas do baseline (Card 3, pendente)
├── tests/                              # (vazio por enquanto)
├── sprint2_preview/                     # FLClient completo, testado, aguardando Sprint 2
│   └── README.md                         # Explica o que tem aqui e por quê
├── requirements.txt
└── README.md
```

## Links úteis

- [Planejamento (Overleaf)](https://www.overleaf.com/read/dbdqrzwmjdgq#9e4951)
- [Quadro Trello](https://trello.com/b/xLPh4nGW/fedlitter)
- [Planilha de referências](https://docs.google.com/spreadsheets/d/1bzcV3RmFYEOkHI2ypBLuRrPeney1tr_KgJ7XCtrOoaE/edit?usp=sharing)
- [Trabalho anterior de Federated Learning (Drive)](https://drive.google.com/drive/u/1/folders/1j6V4jQKRuyRWsNrHqz93fhAWWd6EtNk6)
- [Artigo do dataset (pLitterStreet, arXiv)](https://arxiv.org/pdf/2401.14719)
- [Dataset (Zenodo)](https://zenodo.org/records/8288500) · [Código oficial (GitHub)](https://github.com/gicait/pLitter)

## Status — Sprint 1

🟡 Em andamento. Progresso por card, com DoD explícito:

**Card 1 — Setup YOLOv8n local**
- [x] Treino mínimo completa 3 épocas sem erro
- [x] Pesos salvos e carregáveis (`YOLO(best.pt)` + `model.val()` funcionaram)
- [x] Tempo por época documentado em [`notes/hardware_benchmark.md`](notes/hardware_benchmark.md)
  (~13,1s/época) — **mas com ressalva**: usa o `coco8` (8 imagens), não o pLitterStreet
  real, então o número não é representativo do custo real. Precisa remedir quando o
  mini-subset de verdade existir.

**Card 2 — Esqueleto vazio do FLClient**
- [x] `fl/client.py` existe e importa sem erro (testado)
- [x] Estrutura compatível com a interface `NumPyClient` esperada pelo `server.py`
- Lógica real (não faz parte deste card) guardada em [`sprint2_preview/`](sprint2_preview/README.md)

**Card 3 — Baseline centralizado**
- ⛔ **Bloqueado.** Depende do dataset completo do pLitterStreet. O P2 tem as imagens
  localmente mas ainda não commitou/compartilhou. Não baixei o dataset completo do
  Zenodo (13,7 GB, um único `images.zip`) porque duplicaria o trabalho do P2 — aguardando
  ele disponibilizar.

## Pendências e decisões abertas

- **Versão do Python:** o brief pede 3.11; esta máquina só tem 3.12 instalado (sem 3.11
  disponível). Decisão do grupo: seguir com 3.12 por ora (ambiente já validado), com essa
  divergência documentada aqui.
- **Mini-subset real (Card 1) e dataset completo (Card 3):** aguardando P2 subir as
  imagens do pLitterStreet.
- Itens em aberto do contrato com P3 — ver [`fl/CONTRACT.md`](fl/CONTRACT.md).

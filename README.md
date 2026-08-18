# FedLitter — FL Core (Cliente)

Aprendizado Federado para detecção e mapeamento de descarte irregular de resíduos em
vias urbanas, via câmeras veiculares. Este repositório é o **FL Core** do projeto (par
servidor/cliente Flower); este README documenta especificamente a parte do **P4
(Rodrigo) — FL Core Cliente**.

## Sobre o projeto (contexto geral)

**Pergunta de pesquisa:** em que medida um modelo de detecção de descarte irregular de
resíduos, treinado via Aprendizado Federado (FedAvg) sobre uma frota simulada de
veículos de borda, mantém qualidade comparável a um treinamento centralizado — e
permanece robusto sob conectividade intermitente dos clientes — sem que nenhuma imagem
bruta seja centralizada?

- **Dataset:** [pLitterStreet](https://github.com/AI-Thailand/pLitterStreet) (Mandhati et
  al., 2024) — 13.000+ imagens de lixo/descarte em vias urbanas, coletadas por câmeras
  veiculares (Tailândia/Sri Lanka), formato COCO (compatível com YOLO), com coordenadas
  GPS reais associadas.
- **Arquitetura:** cada veículo é um cliente FL. Treina YOLOv8n localmente sobre as
  imagens capturadas e envia **apenas os pesos do modelo** ao servidor — a imagem bruta
  nunca é centralizada e pode ser descartada após o treino local. O servidor agrega via
  **FedAvg** (McMahan et al., 2017).
- **Simulação:** tudo roda em uma única máquina via **Flower + PyTorch** (Flower simula
  os clientes como processos/atores orquestrados internamente por Ray). Não há hardware
  físico, streaming de vídeo real ou rede distribuída de verdade — "dropout" e
  "conectividade intermitente" são simulados via código (ex.: sortear aleatoriamente
  quais clientes participam de cada round).

### Equipe (P1–P5)

| Pessoa | Papel | Responsabilidade central |
|---|---|---|
| P1 — Camila | Literatura & Validação | Lê/valida artigos-base, escreve trabalhos relacionados |
| P2 — Anabelly | Dados | Aquisição, limpeza e particionamento do pLitterStreet; coordenadas geográficas |
| P3 — Ferraz | FL Core (Servidor) | Servidor Flower, agregação FedAvg, orquestração de rounds — ver [`server.py`](server.py) |
| **P4 — Rodrigo (eu)** | **FL Core (Cliente)** | **Cliente Flower + YOLOv8n, simulação de múltiplos veículos, dropout/conectividade intermitente** |
| P5 — Andressa | Sistema & Dashboard | Pipeline de metadados, dashboard, baseline centralizado |

Todo mundo escreve a seção do artigo referente à própria frente; revisão final em
conjunto na Etapa 6.

## Minha parte — P4: FL Core (Cliente)

Responsabilidades:

1. **Cliente Flower** (`FLClient`) que treina um **YOLOv8n** localmente sobre a partição
   de dados de um veículo.
2. **Simulação de múltiplos veículos**: cada "veículo" é uma instância de cliente com
   sua própria fatia do dataset (por região/rota, conforme particionamento feito por P2).
3. **Robustez a conectividade intermitente / dropout de clientes** — este é o
   **experimento central do projeto** (o diferencial frente à literatura, já que não
   existe trabalho publicado combinando FL com detecção de lixo urbano). Simular veículos
   que nem sempre estão "online" para sincronizar pesos, e avaliar como isso afeta a
   qualidade do modelo agregado.

### Contrato de interface com o servidor (P3)

Definido em [`server.py`](server.py) — **confirmar com P3 antes de fechar a
implementação**:

- **Pesos trocados:** lista de `numpy.ndarray` correspondendo ao `state_dict()` do
  modelo YOLO, na ordem das chaves do `state_dict()`.
  - ⚠️ Preciso confirmar que essa ordem é estável entre execuções/clientes.
- **Métricas retornadas em `fit()`/`evaluate()`:** dict com, no mínimo,
  `{"loss": float, "num_examples": int}`, e opcionalmente `{"map50": float}`.
- **Número de clientes:** `server.py` está com `min_fit_clients` /
  `min_available_clients` como placeholder (`3`). Precisa bater com o número real de
  veículos simulados — atualizar dos dois lados se mudar.
- Ainda não há `fl/CONTRACT.md` no repositório (referenciado nos comentários de
  `server.py`, mas não commitado) — vale criar/alinhar esse arquivo com P3 como fonte
  única da verdade do contrato, em vez de manter isso apenas em comentários.

### O que fica fora do escopo (P4 e geral)

- Comparação entre algoritmos de agregação (só FedAvg).
- Heterogeneidade estatística entre clientes como eixo central de pesquisa.
- Implantação em hardware físico real — tudo é simulado em software.

## Cronograma

| Sprint | Início | Fim | Foco relevante para P4 |
|---|---|---|---|
| 1 — Kickoff e Dados | 12/08 | 27/08 | Alinhar contrato de interface com P3; entender particionamento do P2 |
| 2 — Pipeline FL Funcional | 28/08 | 12/09 | Implementar `FLClient` + YOLOv8n treinando localmente |
| 3 — Experimentos e Dashboard Avançado | 13/09 | 03/10 | Simular múltiplos veículos + dropout/conectividade intermitente |
| 4 — Integração e Redação | 04/10 | 24/10 | Integração com servidor/dashboard; escrever seção do artigo |
| 5 — Validação e Ajustes Finais | 25/10 | **31/10 (entrega)** | Ajustes finais e revisão conjunta |

## Tecnologias

- **Python**
- **Flower** — orquestração de aprendizado federado
- **PyTorch** + **Ultralytics YOLOv8n** — modelo de detecção
- **pLitterStreet** — dataset base (formato COCO)

## Como instalar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install flwr torch ultralytics
```

> `requirements.txt` ainda não existe no repo — criar conforme as dependências forem
> sendo usadas na implementação do cliente.

## Como usar

```bash
# Subir o servidor (P3)
python server.py

# Subir um cliente (a implementar)
python client.py
```

## Estrutura do repositório

```
Servidor-Federated-Learning/
├── server.py       # Servidor Flower (P3) — FedAvg, orquestração de rounds
├── client.py        # Cliente Flower + YOLOv8n (P4) — a implementar
└── README.md
```

## Links úteis

- [Planejamento (Overleaf)](https://www.overleaf.com/read/dbdqrzwmjdgq#9e4951)
- [Quadro Trello](https://trello.com/b/xLPh4nGW/fedlitter)
- [Planilha de referências](https://docs.google.com/spreadsheets/d/1bzcV3RmFYEOkHI2ypBLuRrPeney1tr_KgJ7XCtrOoaE/edit?usp=sharing)
- [Trabalho anterior de Federated Learning (Drive)](https://drive.google.com/drive/u/1/folders/1j6V4jQKRuyRWsNrHqz93fhAWWd6EtNk6)

## Status

🟡 Em desenvolvimento — Sprint 1 (Kickoff e Dados). Servidor tem estrutura inicial
(`server.py`); cliente (parte do P4) ainda não implementado.

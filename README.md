# FedLitter

Projeto de Trabalho de Conclusão de Curso sobre **Aprendizado Federado para detecção de resíduos urbanos**.

## Objetivo

Investigar o uso de Aprendizado Federado para treinar modelos de detecção de resíduos sem centralizar as imagens dos clientes.

O projeto compara diferentes estratégias de agregação (FedAvg, FedProx e FedTrimmed) e avalia o impacto delas no desempenho do modelo sob cenários de dados IID e non-IID.

## Dataset

É utilizado o dataset **TACO (Trash Annotations in Context)**.

O TACO possui originalmente 60 categorias. Para os experimentos, elas são agrupadas no esquema **TACO-10**, composto por 10 categorias:

* Bottle
* Bottle cap
* Can
* Cigarette
* Cup
* Lid
* Other Litter
* Plastic bag + wrapper
* Pop tab
* Straw

O processamento dos dados está documentado em `taco_data/data/README.md`.

## Estrutura

```text
Servidor-Federated-Learning/
├── taco_data/
│   ├── data/
│   │   ├── raw/          # não versionado — baixado pelo script
│   │   ├── processed/    # não versionado — gerado pelo script
│   │   └── test_global/  # imagens/labels não versionados; README/data.yaml versionados
│   └── scripts/
│
├── fl/            # cliente/servidor federado (em desenvolvimento)
├── server.py      # esqueleto — ainda não integrado com cliente real
└── README.md
```

## Status do projeto

> ⚠️ **Nem tudo neste repositório já está funcional de ponta a ponta.**
> Consulte o board do projeto para o status real de cada componente antes de assumir
> que algo "já funciona" só porque o arquivo existe.

- ✅ Download, inspeção e reagrupamento (TACO-10) do dataset — funcional.
- ✅ Separação do teste global (20%, estratificado, seed fixa) — funcional.
- 🚧 Partição de clientes (IID / non-IID via Dirichlet) — em desenvolvimento.
- 🚧 `server.py` (FedAvg) — esqueleto, ainda não integrado com um cliente real.
- 🚧 Cliente federado (`fl/client.py`) — esqueleto oficial; implementação de
  referência existe em `sprint2_preview/`, ainda não portada.
- ⛔ FedProx, FedTrimmed — não iniciados.

## Dados

Para baixar e preparar o TACO, rode em sequência:

```bash
python taco_data\scripts\download_dataset.py
python taco_data\scripts\inspect_dataset.py
python taco_data\scripts\regroup_categories.py
```

Isso gera, entre outros:

```text
taco_data/data/processed/taco10/annotations_regrouped.json
taco_data/data/test_global/
```

## Aprendizado Federado

O projeto utiliza uma arquitetura cliente-servidor para simular diferentes participantes do treinamento federado.

Cada cliente realiza treinamento local e envia os parâmetros do modelo ao servidor. O servidor agrega os modelos utilizando as estratégias avaliadas no projeto.

## Tecnologias

* Python
* PyTorch (build CPU, via índice próprio do PyTorch — ver seção de instalação)
* Flower
* Ultralytics (YOLOv8n)
* scikit-learn
* COCO / pycocotools

## Iniciação

```bash
cd Servidor-Federated-Learning
python -m venv C:\venvs\fedlitter
C:\venvs\fedlitter\Scripts\Activate.ps1
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python .\taco_data\scripts\create_global_test.py
```

## Como validar a instalação

```powershell
python scripts\sanity_check.py
```

# Subir o servidor (P3)
python server.py
```

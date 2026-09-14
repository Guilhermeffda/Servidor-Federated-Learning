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

Para materializar os splits YOLO, as cinco particoes IID/non-IID e o teste global:

```bash
python taco_data/scripts/prepare_experiments.py
```

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
├── fl/            # cliente, modelo, particionamento e agregação federada
├── run_config.py  # runner das baselines IID/non-IID
└── README.md
```

## Status do projeto

> ⚠️ **Nem tudo neste repositório já está funcional de ponta a ponta.**
> Consulte o board do projeto para o status real de cada componente antes de assumir
> que algo "já funciona" só porque o arquivo existe.

- ✅ Download, inspeção e reagrupamento (TACO-10) do dataset — funcional.
- ✅ Separação do teste global (20%, estratificado, seed fixa) — funcional.
- ✅ Partição reproduzível de clientes IID / non-IID — funcional no runner.
- ✅ FedAvg (`fl/server.py`) integrado ao cliente e ao runner de baseline.
- ✅ Cliente federado (`fl/client.py`) com troca de pesos e treino local YOLO.
- ✅ Termo proximal do FedProx integrado ao treino local (`mu > 0`).
- ⛔ FedTrimmed — não iniciado.

## Dados

Para baixar e preparar o TACO, rode em sequência:

```bash
python taco_data/scripts/download_dataset.py
python taco_data/scripts/inspect_dataset.py
python taco_data/scripts/regroup_categories.py
python taco_data/scripts/prepare_experiments.py
```

Isso gera, entre outros:

```text
taco_data/data/processed/taco10/annotations_regrouped.json
taco_data/data/test_global/
```

## Aprendizado Federado

O projeto utiliza uma arquitetura cliente-servidor para simular diferentes participantes do treinamento federado.

Cada cliente realiza treinamento local e envia os parâmetros do modelo ao servidor. O servidor agrega os modelos utilizando as estratégias avaliadas no projeto.

### Baseline FedAvg da Sprint 2 (P4)

O runner reproduz os cenários IID e non-IID com três repetições e grava métricas,
checkpoints e curvas em `results/baseline/`:

```bash
.venv/bin/python run_config.py --all
```

Neste checkout, `configs/baseline.yaml` aponta explicitamente para o mini-subset de
smoke test e, por isso, usa `scientific_valid: false`. Troque o `dataset_yaml` pelas
partições TACO-10 reais antes de usar os números no TCC. Consulte
`results/baseline/README.md` para o formato dos artefatos.

A medicao real de hardware determinou que os treinos finais devem rodar em GPU. O
notebook `notebooks/P4_sprint2_colab.ipynb` executa a baseline centralizada e as seis
baselines federadas TACO-10 no Colab.

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

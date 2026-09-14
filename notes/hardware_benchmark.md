# Benchmark de hardware — TACO real (Sprint 2, P4)

Medição atualizada em 14/09/2026. O valor provisório obtido com oito imagens foi
substituído por uma execução sobre uma partição IID real do TACO-10.

## Ambiente

- CPU: 11th Gen Intel Core i7-1165G7 @ 2.80 GHz, 8 threads lógicas
- Treino efetivo em CPU (`torch.cuda.is_available() == false`)
- Python 3.12.3, PyTorch 2.14.0+cpu, Ultralytics 8.4.152
- YOLOv8n, `imgsz=640`, `batch=16`, 1 época
- Dataset: `data/partitions/iid/client_0/data.yaml`
- Partição: 192 imagens de treino e 48 de validação (240 imagens no cliente)

## Resultado medido

- Tempo total medido: **160,20 segundos**
- Tempo por época: **160,20 segundos (~2,67 minutos)**
- O tempo inclui o ciclo completo executado pelo `model.train()`, inclusive a
  validação final do Ultralytics, da mesma forma que ocorrerá em cada treino local.
- Registro estruturado: `results/hardware_benchmark/measurement.json`

## Estimativa solicitada no card

```text
160,202 s/época × 5 épocas × 50 rounds × 5 clientes × 42 execuções
= 8.410.617 segundos
= 2.336,28 horas
= 97,35 dias de processamento sequencial
```

Essa é uma estimativa conservadora e não inclui paralelismo entre clientes. Mesmo
com variações de cache e overhead, a ordem de grandeza torna as 42 execuções
incompatíveis com o prazo em CPU local.

## Decisão

**Usar Google Colab com GPU para os treinos reais.** A CPU local fica restrita a
smoke tests, testes de integração e validação dos artefatos. O comando usado para a
medição foi:

```bash
.venv/bin/python scripts/sanity_check.py --epochs 1 --imgsz 640 --batch 16 --device cpu
```

Esta decisão está registrada no repositório para comunicação ao grupo. A comunicação
em canal externo (Trello/Slack/Teams) ainda deve ser feita por um integrante do grupo.

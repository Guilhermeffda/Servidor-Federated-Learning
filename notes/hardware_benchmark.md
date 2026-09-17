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

## Medição em GPU (16/09/2026)

O grupo tem GPU disponível e roda localmente no VS Code, então a medição só-CPU
acima não é mais a base de decisão. Nova medição, mesma partição e mesmos
hiperparâmetros, em GPU (Apple M1 Pro / MPS, `--device mps`):

- Tempo por época: **50,08 segundos** (3,2× mais rápido que a CPU)
- Estimativa das 42 execuções: **730,29 horas / 30,43 dias** sequenciais

```bash
.venv/bin/python scripts/sanity_check.py --epochs 1 --imgsz 640 --batch 16 --device mps
```

Uma GPU NVIDIA dedicada deve ser mais rápida que o MPS, mas mesmo otimista
(≈25 s/época) a grade completa fica em ~15 dias sequenciais.

## Decisão

**Treinar em GPU local** (o grupo já tem GPU e roda no VS Code); a CPU fica restrita
a smoke tests e validação de artefatos. O Colab continua sendo alternativa para
paralelizar execuções entre integrantes, não uma necessidade.

⚠️ **Alerta de planejamento:** o gargalo não é mais CPU vs GPU, é o **tamanho da
grade experimental**. 42 execuções × 50 rounds × 5 épocas × 5 clientes = 52.500
épocas de treino. Mesmo em GPU isso não cabe no prazo se rodar sequencialmente numa
máquina só. Opções a decidir com o grupo (cards S3 de P3/P4/P5 já preveem divisão
entre P4 e P5):

1. Paralelizar entre as máquinas dos integrantes (já planejado, mas insuficiente
   sozinho: 3 máquinas ≈ 10 dias).
2. Reduzir a grade de μ do FedProx (5 valores → 3) — o card de P4 já prevê isso como
   primeiro corte se houver atraso no checkpoint de ~20-22/09.
3. Reduzir rounds de 50 para 30, se a curva de convergência estabilizar antes.

Essa conta deve ser refeita por cada integrante na própria máquina antes de fechar a
divisão das execuções.

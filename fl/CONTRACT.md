# Contrato de interface — Servidor (P3) ↔ Cliente (P4)

Formaliza o que estava descrito nos comentários `# ALINHAR COM P4` de
[`server.py`](../server.py). Fonte única da verdade sobre o protocolo trocado entre
servidor e clientes Flower neste projeto.

## Pesos do modelo

Lista de `numpy.ndarray`, na mesma ordem das chaves de
`model.model.state_dict()` (o `state_dict()` do `DetectionModel` do Ultralytics YOLOv8n,
acessado via `YOLO(...).model`).

Implementado em [`fl/model.py`](model.py) (`get_weights` / `set_weights`). A ordem das
chaves de um `state_dict()` do PyTorch é determinística para uma mesma arquitetura de
modelo (segue a ordem de registro dos parâmetros), então é estável entre execuções e
entre clientes desde que todos usem a mesma arquitetura (`yolov8n`).

## Métricas retornadas pelo cliente

Dict retornado por `fit()` e `evaluate()` do `FLClient`, contendo no mínimo:

```python
{"loss": float, "num_examples": int}
```

e opcionalmente:

```python
{"map50": float}
```

`num_examples` é o número de imagens usadas no treino/avaliação local daquele veículo
naquele round.

## Número de clientes

`min_fit_clients` / `min_available_clients` em `server.py` estão como placeholder = 3.
Precisa ser igual ao número de veículos simulados em `simulate_fleet.py`. **Se um dos
dois lados mudar, atualizar o outro.**

## Métricas de fit agregadas

O `FedAvg` nativo do Flower não agrega `fit_metrics` por padrão — só `evaluate_metrics`.
Se quisermos ver `loss`/`map50` do treino (não só da avaliação) agregados no servidor,
P3 precisa registrar um `fit_metrics_aggregation_fn` na `Strategy` (ver TODO em
`server.py::get_strategy`).

## Simulação de dropout / conectividade intermitente

Não faz parte do contrato de payload (pesos/métricas) — é uma decisão só do lado do
cliente (ver [`fl/dropout.py`](dropout.py)). Um cliente "indisponível" em um round
simplesmente não retorna uma atualização de pesos válida para aquele round; o servidor
(via `FedAvg`) já lida com isso naturalmente, considerando apenas os clientes que
responderam.

## Pendências para alinhar com P3

- [ ] Confirmar `min_fit_clients`/`min_available_clients` final com o número real de
      veículos do experimento.
- [ ] Decidir se `fit_metrics_aggregation_fn` será implementado no servidor.
- [ ] Confirmar se o servidor espera receber `map50` em todo round de `evaluate()`, ou
      só nos rounds de avaliação centralizada.

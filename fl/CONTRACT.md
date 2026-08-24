# Contrato de interface — Servidor (P3) ↔ Cliente (P4)

Formaliza o que estava descrito nos comentários `# ALINHAR COM P4` de
[`server.py`](../server.py). Fonte única da verdade sobre o protocolo trocado entre
servidor e clientes Flower neste projeto.

## Pesos do modelo

Lista de `numpy.ndarray`, na mesma ordem das chaves de
`model.model.state_dict()` (o `state_dict()` do `DetectionModel` do Ultralytics YOLOv8n,
acessado via `YOLO(...).model`).

Implementado (Sprint 2) em [`sprint2_preview/model.py`](../sprint2_preview/model.py)
(`get_weights` / `set_weights`), já testado. A ordem das chaves de um `state_dict()` do
PyTorch é determinística para uma mesma arquitetura de modelo (segue a ordem de
registro dos parâmetros), então é estável entre execuções e entre clientes desde que
todos usem a mesma arquitetura (`yolov8n`).

## Config enviado pelo servidor a cada round (`on_fit_config_fn`)

Dict repassado ao `fit()` do cliente via o parâmetro `config`:

```python
{"local_epochs": 5, "batch_size": 16}
```

O cliente **lê esses valores de `config`** (`config.get("local_epochs", ...)`) — nunca
hardcoda hiperparâmetros de treino local. Ainda não implementado no lado do servidor
(`server.py` não define `on_fit_config_fn`) nem lido pelo `fl/client.py` (esqueleto do
Card 2, Sprint 1) — fica para o card de Sprint 2.

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
cliente (ver [`sprint2_preview/dropout.py`](../sprint2_preview/dropout.py)). Um cliente
"indisponível" em um round simplesmente não retorna uma atualização de pesos válida
para aquele round; o servidor (via `FedAvg`) já lida com isso naturalmente,
considerando apenas os clientes que responderam.

## Pendências para alinhar com P3

- [ ] Decidir se `fit_metrics_aggregation_fn` será implementado no servidor (`FedAvg`
      nativo não agrega `fit_metrics` por padrão).
- [ ] `evaluate()` sem backward pass não produz uma loss de detecção real (o
      Ultralytics só calcula box/cls/dfl loss durante o treino) — o protótipo em
      `sprint2_preview/model.py` usa `1 - map50` como proxy. Decidir se o servidor
      precisa de uma loss "de verdade" aqui.
- [ ] Confirmar se o servidor espera receber `map50` em todo round de `evaluate()`, ou
      só nos rounds de avaliação centralizada.

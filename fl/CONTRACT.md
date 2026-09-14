# Contrato de interface — Servidor (P3) ↔ Cliente (P4)

Formaliza o que estava descrito nos comentários `# ALINHAR COM P4` de
[`server.py`](../server.py). Fonte única da verdade sobre o protocolo trocado entre
servidor e clientes Flower neste projeto.

## Pesos do modelo

Lista de `numpy.ndarray`, na mesma ordem das chaves de
`model.model.state_dict()` (o `state_dict()` do `DetectionModel` do Ultralytics YOLOv8n,
acessado via `YOLO(...).model`).

Implementado em [`fl/model.py`](model.py) (`get_weights` / `set_weights`), com
validação da quantidade e do shape dos tensores. A ordem das chaves de um `state_dict()` do
PyTorch é determinística para uma mesma arquitetura de modelo (segue a ordem de
registro dos parâmetros), então é estável entre execuções e entre clientes desde que
todos usem a mesma arquitetura (`yolov8n`).

## Config enviado pelo servidor a cada round (`on_fit_config_fn`)

Dict repassado ao `fit()` do cliente via o parâmetro `config`:

```python
{"local_epochs": 5, "batch_size": 16, "mu": 0.0}
```

O cliente **lê esses valores de `config`** (`config.get("local_epochs", ...)`) — nunca
hardcoda hiperparâmetros de treino local. `run_config.py` já envia esses campos; um
servidor Flower distribuído deve usar o mesmo payload em `on_fit_config_fn`.

`mu=0` executa FedAvg. Quando `mu>0`, `fl/model.py` injeta o termo
`(mu/2) * ||w - w_global||²` no loss do Ultralytics antes do `backward`, usando os
pesos globais recebidos no início do round como referência. Trata-se do termo FedProx
no objetivo local, não de uma interpolação aproximada após o treino.

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

`fl/server.py::get_strategy` recebe o número de clientes da configuração e usa o
mesmo valor em `min_fit_clients` e `min_available_clients`.

## Métricas de fit agregadas

O `FedAvg` nativo do Flower não agrega `fit_metrics` por padrão — só `evaluate_metrics`.
Se quisermos ver `loss`/`map50` do treino (não só da avaliação) agregados no servidor,
P3 precisa registrar um `fit_metrics_aggregation_fn` na `Strategy` (ver TODO em
`server.py::get_strategy`).

## Simulação de dropout / conectividade intermitente (desativada)

Não faz parte do experimento principal da Sprint 2. O código histórico permanece em
[`sprint2_preview/dropout.py`](../sprint2_preview/dropout.py), isolado e sem imports no
cliente ativo. Isso permite reativá-lo futuramente como experimento secundário.

## Pendências para alinhar com P3

- [ ] Decidir se `fit_metrics_aggregation_fn` será necessário no servidor distribuído.
      O runner já registra as métricas locais diretamente em `clients.csv`.
- [ ] `evaluate()` sem backward pass não produz uma loss de detecção real (o
      Ultralytics só calcula box/cls/dfl loss durante o treino) — `fl/model.py` usa
      `1 - map50` como proxy. Decidir se o servidor
      precisa de uma loss "de verdade" aqui.
- [ ] Confirmar se o servidor espera receber `map50` em todo round de `evaluate()`, ou
      só nos rounds de avaliação centralizada.

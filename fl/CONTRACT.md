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
{"local_epochs": 5, "batch_size": 16, "mu": 0.0, "nbs": 64}
```

`nbs` é o batch nominal do Ultralytics: gradientes são acumulados até
`round(nbs/batch_size)` iterações antes de cada `optimizer.step()`. Com partições
pequenas (ex.: smoke test com 2 iterações/round e `nbs=64`), o otimizador nunca
daria um passo — nesses casos envie `nbs` igual ao `batch_size`. Para as execuções
científicas TACO-10 (batch 16), mantenha o padrão `nbs=64` (prática padrão YOLO).

Chaves opcionais com prefixo `aug_` (ex.: `aug_hsv_h`, `aug_mosaic`) são achatadas
para manter o config do Flower restrito a escalares; o cliente as converte nos
kwargs de augmentation do `model.train()` do Ultralytics. O bloco `augmentation`
de `configs/baseline_taco.yaml` é a fonte desses valores nas execuções científicas.

O cliente **lê esses valores de `config`** (`config.get("local_epochs", ...)`) — nunca
hardcoda hiperparâmetros de treino local. `run_config.py` já envia esses campos; um
servidor Flower distribuído deve usar o mesmo payload em `on_fit_config_fn`.

`mu=0` executa FedAvg. Quando `mu>0`, `fl/model.py` injeta o termo
`(mu/2) * ||w - w_global||²` no loss do Ultralytics antes do `backward`, usando os
pesos globais recebidos no início do round como referência. Trata-se do termo FedProx
no objetivo local, não de uma interpolação aproximada após o treino.

## Sincronização dos pesos treinados

O `model.train()` do Ultralytics treina uma **cópia interna** (`model.trainer.model`)
e, com `save=False`, não recarrega checkpoint nenhum no wrapper ao final. Por isso
`fl/model.py::train_local` copia explicitamente o `state_dict` treinado de volta
para `model.model` (`_sync_trained_weights`) antes de retornar. Sem isso, o cliente
devolveria os pesos globais inalterados e todas as execuções federadas seriam
no-ops silenciosos (bug corrigido em 15/09/2026 — execuções anteriores a essa data
em `results/baseline` não continham treino real).

Os pesos devolvidos são os **pesos brutos treinados**, não a média móvel EMA que o
Ultralytics usa em checkpoints/validação: com poucos passos locais por round, a EMA
ficaria dominada pela inicialização e mascararia a atualização do cliente.

## Métricas retornadas pelo cliente

`fit()` e `evaluate()` têm **formatos diferentes** — não confunda um com o outro.

`evaluate()` roda `model.val()` e retorna:

```python
{
    "loss": float,          # 1 - map50 (proxy; Ultralytics não expõe loss de detecção sem backward)
    "map50": float,
    "map50_95": float,
    "precision": float,
    "recall": float,
    "num_examples": int,    # imagens do split "val" usado na avaliação
}
```

`fit()` roda `train_local()` (treino local) e retorna:

```python
{
    "train_loss": float,               # soma dos componentes de loss do Ultralytics (box+cls+dfl)
    "fitness": float,                  # metrics.fitness do Ultralytics ao fim do treino local
    "mu": float,                       # eco do mu recebido em config, para auditoria
    "proximal_distance_sq_half": float,  # (1/2)||w - w_global||² ao final do treino — 0 quando mu=0
    "num_examples": int,               # adicionado pelo FLClient, não por train_local()
}
```

Não existe chave `"loss"` no retorno de `fit()` — quem espera isso (ex. um agregador de métricas de treino) precisa ler `"train_loss"`.

`num_examples` é o número de imagens usadas no treino/avaliação local daquele **cliente**
naquele round.

## Dispositivo e metadados de hardware

`device: "0"` representa GPU via APIs CUDA tanto em NVIDIA quanto em AMD ROCm.
O backend AMD é identificado por `torch.version.hip`, sem alterar o payload
Flower, pesos, loss, seeds ou hiperparâmetros. Compatibilidade numérica Windows
ROCm (BatchNorm nativo e limite inicial de escala AMP) fica em `fl/runtime.py`.
O runner pode gerar `hardware.json` informativo separado, com backend, GPU,
VRAM, versões HIP/CUDA/PyTorch e limite inicial AMP; esse arquivo é opcional.
Não modifica o schema dos quatro artefatos exigidos pelo validador nem impede
validar execuções antigas. Não é configuração científica adicional do servidor.

## Número de clientes

`fl/server.py::get_strategy` recebe o número de clientes da configuração e usa o
mesmo valor em `min_fit_clients` e `min_available_clients`.

## Estratégia de agregação (`strategy` no config)

Chave opcional do config lido por `run_config.py`/`get_strategy`, não repassada ao
cliente (não faz parte do payload de `on_fit_config_fn`): `"fedavg"` (padrão) ou
`"fedtrimmed"`. Ao contrário de `mu`, que muda o treino local sem tocar na
agregação, `strategy: fedtrimmed` troca a função de agregação usada no servidor —
`fl/server.py::aggregate_fedtrimmed` — por uma trimmed mean coordenada a
coordenada: descarta, por coordenada, os `k = int(num_clients * beta / 2)`
clientes com o maior e os `k` com o menor valor recebido, e faz a média
ponderada por `num_examples` dos restantes (com 5 clientes e `beta=0.4`, k=1).
`beta` é outra chave opcional do config, default `0.4`. `fedavg` e `fedtrimmed`
podem ser combinados com `mu > 0` (FedProx no cliente + FedTrimmed no servidor).

## Métricas de fit agregadas

O `FedAvg` nativo do Flower não agrega `fit_metrics` por padrão — só `evaluate_metrics`.
Se quisermos ver `loss`/`map50` do treino (não só da avaliação) agregados no servidor,
P3 precisa registrar um `fit_metrics_aggregation_fn` na `Strategy` (ver TODO em
`server.py::get_strategy`).


## Pendências para alinhar com P3

- [ ] Decidir se `fit_metrics_aggregation_fn` será necessário no servidor distribuído.
      O runner já registra as métricas locais diretamente em `clients.csv`.
- [ ] `evaluate()` sem backward pass não produz uma loss de detecção real (o
      Ultralytics só calcula box/cls/dfl loss durante o treino) — `fl/model.py` usa
      `1 - map50` como proxy. Decidir se o servidor
      precisa de uma loss "de verdade" aqui.
- [ ] Confirmar se o servidor espera receber `map50` em todo round de `evaluate()`, ou
      só nos rounds de avaliação centralizada.

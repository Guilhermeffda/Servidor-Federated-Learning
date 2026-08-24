# sprint2_preview/

Implementação **completa** do `FLClient` (pesos reais, treino local, avaliação,
simulação de dropout) e de uma simulação de frota multi-veículo, feita numa sessão
anterior a este planejamento formal do Trello.

Corresponde ao card **"[P4] Implementar simulação de 3-5 clientes virtuais"**
(Sprint 2), não ao Card 2 do Sprint 1 (que pede um esqueleto vazio). Foi movida para
cá — em vez de descartada — porque já está testada ponta a ponta (rodei
`simulate_fleet.py` com o dataset `coco8`: 3 veículos, 3 rounds, agregação FedAvg,
0 falhas, e a simulação de dropout levantando `ClientUnavailableError` corretamente).

**Não está ativa.** `fl/client.py` (Card 2) é o esqueleto oficial em uso agora. Quando
o Sprint 2 for autorizado, dá pra portar esta lógica para `fl/client.py` sem reescrever
do zero — só falta trocar o dataset de smoke-test (`coco8`) pelas partições reais do
pLitterStreet quando o P2 entregar.

| Arquivo aqui | Vira o quê em `fl/` quando o Sprint 2 for autorizado |
|---|---|
| `client_full.py` | lógica de `fl/client.py` (substitui o esqueleto) |
| `model.py` | `fl/model.py` |
| `dataset.py` | `fl/dataset.py` |
| `dropout.py` | `fl/dropout.py` |
| `simulate_fleet.py` | `simulate_fleet.py` (raiz) |

"""
Servidor Federated Learning (Flower) - Estrutura inicial.

Este módulo define a estrutura mínima do servidor federado, com a
Strategy do Flower como ponto de extensão para a lógica de agregação
(FedAvg será implementado em detalhe no próximo card).

Contrato de interface (ver fl/CONTRACT.md para detalhes completos):
    - Pesos trocados: lista de numpy.ndarray correspondendo ao
      state_dict() do modelo YOLO (na ordem das chaves do state_dict).
    - Métricas retornadas por cada cliente: dict contendo, no mínimo,
      {"loss": float, "num_examples": int}, e opcionalmente
      {"map50": float}.
"""

#  ALINHAR COM P4 
# 
# Os pontos abaixo estão marcados com "# ALINHAR COM P4" no corpo
# do código. Resumo do que precisa ser combinado ANTES de P4
# começar a implementar o FLClient:
#
#   1. Formato dos pesos (get_parameters / fit / evaluate do
#      cliente) — já proposto em fl/CONTRACT.md: lista de
#      numpy.ndarray na ordem do state_dict() do YOLO. P4 precisa
#      confirmar que a ordem das chaves do state_dict é estável
#      entre execuções/clientes.
#   2. Formato do dict de métricas retornado pelo cliente em
#      fit()/evaluate() — chaves obrigatórias "loss" e
#      "num_examples", opcional "map50". P4 precisa confirmar que
#      o cliente vai popular exatamente essas chaves.
#   3. Número real de clientes esperados no experimento
#      (min_fit_clients / min_available_clients abaixo estão como
#      placeholder = 3).
#   4. Se o cliente vai reportar métricas em fit() além de
#      evaluate() (hoje o FedAvg nativo não agrega fit_metrics por
#      padrão — ver TODO em get_strategy()).
# ============================================================

import flwr as fl


def get_strategy() -> fl.server.strategy.Strategy:
    """
    Retorna a Strategy usada pelo servidor federado.

    Placeholder: por enquanto usa o FedAvg nativo do Flower, sem
    customização. Será substituído/estendido pela lógica real de
    FedAvg (ou agregação customizada, se necessário) no próximo card.
    """
    # ALINHAR COM P4 (item 3): min_fit_clients / min_available_clients
    # estão como placeholder = 3. Precisa bater com o número real de
    # clientes que P4 vai simular/rodar no experimento — se for
    # diferente de 3, atualizar aqui.
    # TODO: revisar hiperparâmetros (fraction_fit, min_fit_clients,
    #       min_available_clients) de acordo com o número real de
    #       clientes disponíveis no experimento.
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        # TODO: implementar lógica de agregação customizada aqui,
        #       caso o FedAvg nativo não seja suficiente (ex.:
        #       agregação ponderada por métrica de qualidade, e não
        #       apenas por num_examples).
        #
        # ALINHAR COM P4 (itens 1 e 2): as duas linhas de
        # aggregation_fn abaixo (comentadas) dependem do formato
        # exato do dict de métricas que o FLClient de P4 vai
        # retornar em fit()/evaluate(). Confirmar com P4 as chaves
        # ("loss", "num_examples", "map50") antes de implementar as
        # funções de agregação.
        # TODO: definir evaluate_metrics_aggregation_fn e
        #       fit_metrics_aggregation_fn para agregar loss/map50
        #       reportados pelos clientes (contrato em CONTRACT.md).
        # evaluate_metrics_aggregation_fn=...,  # ALINHAR COM P4
        # fit_metrics_aggregation_fn=...,       # ALINHAR COM P4
    )


def start_server(num_rounds: int = 1) -> None:
    """
    Inicializa e sobe o servidor Flower.

    Args:
        num_rounds: número de rounds de treinamento federado.
    """
    strategy = get_strategy()

    # TODO: parametrizar server_address (host/porta) via variável de
    #       ambiente ou argumento de linha de comando, em vez de
    #       hardcoded.
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()

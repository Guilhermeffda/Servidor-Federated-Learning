"""Configuracao e agregacao FedAvg compartilhadas pelo servidor e pelo runner."""

from __future__ import annotations

import numpy as np
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


def get_strategy(num_clients: int) -> FedAvg:
    """Retorna FedAvg puro, exigindo todos os clientes da baseline."""
    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        accept_failures=False,
    )


def aggregate_fedavg(
    updates: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    """Media ponderada pelo numero de exemplos, igual ao FedAvg do Flower."""
    if not updates:
        raise ValueError("FedAvg requer ao menos uma atualizacao de cliente")
    if any(num_examples <= 0 for _, num_examples in updates):
        raise ValueError("num_examples deve ser positivo")
    return aggregate(updates)

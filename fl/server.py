"""Configuracao e agregacao FedAvg compartilhadas pelo servidor e pelo runner."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate


def make_fit_config_fn(config: dict[str, Any]) -> Callable[[int], dict[str, Any]]:
    """Retorna o on_fit_config_fn do contrato (fl/CONTRACT.md) por round."""
    fit_config: dict[str, Any] = {
        "local_epochs": int(config.get("local_epochs", 5)),
        "batch_size": int(config.get("batch_size", 16)),
        "image_size": int(config.get("image_size", 640)),
        "device": str(config.get("device", "cpu")),
        "mu": float(config.get("mu", 0.0)),
        "nbs": int(config.get("nbs", 64)),
    }
    for key, value in (config.get("augmentation") or {}).items():
        fit_config[f"aug_{key}"] = float(value)

    def on_fit_config(server_round: int) -> dict[str, Any]:
        return {**fit_config, "server_round": server_round}

    return on_fit_config


def weighted_evaluate_metrics(
    results: list[tuple[int, dict[str, Any]]],
) -> dict[str, float]:
    """Media das metricas de avaliacao ponderada por num_examples.

    O FedAvg nativo do Flower nao agrega metricas de avaliacao por padrao; esta
    funcao cobre o requisito de `evaluate_metrics_aggregation_fn` do contrato.
    """
    if not results:
        return {}
    total = sum(num_examples for num_examples, _ in results)
    if total <= 0:
        raise ValueError("num_examples agregado deve ser positivo")
    aggregated: dict[str, float] = {}
    numeric_keys = {
        key
        for _, metrics in results
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    for key in sorted(numeric_keys):
        aggregated[key] = (
            sum(
                num_examples * float(metrics[key])
                for num_examples, metrics in results
                if key in metrics
            )
            / total
        )
    return aggregated


def get_strategy(
    num_clients: int, fit_config: dict[str, Any] | None = None
) -> FedAvg:
    """Retorna FedAvg puro, exigindo todos os clientes da baseline."""
    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        accept_failures=False,
        on_fit_config_fn=make_fit_config_fn(fit_config or {}),
        evaluate_metrics_aggregation_fn=weighted_evaluate_metrics,
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

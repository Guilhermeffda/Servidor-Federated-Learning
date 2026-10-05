"""Configuracao e agregacao FedAvg/FedTrimmed compartilhadas pelo servidor e pelo runner."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
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
    """Retorna a Strategy do contrato: FedAvg (padrao/FedProx) ou FedTrimmed.

    A escolha e feita pela chave opcional `strategy` do config ("fedavg" por
    padrao, "fedtrimmed" para trimmed mean). FedProx nao muda a agregacao — o
    termo proximal age so no cliente (ver `fl/model.py`) — por isso configs com
    `mu > 0` continuam usando FedAvg aqui.
    """
    config = fit_config or {}
    common_kwargs: dict[str, Any] = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        accept_failures=False,
        on_fit_config_fn=make_fit_config_fn(config),
        evaluate_metrics_aggregation_fn=weighted_evaluate_metrics,
    )
    if str(config.get("strategy", "fedavg")).lower() == "fedtrimmed":
        return FedTrimmed(beta=float(config.get("beta", 0.4)), **common_kwargs)
    return FedAvg(**common_kwargs)


def aggregate_fedavg(
    updates: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    """Media ponderada pelo numero de exemplos, igual ao FedAvg do Flower."""
    if not updates:
        raise ValueError("FedAvg requer ao menos uma atualizacao de cliente")
    if any(num_examples <= 0 for _, num_examples in updates):
        raise ValueError("num_examples deve ser positivo")
    return aggregate(updates)


def aggregate_fedtrimmed(
    updates: list[tuple[list[np.ndarray], int]],
    beta: float = 0.4,
) -> list[np.ndarray]:
    """Trimmed mean coordenada a coordenada, ponderada por num_examples.

    Para cada coordenada de cada camada, descarta independentemente os
    `k = int(n * beta / 2)` clientes com o maior valor e os `k` com o menor
    valor recebidos naquela coordenada, e calcula a media dos `n - 2k`
    restantes ponderada pelo `num_examples` de cada um deles (nao uma media
    simples) — com n=5 clientes e beta=0.4, k=1.

    `k=0` nao e tratado como caso-limite aceitavel: um `beta` pequeno demais
    para o numero de clientes faria esta funcao degenerar silenciosamente em
    FedAvg, escondendo um erro de configuracao.
    """
    if not updates:
        raise ValueError("FedTrimmed requer ao menos uma atualizacao de cliente")
    if any(num_examples <= 0 for _, num_examples in updates):
        raise ValueError("num_examples deve ser positivo")

    weights_list = [weights for weights, _ in updates]
    num_examples = np.array([count for _, count in updates], dtype=np.float64)
    n = len(weights_list)
    k = int(n * beta / 2)
    if k <= 0:
        raise ValueError(
            f"beta={beta} com n={n} clientes produz k={k} (nenhum corte); ajuste "
            "beta — k=0 indica erro de configuracao, nao uma limitacao aceitavel"
        )
    if n - 2 * k <= 0:
        raise ValueError(f"beta={beta} com n={n} clientes corta todos os clientes (k={k})")

    trimmed_layers: list[np.ndarray] = []
    for layer_index in range(len(weights_list[0])):
        layer_dtype = weights_list[0][layer_index].dtype
        stacked = np.stack([weights[layer_index] for weights in weights_list], axis=0)

        # Indices dos clientes por coordenada, ordenados do menor para o maior
        # valor recebido naquela coordenada.
        order = np.argsort(stacked, axis=0)
        keep_order = order[k : n - k]

        kept_values = np.take_along_axis(stacked, keep_order, axis=0)
        weights_broadcast = np.broadcast_to(
            num_examples.reshape((n,) + (1,) * (stacked.ndim - 1)), stacked.shape
        )
        kept_weights = np.take_along_axis(weights_broadcast, keep_order, axis=0)

        weighted_sum = (kept_values * kept_weights).sum(axis=0)
        weight_total = kept_weights.sum(axis=0)
        trimmed_layers.append((weighted_sum / weight_total).astype(layer_dtype))

    return trimmed_layers


class FedTrimmed(FedAvg):
    """FedAvg com `aggregate_fit` substituido por trimmed mean coordenada a
    coordenada (ver `aggregate_fedtrimmed`)."""

    def __init__(self, beta: float = 0.4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.beta = beta

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Any]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        updates = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = aggregate_fedtrimmed(updates, beta=self.beta)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated: dict[str, Any] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

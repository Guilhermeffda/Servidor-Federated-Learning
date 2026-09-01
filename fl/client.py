"""
fl/client.py
Cliente Federated Learning (Flower) — esqueleto da interface NumPyClient.

Define a estrutura mínima que o FLServer (P3) espera poder instanciar e chamar a
cada round de treino federado (contrato em fl/CONTRACT.md). A lógica real de treino
local do YOLOv8n fica para o próximo card (Sprint 2, "Implementar simulação de 3-5
clientes virtuais") — os `TODO` abaixo são intencionais, não esquecimento.
"""

from typing import Any

import flwr as fl
from ultralytics import YOLO


class FLClient(fl.client.NumPyClient):
    """Representa um veículo/cliente federado, dono do seu próprio YOLOv8n local."""

    def __init__(self, client_id: str, data_yaml_path: str) -> None:
        """
        Args:
            client_id: identificador do cliente (ex.: "client_0"), usado em logs.
            data_yaml_path: caminho do data.yaml (formato Ultralytics) com a
                partição de dados deste cliente.
        """
        self.client_id = client_id
        self.data_yaml_path = data_yaml_path
        self.model = YOLO("yolov8n.pt")

    def get_parameters(self, config: dict[str, Any]) -> list:
        """
        Retorna os pesos atuais do modelo local, no formato do contrato
        (fl/CONTRACT.md): lista de numpy.ndarray na ordem do state_dict().
        """
        # TODO (próximo card): extrair pesos de model.model.state_dict()
        return []

    def fit(self, parameters: list, config: dict[str, Any]) -> tuple[list, int, dict]:
        """
        Treina o modelo local a partir dos pesos globais recebidos.

        Returns:
            Tupla (pesos atualizados, número de exemplos usados, métricas).
        """
        # TODO (próximo card): setar pesos, treinar localmente, retornar pesos
        #       atualizados. `config` traz {"local_epochs": int, "batch_size": int}
        #       enviados pelo servidor (fl/CONTRACT.md) -- nunca hardcodar esses
        #       valores.
        return [], 0, {}

    def evaluate(self, parameters: list, config: dict[str, Any]) -> tuple[float, int, dict]:
        """
        Avalia o modelo local com os pesos recebidos.

        Returns:
            Tupla (loss, número de exemplos avaliados, métricas).
        """
        # TODO (próximo card): avaliar localmente e retornar métricas (loss, map50)
        return 0.0, 0, {}

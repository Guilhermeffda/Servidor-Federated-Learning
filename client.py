"""
Cliente Federated Learning (Flower) - FLClient.

Representa um veículo: treina um YOLOv8n localmente sobre a partição de dados do
veículo e troca pesos com o servidor (fl/CONTRACT.md) a cada round. Simula
conectividade intermitente via fl/dropout.py -- um veículo "fora de conexão" no round
simplesmente não retorna uma atualização válida para o servidor.
"""

import flwr as fl

from fl.dataset import resolve_smoke_test_dataset, resolve_vehicle_dataset
from fl.dropout import ClientUnavailableError, ConnectivitySimulator
from fl.model import create_model, evaluate, get_weights, set_weights, train_one_round


class FLClient(fl.client.NumPyClient):
    """Cliente Flower que representa um único veículo simulado."""

    def __init__(self, vehicle_id: str, data_yaml: str, availability_probability: float = 1.0):
        self.vehicle_id = vehicle_id
        self.data_yaml = data_yaml
        self.model = create_model()
        self.connectivity = ConnectivitySimulator(availability_probability)

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        if not self.connectivity.is_available():
            raise ClientUnavailableError(
                f"Veículo '{self.vehicle_id}' está fora de conexão neste round."
            )
        set_weights(self.model, parameters)
        metrics = train_one_round(self.model, self.data_yaml)
        return get_weights(self.model), metrics["num_examples"], metrics

    def evaluate(self, parameters, config):
        if not self.connectivity.is_available():
            raise ClientUnavailableError(
                f"Veículo '{self.vehicle_id}' está fora de conexão neste round."
            )
        set_weights(self.model, parameters)
        metrics = evaluate(self.model, self.data_yaml)
        return metrics["loss"], metrics["num_examples"], metrics


def make_client(vehicle_id: str, availability_probability: float = 1.0) -> FLClient:
    """
    Cria um FLClient para o veículo `vehicle_id`.

    Usa a partição real de dados do veículo (fl/dataset.py) se ela já existir;
    caso contrário cai para o dataset de smoke-test (coco8), já que as partições do
    pLitterStreet ainda dependem da entrega do P2.
    """
    try:
        data_yaml = resolve_vehicle_dataset(vehicle_id)
    except FileNotFoundError:
        data_yaml = resolve_smoke_test_dataset()
    return FLClient(vehicle_id, data_yaml, availability_probability)


if __name__ == "__main__":
    # TODO: parametrizar vehicle_id, server_address e availability_probability via
    #       linha de comando/variável de ambiente, em vez de hardcoded.
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=make_client(vehicle_id="vehicle-0").to_client(),
    )

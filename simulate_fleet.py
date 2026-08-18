"""
simulate_fleet.py
Sobe uma frota de veículos simulados (clientes Flower) e o servidor (server.py) num
único processo, usando fl.simulation. É como o experimento de conectividade
intermitente do P4 é rodado na prática: sem hardware real, tudo em software.

Uso:
    python simulate_fleet.py
"""

import flwr as fl
from flwr.simulation import run_simulation

from client import make_client
from server import get_strategy

NUM_VEHICLES = 3
NUM_ROUNDS = 3

# Probabilidade de cada veículo estar "online" (disponível) em um round qualquer.
# 1.0 = sempre disponível. Baixar esse valor é como simulamos conectividade
# intermitente/dropout de veículos -- o cerne do experimento do P4.
VEHICLE_AVAILABILITY_PROBABILITY = 0.8


def client_fn(context: fl.common.Context) -> fl.client.Client:
    vehicle_id = f"vehicle-{context.node_config['partition-id']}"
    return make_client(vehicle_id, VEHICLE_AVAILABILITY_PROBABILITY).to_client()


def main() -> None:
    run_simulation(
        client_app=fl.client.ClientApp(client_fn=client_fn),
        server_app=fl.server.ServerApp(
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=get_strategy(),
        ),
        num_supernodes=NUM_VEHICLES,
    )


if __name__ == "__main__":
    main()

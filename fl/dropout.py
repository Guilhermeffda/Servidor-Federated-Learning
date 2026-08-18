"""
fl/dropout.py
Simula a conectividade intermitente de um veículo: cada cliente tem uma
probabilidade independente de estar "online" (alcançável pelo servidor) em cada
round de treinamento federado. É o mecanismo central do experimento do P4 (ver
README.md).
"""

import random


class ClientUnavailableError(RuntimeError):
    """Levantada pelo FLClient quando o veículo simula estar fora de conexão no round."""


class ConnectivitySimulator:
    """Decide, a cada round, se um veículo está disponível para participar."""

    def __init__(self, availability_probability: float = 1.0, seed: int | None = None):
        if not 0.0 <= availability_probability <= 1.0:
            raise ValueError("availability_probability deve estar entre 0.0 e 1.0")
        self.availability_probability = availability_probability
        self._rng = random.Random(seed)

    def is_available(self) -> bool:
        """Sorteia se o veículo está online neste round."""
        return self._rng.random() < self.availability_probability

"""Cliente Flower usado pelas execucoes federadas do projeto.

O dropout de conectividade do escopo anterior (frota de veiculos) foi descartado
junto com esse escopo; nao faz parte deste projeto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import flwr as fl
import yaml

from fl.model import evaluate_model, get_weights, load_model, set_weights, train_local


class FLClient(fl.client.NumPyClient):
    """Um cliente YOLO com dados locais e configuracao recebida por round."""

    def __init__(
        self,
        client_id: str,
        data_yaml_path: str,
        num_train_examples: int,
        model_weights: str = "yolov8n.pt",
    ) -> None:
        self.client_id = client_id
        self.data_yaml_path = data_yaml_path
        self.num_train_examples = num_train_examples
        dataset = yaml.safe_load(Path(data_yaml_path).read_text(encoding="utf-8"))
        self.model = load_model(model_weights, num_classes=len(dataset["names"]))

    def get_parameters(self, config: dict[str, Any]) -> list:
        return get_weights(self.model)

    def fit(self, parameters: list, config: dict[str, Any]) -> tuple[list, int, dict]:
        set_weights(self.model, parameters)
        metrics = train_local(
            model=self.model,
            data_yaml=self.data_yaml_path,
            epochs=int(config.get("local_epochs", 1)),
            batch_size=int(config.get("batch_size", 8)),
            image_size=int(config.get("image_size", 640)),
            device=str(config.get("device", "cpu")),
            seed=int(config.get("seed", 0)),
            output_dir=Path(str(config.get("output_dir", "runs/federated"))),
            run_name=str(config.get("run_name", self.client_id)),
            mu=float(config.get("mu", 0.0)),
            nbs=int(config.get("nbs", 64)),
            # Chaves "aug_<nome>" chegam achatadas para manter o config do Flower
            # restrito a escalares; viram kwargs de augmentation do Ultralytics.
            augmentation={
                key[4:]: float(value)
                for key, value in config.items()
                if key.startswith("aug_")
            },
        )
        metrics["num_examples"] = self.num_train_examples
        return get_weights(self.model), self.num_train_examples, metrics

    def evaluate(
        self, parameters: list, config: dict[str, Any]
    ) -> tuple[float, int, dict]:
        set_weights(self.model, parameters)
        metrics = evaluate_model(
            self.model,
            self.data_yaml_path,
            image_size=int(config.get("image_size", 640)),
            batch_size=int(config.get("batch_size", 8)),
            device=str(config.get("device", "cpu")),
        )
        return metrics["loss"], int(metrics["num_examples"]), metrics

"""
fl/model.py
Ponte entre o YOLOv8n (Ultralytics) e o protocolo do Flower: conversão de pesos para
o formato trocado com o servidor (ver fl/CONTRACT.md) e treino/avaliação local usados
pelo FLClient em cada round.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def create_model(weights: str = "yolov8n.pt") -> YOLO:
    """Cria um modelo YOLOv8n a partir de pesos pré-treinados (padrão) ou de um checkpoint."""
    return YOLO(weights)


def get_weights(model: YOLO) -> List[np.ndarray]:
    """Extrai os pesos do modelo como lista de numpy.ndarray, na ordem do state_dict()."""
    return [tensor.cpu().numpy() for tensor in model.model.state_dict().values()]


def set_weights(model: YOLO, weights: List[np.ndarray]) -> None:
    """Carrega no modelo uma lista de numpy.ndarray no formato produzido por get_weights."""
    state_dict = model.model.state_dict()
    new_state_dict = {
        key: torch.as_tensor(array, dtype=tensor.dtype)
        for (key, tensor), array in zip(state_dict.items(), weights)
    }
    model.model.load_state_dict(new_state_dict, strict=True)


def train_one_round(model: YOLO, data_yaml: str, epochs: int = 1, imgsz: int = 640) -> dict:
    """
    Treina o modelo localmente por `epochs` épocas sobre `data_yaml`.

    Retorna as métricas no formato do contrato (fl/CONTRACT.md):
    {"loss": float, "num_examples": int, "map50": float}.
    """
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        verbose=False,
        plots=False,
        val=True,
    )
    trainer = model.trainer
    return {
        # trainer.tloss = {"box_loss": ..., "cls_loss": ..., "dfl_loss": ...} da
        # última época; a loss total de detecção é a soma dos três componentes.
        "loss": _sum_tensor_dict(trainer.tloss),
        "num_examples": len(trainer.train_loader.dataset),
        "map50": _extract_map50(trainer.metrics),
    }


def evaluate(model: YOLO, data_yaml: str, imgsz: int = 640) -> dict:
    """
    Avalia o modelo (sem treinar) sobre `data_yaml`.

    Retorna as métricas no formato do contrato (fl/CONTRACT.md):
    {"loss": float, "num_examples": int, "map50": float}.
    """
    results = model.val(data=data_yaml, imgsz=imgsz, verbose=False, plots=False)
    map50 = float(results.box.map50)
    return {
        # O Ultralytics só calcula a loss de treino (box/cls/dfl) durante o
        # backward pass do treino -- uma validação isolada (sem gradiente) não a
        # produz. Usamos "1 - mAP50" como proxy de "quão ruim" o modelo está neste
        # round. ALINHAR COM P3: decidir se o servidor precisa de uma loss de
        # detecção real aqui (exigiria reimplementar o cálculo de loss fora do
        # ciclo de treino) ou se esse proxy é suficiente.
        "loss": 1.0 - map50,
        "num_examples": _count_dataset_images(data_yaml, split="val"),
        "map50": map50,
    }


def _sum_tensor_dict(tensor_dict) -> float:
    if not tensor_dict:
        return float("nan")
    return float(sum(tensor_dict.values()))


def _extract_map50(metrics: dict) -> float:
    for key in metrics:
        if key.endswith("mAP50(B)"):
            return float(metrics[key])
    return float("nan")


def _count_dataset_images(data_yaml: str, split: str) -> int:
    """Conta as imagens do split (`train`/`val`/`test`) resolvido a partir do data.yaml."""
    dataset_info = check_det_dataset(data_yaml)
    split_path = dataset_info[split]
    split_dir = Path(split_path[0] if isinstance(split_path, list) else split_path)
    return sum(1 for path in split_dir.iterdir() if path.suffix.lower() in _IMAGE_EXTENSIONS)

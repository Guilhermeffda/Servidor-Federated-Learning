"""Adaptacao entre pesos NumPy do Flower e modelos YOLO Ultralytics."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import unwrap_model


def load_model(weights: str = "yolov8n.pt", num_classes: int | None = None) -> YOLO:
    """Carrega pesos e, se preciso, recria a cabeca para o numero de classes do dataset."""
    model = YOLO(weights)
    current_classes = int(model.model.model[-1].nc)
    if num_classes is not None and num_classes != current_classes:
        architecture = DetectionModel(
            cfg=model.model.yaml, ch=3, nc=num_classes, verbose=False
        )
        architecture.load(model.model)
        model.model = architecture
    return model


def get_weights(model: YOLO) -> list[np.ndarray]:
    return [value.detach().cpu().numpy().copy() for value in model.model.state_dict().values()]


def set_weights(model: YOLO, weights: list[np.ndarray]) -> None:
    state = model.model.state_dict()
    if len(weights) != len(state):
        raise ValueError(f"Quantidade de tensores invalida: {len(weights)} != {len(state)}")

    restored = {}
    for (name, current), received in zip(state.items(), weights):
        if tuple(received.shape) != tuple(current.shape):
            raise ValueError(
                f"Shape invalido em {name}: {tuple(received.shape)} != {tuple(current.shape)}"
            )
        restored[name] = torch.as_tensor(received, dtype=current.dtype, device=current.device)
    model.model.load_state_dict(restored, strict=True)


def proximal_penalty(
    model: torch.nn.Module, reference: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Retorna 1/2 * ||w - w_global||^2 para os parametros treinaveis."""
    terms = [
        (parameter - reference[name].to(parameter.device)).pow(2).sum()
        for name, parameter in model.named_parameters()
    ]
    if not terms:
        return torch.zeros((), device=next(model.buffers()).device)
    return 0.5 * torch.stack(terms).sum()


def _install_fedprox_loss(trainer, reference: dict[str, torch.Tensor], mu: float) -> None:
    """Adiciona o termo proximal ao loss antes do backward do Ultralytics."""
    train_model = unwrap_model(trainer.model)
    current_names = {name for name, _ in train_model.named_parameters()}
    if current_names != reference.keys():
        raise ValueError("Parametros do modelo local divergem da referencia global")
    original_loss = train_model.loss

    def loss_with_proximal_term(batch, preds=None):
        detection_loss, loss_items = original_loss(batch, preds)
        penalty = proximal_penalty(train_model, reference).to(detection_loss.dtype)
        return detection_loss + mu * penalty, loss_items

    train_model.loss = loss_with_proximal_term


def train_local(
    model: YOLO,
    data_yaml: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    device: str,
    seed: int,
    output_dir: Path,
    run_name: str,
    mu: float = 0.0,
) -> dict[str, float]:
    if mu < 0:
        raise ValueError("mu deve ser maior ou igual a zero")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    reference = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.model.named_parameters()
    }
    callback = None
    if mu > 0:
        callback = lambda trainer: _install_fedprox_loss(trainer, reference, mu)
        model.add_callback("on_pretrain_routine_end", callback)
    try:
        result = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=device,
            seed=seed,
            deterministic=True,
            workers=0,
            project=str(output_dir),
            name=run_name,
            exist_ok=True,
            save=False,
            plots=False,
            val=False,
            verbose=False,
        )
    finally:
        if callback is not None:
            model.callbacks["on_pretrain_routine_end"].remove(callback)
    loss_items = getattr(model.trainer, "tloss", None)
    if loss_items is None:
        train_loss = float("nan")
    elif hasattr(loss_items, "sum"):
        train_loss = float(loss_items.sum().detach().cpu())
    else:
        train_loss = float(sum(loss_items.values()))
    final_penalty = float(proximal_penalty(model.model, reference).detach().cpu())
    return {
        "train_loss": train_loss,
        "fitness": float(getattr(result, "fitness", 0.0) or 0.0),
        "mu": mu,
        "proximal_distance_sq_half": final_penalty,
    }


def evaluate_model(
    model: YOLO,
    data_yaml: str,
    image_size: int,
    batch_size: int,
    device: str,
) -> dict[str, float | int]:
    from fl.partition import count_split_images

    result = model.val(
        data=data_yaml,
        split="val",
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
    )
    map50 = float(result.box.map50)
    return {
        "loss": 1.0 - map50,
        "map50": map50,
        "map50_95": float(result.box.map),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "num_examples": count_split_images(Path(data_yaml), "val"),
    }

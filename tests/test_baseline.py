from pathlib import Path

import numpy as np
import torch

from fl.model import get_weights, load_model, proximal_penalty, set_weights
from fl.partition import partition_images
from fl.server import (
    aggregate_fedavg,
    get_strategy,
    make_fit_config_fn,
    weighted_evaluate_metrics,
)


def test_fedavg_is_weighted_by_num_examples() -> None:
    updates = [([np.array([1.0, 3.0])], 1), ([np.array([5.0, 7.0])], 3)]
    result = aggregate_fedavg(updates)
    np.testing.assert_allclose(result[0], [4.0, 6.0])


def test_fit_config_fn_follows_contract() -> None:
    on_fit_config = make_fit_config_fn(
        {"local_epochs": 5, "batch_size": 16, "mu": 0.01, "augmentation": {"mosaic": 1.0}}
    )
    config = on_fit_config(7)
    assert config["local_epochs"] == 5
    assert config["batch_size"] == 16
    assert config["mu"] == 0.01
    assert config["nbs"] == 64
    assert config["aug_mosaic"] == 1.0
    assert config["server_round"] == 7


def test_weighted_evaluate_metrics_aggregates_by_num_examples() -> None:
    results = [(1, {"map50": 0.2, "loss": 0.8}), (3, {"map50": 0.6, "loss": 0.4})]
    aggregated = weighted_evaluate_metrics(results)
    assert np.isclose(aggregated["map50"], 0.5)
    assert np.isclose(aggregated["loss"], 0.5)
    assert weighted_evaluate_metrics([]) == {}


def test_get_strategy_requires_all_clients_and_aggregates_metrics() -> None:
    strategy = get_strategy(5, {"local_epochs": 5, "batch_size": 16})
    assert strategy.min_fit_clients == 5
    assert strategy.min_available_clients == 5
    assert strategy.on_fit_config_fn(1)["local_epochs"] == 5
    assert strategy.evaluate_metrics_aggregation_fn is weighted_evaluate_metrics


def test_partition_is_complete_disjoint_and_reproducible() -> None:
    images = [Path(f"image_{index}.jpg") for index in range(11)]
    first = partition_images(images, "iid", 3, 42)
    second = partition_images(images, "iid", 3, 42)
    assert first == second
    flattened = [image for partition in first for image in partition]
    assert sorted(flattened) == sorted(images)
    assert len(flattened) == len(set(flattened))
    assert [len(partition) for partition in first] == [4, 4, 3]


def test_proximal_penalty_is_zero_at_global_model_and_positive_after_change() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    reference = {name: value.detach().clone() for name, value in model.named_parameters()}
    assert proximal_penalty(model, reference).item() == 0.0
    with torch.no_grad():
        model.weight.add_(1.0)
    assert np.isclose(proximal_penalty(model, reference).item(), 1.0)


def test_train_local_changes_trainable_parameters(tmp_path: Path) -> None:
    """Regressao: model.train() do Ultralytics treina uma copia interna e, com
    save=False, os pesos treinados nao voltavam para model.model — o cliente
    devolvia os pesos globais inalterados e o federado inteiro era um no-op."""
    from fl.model import train_local

    model = load_model("yolov8n.pt")
    parameter_names = {name for name, _ in model.model.named_parameters()}
    before = {
        name: value.detach().clone() for name, value in model.model.state_dict().items()
    }
    train_local(
        model=model,
        data_yaml="data/mini_test/data.yaml",
        epochs=1,
        batch_size=2,
        image_size=128,
        device="cpu",
        seed=0,
        output_dir=tmp_path,
        run_name="regression_probe",
        nbs=2,
    )
    after = model.model.state_dict()
    changed_parameters = [
        name
        for name in parameter_names
        if not torch.equal(before[name], after[name].detach().cpu())
    ]
    assert changed_parameters, "treino local nao alterou nenhum parametro treinavel"


def test_yolo_weight_roundtrip_is_exact() -> None:
    model = load_model("yolov8n.pt")
    before = get_weights(model)
    set_weights(model, before)
    after = get_weights(model)
    assert len(before) == len(after)
    assert all(np.array_equal(left, right) for left, right in zip(before, after))

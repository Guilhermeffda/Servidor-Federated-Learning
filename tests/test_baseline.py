from pathlib import Path

import numpy as np
import torch

from fl.model import get_weights, load_model, proximal_penalty, set_weights
from fl.partition import partition_images
from fl.server import aggregate_fedavg


def test_fedavg_is_weighted_by_num_examples() -> None:
    updates = [([np.array([1.0, 3.0])], 1), ([np.array([5.0, 7.0])], 3)]
    result = aggregate_fedavg(updates)
    np.testing.assert_allclose(result[0], [4.0, 6.0])


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


def test_yolo_weight_roundtrip_is_exact() -> None:
    model = load_model("yolov8n.pt")
    before = get_weights(model)
    set_weights(model, before)
    after = get_weights(model)
    assert len(before) == len(after)
    assert all(np.array_equal(left, right) for left, right in zip(before, after))

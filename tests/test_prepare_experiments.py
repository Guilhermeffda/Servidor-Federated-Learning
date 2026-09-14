from taco_data.scripts.prepare_experiments import iid_partition, non_iid_partition


def _assert_valid(parts: list[list[int]], expected: list[int]) -> None:
    flattened = [image_id for part in parts for image_id in part]
    assert sorted(flattened) == sorted(expected)
    assert len(flattened) == len(set(flattened))
    assert all(parts)


def test_iid_and_non_iid_partitions_are_complete_and_reproducible() -> None:
    image_ids = list(range(100))
    labels = {image_id: image_id % 10 for image_id in image_ids}
    iid = iid_partition(image_ids, labels, 5, 42)
    non_iid = non_iid_partition(image_ids, labels, 5, 42, 0.5)
    _assert_valid(iid, image_ids)
    _assert_valid(non_iid, image_ids)
    assert iid == iid_partition(image_ids, labels, 5, 42)
    assert non_iid == non_iid_partition(image_ids, labels, 5, 42, 0.5)

import pytest

@pytest.mark.slow
def test_cifar10_partitioner_does_not_crash():
    from qpriviot_fl.task import load_data
    # Use small batch size and check if it runs without TypeError
    train, val = load_data(partition_id=0, num_partitions=10, 
                            batch_size=8, dataset_name="cifar10")
    batch = next(iter(train))
    assert batch["img"].shape[0] <= 8
    assert "label" in batch

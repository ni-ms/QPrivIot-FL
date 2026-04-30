import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

def test_dirichlet_creates_imbalance():
    p = DirichletPartitioner(num_partitions=10, partition_by="label", alpha=0.3, seed=1)
    fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": p})
    sizes = [len(fds.load_partition(i, "train")) for i in range(10)]
    cv = np.std(sizes) / np.mean(sizes)
    assert cv > 0.10, f"Dirichlet partitions are too uniform (cv={cv:.2f})"

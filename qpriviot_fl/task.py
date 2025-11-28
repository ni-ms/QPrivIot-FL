import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np
from datasets import load_dataset


class Net(nn.Module):
    """CIFAR-10 CNN Model"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FemnistNet(nn.Module):
    """FEMNIST CNN Model"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class IotModel(nn.Module):
    """Simple IoT Sensor Model"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def make_model(dataset_name: str):
    """Creates the appropriate PyTorch model based on the dataset name."""
    if dataset_name == "iot":
        return IotModel()
    if dataset_name == "femnist":
        return FemnistNet()
    return Net()


def load_data(partition_id: int, num_partitions: int, batch_size: int, dataset_name: str):
    """Load federated training data (partitioned) and global evaluation data (full test set)."""

    if dataset_name == "iot":
        np.random.seed(partition_id)
        data = torch.randn(500, 10)
        targets = torch.randint(0, 2, (500,))
        train_ds = TensorDataset(data, targets)
        val_ds = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
            DataLoader(val_ds, batch_size=batch_size)

    partitioner = IidPartitioner(num_partitions=num_partitions)

    if dataset_name == "femnist":
        hub_dataset_name = "flwrlabs/femnist"
    else:
        hub_dataset_name = "uoft-cs/cifar10"

    fds = FederatedDataset(dataset=hub_dataset_name, partitioners={"train": partitioner})

    val_set = load_dataset(hub_dataset_name, split="test")

    if dataset_name == "femnist":
        transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    else:
        transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        batch["img"] = [transforms(i) for i in batch["img"]]
        return batch

    train_partition = fds.load_partition(partition_id)
    train_partition = train_partition.with_transform(apply_transforms)
    train_partition.set_format('torch')  # <--- PREVIOUS FIX: Set PyTorch format
    train_loader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)

    val_set = val_set.with_transform(apply_transforms)
    val_set.set_format('torch')  # <--- PREVIOUS FIX: Set PyTorch format
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader


def train(model, loader, epochs, lr, device):
    """Train the model for a specified number of epochs."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for batch in loader:
            if isinstance(batch, dict):
                x, y = batch["img"], batch["label"]
            else:
                x, y = batch

            # CRITICAL FIX: Cast input image tensor 'x' to float32
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / (num_batches if num_batches > 0 else 1)


def test(model, loader, device):
    """Test the model and return loss and accuracy."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                x, y = batch["img"], batch["label"]
            else:
                x, y = batch

            # CRITICAL FIX: Cast input image tensor 'x' to float32
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)

            out = model(x)

            loss = criterion(out, y).item()
            total_loss += loss

            pred = out.argmax(1)
            batch_correct = (pred == y).sum().item()

            correct += batch_correct

    num_samples = len(loader.dataset)
    num_batches = len(loader)

    if num_samples == 0 or num_batches == 0:
        return 0.0, 0.0

    avg_loss = total_loss / num_batches
    accuracy = correct / num_samples

    return avg_loss, accuracy

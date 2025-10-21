import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from opacus import PrivacyEngine
import numpy as np

# PyTorch transforms for CIFAR-10
pytorch_transforms = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class Net(nn.Module):
    """Simple CNN for CIFAR-10."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR-10 data."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def signal_sensitivity_estimation(data_loader):
    """
    Estimate signal sensitivity from data characteristics.
    In a real implementation, analyze feature variance, gradient norms, etc.
    """
    # Placeholder: return random sensitivity factor
    return np.random.uniform(0.7, 1.3)


def train_with_dp(
        net,
        trainloader,
        epochs,
        lr,
        device,
        noise_multiplier,
        max_grad_norm,
        target_delta,
):
    """Train model with differential privacy using Opacus."""

    net = net.to(device)

    # Make model compatible with Opacus
    net = ModuleValidator.fix(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Attach privacy engine
    privacy_engine = PrivacyEngine()
    net, optimizer, trainloader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    net.train()
    running_loss = 0.0

    for epoch in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    avg_loss = running_loss / (len(trainloader) * epochs)

    # Get privacy spent
    epsilon = privacy_engine.get_epsilon(delta=target_delta)

    return avg_loss, epsilon, 0.0, privacy_engine


def test(net, testloader, device):
    """Evaluate model on test set."""
    net = net.to(device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            # Count correct predictions (handles variable batch sizes)
            correct += int((predicted == labels).sum())

    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader)

    return avg_loss, accuracy


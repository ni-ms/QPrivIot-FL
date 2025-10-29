"""Model, data loading, training, and testing - FINAL REVIEWED VERSION."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale, Resize
from typing import Tuple, Optional, Dict
from collections import OrderedDict
import numpy as np


class Net(nn.Module):
    """Simple CNN for CIFAR-10"""

    def __init__(self):
        super().__init__()
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


class FEMNISTNet(nn.Module):
    """CNN for FEMNIST (28x28 grayscale, 62 classes)"""

    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class IoTSensorNet(nn.Module):
    """Simple MLP for IoT sensor data (simulated)"""

    def __init__(self, input_dim=10, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def make_model(dataset_name: str = "cifar10") -> nn.Module:
    """Factory function to create appropriate model for dataset."""
    if dataset_name == "cifar10":
        return Net()
    elif dataset_name == "femnist":
        return FEMNISTNet()
    elif dataset_name == "iot":
        return IoTSensorNet()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_data(
        partition_id: int,
        num_partitions: int,
        batch_size: int = 32,
        dataset_name: str = "cifar10"
) -> Tuple[DataLoader, DataLoader]:
    """Load partitioned data for FL."""
    if dataset_name == "cifar10":
        return load_cifar10(partition_id, num_partitions, batch_size)
    elif dataset_name == "femnist":
        return load_femnist(partition_id, num_partitions, batch_size)
    elif dataset_name == "iot":
        return load_iot_simulated(partition_id, num_partitions, batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_cifar10(partition_id: int, num_partitions: int, batch_size: int):
    """Load CIFAR-10 partition."""
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    partition_split = partition.train_test_split(test_size=0.2, seed=42)

    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        batch["img"] = [transforms(img) for img in batch["img"]]
        return batch

    partition_split = partition_split.with_transform(apply_transforms)
    trainloader = DataLoader(partition_split["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_split["test"], batch_size=batch_size)

    return trainloader, testloader


def load_femnist(partition_id: int, num_partitions: int, batch_size: int):
    """Load FEMNIST partition."""
    try:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="flwrlabs/femnist",
            partitioners={"train": partitioner},
        )
        partition = fds.load_partition(partition_id)
        partition_split = partition.train_test_split(test_size=0.2, seed=42)

        transforms = Compose([Resize((28, 28)), Grayscale(), ToTensor(), Normalize((0.5,), (0.5,))])

        def apply_transforms(batch):
            batch["image"] = [transforms(img) for img in batch["image"]]
            return batch

        partition_split = partition_split.with_transform(apply_transforms)
        trainloader = DataLoader(partition_split["train"], batch_size=batch_size, shuffle=True)
        testloader = DataLoader(partition_split["test"], batch_size=batch_size)

        return trainloader, testloader
    except Exception as e:
        print(f"⚠️ FEMNIST loading failed: {e}. Using CIFAR-10 as fallback.")
        return load_cifar10(partition_id, num_partitions, batch_size)


def load_iot_simulated(partition_id: int, num_partitions: int, batch_size: int):
    """Simulated IoT sensor data (placeholder)."""
    np.random.seed(partition_id)
    n_samples = 1000
    X = np.random.randn(n_samples, 10).astype(np.float32)
    y = np.random.randint(0, 3, n_samples).astype(np.int64)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    return trainloader, testloader


def get_weights(model: nn.Module) -> list:
    """Extract model weights as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: list):
    """Set model weights from numpy arrays."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        epochs: int,
        learning_rate: float,
        device: torch.device,
        dp_config: Optional[Dict] = None,
        dataset_name: str = "cifar10"
) -> Dict:
    """
    Train model with optional DP.
    
    Args:
        model: PyTorch model
        trainloader: Training data loader
        valloader: Validation data loader
        epochs: Number of local epochs
        learning_rate: Learning rate
        device: Device (CPU/GPU)
        dp_config: DP configuration (None for no DP)
        dataset_name: Dataset name for data extraction
    
    Returns:
        Dictionary with training results
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    privacy_engine = None
    epsilon = 0.0
    dp_attached = False

    if dp_config:
        from qpriviot_fl.privacy import attach_dp_to_optimizer
        try:
            print(f"🔧 Attaching DP: noise={dp_config['noise_multiplier']:.2f}, "
                  f"clip={dp_config['max_grad_norm']:.2f}")

            model, optimizer, trainloader, privacy_engine = attach_dp_to_optimizer(
                model, optimizer, trainloader,
                dp_config["noise_multiplier"],
                dp_config["max_grad_norm"],
                device
            )
            dp_attached = True
            print("✅ DP attached successfully")

        except Exception as e:
            print(f"❌ DP attachment failed: {e}")
            import traceback
            traceback.print_exc()
            dp_config = None

    model.train()
    total_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(trainloader):

            if dataset_name == "iot":
                images, labels = batch
            else:
                images = batch["img"] if "img" in batch else batch["image"]
                labels = batch["label"]

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(trainloader)
        total_loss += avg_epoch_loss
        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_epoch_loss:.4f}")

    avg_train_loss = total_loss / epochs

    val_loss, val_acc = test(model, valloader, device, dataset_name)

    if privacy_engine and dp_attached:
        try:
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            print(f"✅ Privacy spent: ε={epsilon:.2f} (δ=1e-5)")
        except Exception as e:
            print(f"⚠️ Could not compute epsilon: {e}")
            epsilon = 0.0

    print(f"📊 Results: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
          f"val_acc={val_acc:.2%}, ε={epsilon:.2f}")

    return {
        "train_loss": float(avg_train_loss),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "epsilon": float(epsilon),
        "dp_attached": dp_attached,
    }


def test(
        model: nn.Module,
        testloader: DataLoader,
        device: torch.device,
        dataset_name: str = "cifar10"
) -> Tuple[float, float]:
    """Evaluate model on test set."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct, total_loss = 0, 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in testloader:

            if dataset_name == "iot":
                images, labels = batch
            else:
                images = batch["img"] if "img" in batch else batch["image"]
                labels = batch["label"]

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = torch.max(outputs, 1)[1]
            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0

    return avg_loss, accuracy


class ConvergenceTracker:
    """Track training convergence for adaptive privacy scheduling."""

    def __init__(self, window_size: int = 5, threshold: float = 0.01):
        """
        Args:
            window_size: Number of recent losses to consider
            threshold: Loss change threshold for convergence
        """
        self.window_size = window_size
        self.threshold = threshold
        self.loss_history = []

    def update(self, loss: float):
        """Add new loss value."""
        self.loss_history.append(float(loss))
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)

    def get_convergence_score(self) -> float:
        """
        Compute convergence score [0, 1].
        1 = fully converged, 0 = not converged.
        """
        if len(self.loss_history) < 2:
            return 0.0

        recent_losses = self.loss_history[-self.window_size:]
        loss_change = abs(recent_losses[-1] - recent_losses[0])

        convergence = 1.0 - min(1.0, loss_change / self.threshold)

        return convergence

    def is_converged(self) -> bool:
        """Check if training has converged."""
        return self.get_convergence_score() > 0.8
